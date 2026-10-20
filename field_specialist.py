"""Fixed-view proposal inference for the small Goal/Pin/Cup crop specialist.

Output is draft detection, never Placed/scoring evidence. Goal calibration is
separate from observed boxes. Short optical-flow propagation is marked estimated.
"""
import cv2
import numpy as np

NAMES={6:'goal',7:'pin_half_red',8:'pin_half_blue',9:'pin_half_yellow',10:'cup'}
PALETTE={6:(80,240,110),7:(60,60,255),8:(255,170,40),9:(40,240,250),10:(230,130,230)}


def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def suppress(objects,threshold=.35):
    kept=[]
    for obj in sorted(objects,key=lambda o:o['confidence'],reverse=True):
        if not any(obj['class_id']==other['class_id'] and overlap(obj['xyxy'],other['xyxy'])>threshold for other in kept):
            kept.append(obj)
    return kept


def field_mask(shape,polygon):
    mask=np.zeros(shape[:2],np.uint8)
    cv2.fillPoly(mask,[np.array(polygon,dtype=np.int32)],255)
    return mask


class Specialist:
    def __init__(self,weights,max_estimate_seconds=1.1,calibration=None):
        import os
        import json
        from pathlib import Path
        os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parent/'runs/config'))
        from ultralytics import YOLO
        if calibration is None:
            calibration=json.loads((Path(weights).resolve().parents[2]/'manifest.json').read_text(encoding='utf-8'))
        self.goals=calibration['goals']
        self.field_polygon=calibration['field_polygon']
        self.model=YOLO(str(weights))
        self.previous=None
        self.objects=[]
        self.sample_seconds=None
        self.max_estimate_seconds=max_estimate_seconds

    def detect(self,image,seconds,robots):
        """Input is 2048x1152 review coordinates; output uses normalized boxes."""
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        # A tall stack projects above/outside the floor polygon in perspective.
        # Floor geometry must not clip its upper halves.
        # Search only above calibrated Goal mouths. Contact with a robot does
        # not by itself negate Placed; uncertainty is handled by goal_observer.
        stack_mask=np.zeros(image.shape[:2],np.uint8)
        stack_windows=[]
        for goal in self.goals.values():
            gx=(goal[0]+goal[2])/2;gw=goal[2]-goal[0];gh=goal[3]-goal[1]
            left,right=max(0,round(gx-gw*4)),min(2047,round(gx+gw*4))
            top,bottom=max(0,round(goal[1]-gh*9)),min(1151,round(goal[1]+gh*.3))
            cv2.rectangle(stack_mask,(left,top),(right,bottom),255,-1)
            for y in range(top+40,max(top+41,bottom),80):
                for dx in (-1.6,0,1.6):
                    stack_windows.append((round(gx+dx*gw),y,144))
        mask=stack_mask
        colored=(cv2.inRange(hsv,(0,120,75),(10,255,255))|
                 cv2.inRange(hsv,(170,120,75),(179,255,255))|
                 cv2.inRange(hsv,(86,130,90),(115,255,255))|
                 cv2.inRange(hsv,(20,120,100),(40,255,255))) & mask
        _,_,stats,_=cv2.connectedComponentsWithStats(colored)
        centers=[]
        goal_centers=[((b[0]+b[2])/2,(b[1]+b[3])/2) for b in self.goals.values()]
        for x,y,w,h,area in stats[1:]:
            if 55<=area<=6000 and 5<=w<=110 and 6<=h<=120:
                cx,cy=int(x+w/2),int(y+h/2)
                if any((cx-gx)**2+(cy-gy)**2<45**2 for gx,gy in goal_centers):continue
                if any((cx-px)**2+(cy-py)**2<35**2 for px,py,_ in centers):continue
                centers.append((cx,cy,112))
        centers.extend((int((b[0]+b[2])/2),int((b[1]+b[3])/2),144) for b in self.goals.values())
        # Gray Cup proposals also stay inside Goal stack search areas.
        centers.extend(stack_windows)
        windows=[]
        for cx,cy,side in centers:
            left=max(0,min(2048-side,cx-side//2));top=max(0,min(1152-side,cy-side//2))
            box=(left,top,left+side,top+side)
            if box not in windows:windows.append(box)
        crops=[image[y:y2,x:x2] for x,y,x2,y2 in windows]
        results=self.model.predict(crops,imgsz=320,conf=.5,iou=.45,device='cpu',batch=16,
                                   classes=list(NAMES),verbose=False)
        objects=[]
        for window,result in zip(windows,results):
            ox,oy,x2,y2=window
            for box,cls,confidence in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                cls=int(cls);a,b,c,d=box
                global_box=[(ox+a)/2048,(oy+b)/1152,(ox+c)/2048,(oy+d)/1152]
                cx,cy=round((ox+a+ox+c)/2),round((oy+b+oy+d)/2)
                if min(c-a,d-b)<5:continue
                if cls in (7,8,9) and d-b<.35*(c-a):continue # thin Goal rims are not protruding halves
                if cls==6:
                    # A learned detection must overlap a separately reviewed Goal ROI.
                    if not any(overlap(global_box,[g[0]/2048,g[1]/1152,g[2]/2048,g[3]/1152])>.2 for g in self.goals.values()):continue
                elif not mask[max(0,min(1151,cy)),max(0,min(2047,cx))]:continue
                objects.append(dict(class_id=cls,**{'class':NAMES[cls]},xyxy=global_box,confidence=confidence,
                                    observation='draft_model_detection',sample_seconds=seconds,scoring_validity='unconfirmed'))
        self.objects=suppress(objects)
        self.previous=cv2.cvtColor(cv2.resize(image,(1024,576)),cv2.COLOR_BGR2GRAY)
        self.sample_seconds=seconds
        return self.objects

    def propagate(self,image,seconds):
        current=cv2.cvtColor(cv2.resize(image,(1024,576)),cv2.COLOR_BGR2GRAY)
        if self.previous is None or self.sample_seconds is None or seconds-self.sample_seconds>getattr(self,'max_estimate_seconds',1.1):
            self.previous=current;self.objects=[];return []
        active=[o for o in self.objects if o['class_id']!=6]
        union_mask=np.zeros(current.shape,np.uint8)
        pixel_boxes=[]
        for obj in active:
            x,y,x2,y2=[round(v*d) for v,d in zip(obj['xyxy'],[1024,576,1024,576])]
            x,y=max(0,x),max(0,y);x2,y2=min(1024,x2),min(576,y2)
            pixel_boxes.append((x,y,x2,y2))
            if x2>x and y2>y:union_mask[y:y2,x:x2]=255
        corners=cv2.goodFeaturesToTrack(self.previous,maxCorners=1200,qualityLevel=.03,minDistance=2,mask=union_mask) if active else None
        all_points=corners[:,0,:] if corners is not None else np.empty((0,2),dtype=np.float32)
        points=[];ranges=[]
        for x,y,x2,y2 in pixel_boxes:
            selected=all_points[(all_points[:,0]>=x)&(all_points[:,0]<x2)&(all_points[:,1]>=y)&(all_points[:,1]<y2)][:10]
            start=len(points)
            points.extend(selected.tolist())
            ranges.append((start,len(points)))
        kept=[dict(o,observation='previous_goal_detection') for o in self.objects if o['class_id']==6]
        if points:
            points=np.array(points,dtype=np.float32).reshape(-1,1,2)
            moved,status,_=cv2.calcOpticalFlowPyrLK(self.previous,current,points,None,winSize=(15,15),maxLevel=2)
            back,back_status,_=cv2.calcOpticalFlowPyrLK(current,self.previous,moved,None,winSize=(15,15),maxLevel=2)
            for obj,(start,end) in zip(active,ranges):
                good=(status[start:end,0]==1)&(back_status[start:end,0]==1)&(np.linalg.norm(back[start:end,0]-points[start:end,0],axis=1)<1.5)
                if good.sum()<2:continue
                delta=np.median(moved[start:end,0][good]-points[start:end,0][good],axis=0)
                if np.linalg.norm(delta)>30:continue
                x,y,x2,y2=obj['xyxy'];dx,dy=delta/[1024,576]
                box=[x+dx,y+dy,x2+dx,y2+dy]
                if min(box)<0 or max(box)>1:continue
                kept.append(dict(obj,xyxy=[float(v) for v in box],observation='optical_flow_estimate'))
        self.previous=current;self.objects=kept
        return kept
