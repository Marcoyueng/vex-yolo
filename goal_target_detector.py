"""Opt-in, Goal-conditioned Pin detector. Not the general object detector.

The three classes mean visible Pin halves on the requested Goal, not all Pins
in the image. Counts remain provisional. Calibrated ROIs are input conditions,
never drawn as learned Goal detections. This experiment is not enabled in app.
"""
import math

TARGET_NAMES=('goal_pin_red','goal_pin_blue','goal_pin_yellow')
GENERAL_CLASS_IDS=(7,8,9)


def target_window(goal,shape,width_factor=2.6,height_factor=8.5,dx=0,dy=0):
    """Normalized Goal ROI -> native image crop, including above-floor stacks."""
    height,width=shape[:2]
    a,b,c,d=[v*s for v,s in zip(goal,[width,height,width,height])]
    gw,gh=c-a,d-b
    if gw<=0 or gh<=0:raise ValueError('Goal ROI must have positive area')
    cx=(a+c)/2+dx*gw
    box=(max(0,math.floor(cx-width_factor*gw)),max(0,math.floor(b-height_factor*gh+dy*gh)),
         min(width,math.ceil(cx+width_factor*gw)),min(height,math.ceil(d+.5*gh+dy*gh)))
    if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Goal ROI lies outside image')
    return box


class GoalTargetDetector:
    def __init__(self,weights):
        from ultralytics import YOLO
        self.model=YOLO(str(weights))
        if tuple(self.model.names.values())!=TARGET_NAMES:
            raise ValueError('Requires the separate three-class Goal-conditioned weights')

    def detect(self,image,goal_rois,seconds=0,confidence=.25):
        h,w=image.shape[:2]
        windows={gid:target_window(box,image.shape) for gid,box in goal_rois.items()}
        crops=[image[y:y2,x:x2] for x,y,x2,y2 in windows.values()]
        if not crops:return {}
        results=self.model.predict(crops,imgsz=640,conf=confidence,iou=.45,device='cpu',batch=4,verbose=False)
        output={}
        for (gid,window),result in zip(windows.items(),results):
            x,y,_,_=window;objects=[]
            for box,cls,conf in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                cls=int(cls);a,b,c,d=box
                objects.append(dict(class_id=GENERAL_CLASS_IDS[cls],target_class=TARGET_NAMES[cls],
                    xyxy=[(x+a)/w,(y+b)/h,(x+c)/w,(y+d)/h],confidence=float(conf),goal_id=gid,
                    observation='goal_conditioned_model_detection',scoring_validity='unconfirmed',sample_seconds=seconds))
            output[gid]=objects
        return output
