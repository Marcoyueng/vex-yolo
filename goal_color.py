"""Narrow current-pixel corroboration for an incompletely detected stack.

Requires a learned Goal and an upper yellow half. Yellow connected components
and dark neutral connectors are proposals, not proof of Cup identity or Placed.
"""
import cv2
import numpy as np


def corroborate_yellow(image,goals,objects,seconds):
    from goal_observer import rooted,intersection_fraction
    h,w=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);extra=[]
    for base in goals.values():
        gx=(base[0]+base[2])/2;gw=base[2]-base[0];gh=base[3]-base[1]
        normalized=[base[0]/w,base[1]/h,base[2]/w,base[3]/h]
        observed=any(o['class_id']==6 and intersection_fraction(normalized,o['xyxy'])>.25 for o in objects)
        learned=[o for o in objects if o['class_id']==9 and o['confidence']>=.6 and abs((o['xyxy'][0]+o['xyxy'][2])/2*w-gx)<gw*.7]
        if not observed or not learned or any(rooted(o['xyxy'],normalized) for o in learned):continue
        x1,x2=max(0,round(gx-gw*.7)),min(w,round(gx+gw*.7))
        y1,y2=max(0,round(base[1]-gh*4)),min(h,round(base[1]))
        crop=hsv[y1:y2,x1:x2]
        mask=cv2.inRange(crop,(20,110,100),(40,255,255))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        heads=[]
        for x,y,bw,bh,area in stats[1:]:
            if area<12 or bw<5 or bh<4 or bw>gw*1.2 or bh>gh*1.3 or bh<bw*.25:continue
            heads.append([x+x1,y+y1,x+x1+bw,y+y1+bh])
        heads.sort(key=lambda b:(b[1]+b[3])/2,reverse=True)
        if len(heads)<2:continue
        # Only bridge short, actually gray intervals up from this Goal mouth.
        lower=[gx,base[1],gx,base[1]];connectors=[];valid=[]
        for head in heads:
            cx=(head[0]+head[2])/2;px=(lower[0]+lower[2])/2
            gap=lower[1]-head[3];hh=head[3]-head[1]
            if not 0<=gap<=max(hh,gh*.45)*1.8 or abs(cx-px)>max(head[2]-head[0],gw*.4):break
            if gap>=3:
                half=max(3,round((head[2]-head[0])*.3));mid=(cx+px)/2
                box=[max(0,round(mid-half)),round(head[3]),min(w,round(mid+half)),round(lower[1]+2)]
                patch=hsv[box[1]:box[3],box[0]:box[2]]
                gray=(patch[:,:,1]<160)&(patch[:,:,2]<190)&(patch[:,:,2]>18)
                if not gray.size or gray.mean()<.4:break
                connectors.append(box)
            valid.append(head);lower=head
        if len(valid)<2:continue
        for cls,boxes in ((9,valid),(10,connectors)):
            for box in boxes:
                nb=[box[0]/w,box[1]/h,box[2]/w,box[3]/h]
                if any(o['class_id']==cls and max(intersection_fraction(nb,o['xyxy']),intersection_fraction(o['xyxy'],nb))>.6 for o in objects):continue
                extra.append(dict(class_id=cls,**{'class':'pin_half_yellow' if cls==9 else 'cup'},xyxy=nb,
                                  confidence=.5,confidence_kind='heuristic',observation='color_connector_estimate',
                                  sample_seconds=seconds,scoring_validity='unconfirmed'))
    return extra
