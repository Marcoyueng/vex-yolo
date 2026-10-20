"""Only Goal-connected proposals are drawn; loose floor objects are hidden."""
import cv2
from overlay_style import draw,caption
from field_specialist import PALETTE
from goal_observer import observe_goal_stacks


def goal_view(frame,row,toggle_rois,goal_rois=None):
    observation=observe_goal_stacks(row,goal_rois)
    rois=goal_rois if goal_rois is not None else row.get('goal_rois',{})
    image=draw(frame,row.get('robots',[]),row.get('toggles',{}),toggle_rois)
    h,w=image.shape[:2]
    for goal,box in rois.items():
        pixel=[round(v*d) for v,d in zip(box,[w,h,w,h])]
        color=(80,240,110) if observation['goals'][goal]['status']=='estimate' else (160,160,160)
        cv2.rectangle(image,tuple(pixel[:2]),tuple(pixel[2:]),color,1)
        caption(image,goal,(pixel[0],pixel[3]+18),color,.5)
    for obj in observation['objects']:
        pixel=[round(v*d) for v,d in zip(obj['xyxy'],[w,h,w,h])];cls=obj['class_id'];color=PALETTE[cls]
        cv2.rectangle(image,tuple(pixel[:2]),tuple(pixel[2:]),color,2)
        label=obj['goal_id']+'/'+{7:'R',8:'B',9:'Y',10:'Cup'}[cls]
        if obj.get('observation')=='optical_flow_estimate':label+=' ~'
        caption(image,label,(pixel[0],pixel[1]-4),color,.45)
    return image,observation
