"""Image-verified display geometry, separate from learned Pin detections.

This fixed-camera replay aid uses a reviewed stack patch. If its current
appearance no longer agrees, no box is drawn. Counts are retained separately.
"""
import cv2
import numpy as np


class ReviewedPinOverlay:
    def __init__(self,config):
        self.config=config
        image=cv2.imdecode(np.fromfile(config['image'],dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Missing reviewed Pin source image')
        x1,y1,x2,y2=config['context_box']
        self.patch=image[y1:y2,x1:x2].copy()

    def observe(self,image,frame):
        if frame<self.config['start_frame']:return []
        x1,y1,x2,y2=self.config['context_box'];radius=self.config.get('search_radius',4)
        left,top=max(0,x1-radius),max(0,y1-radius)
        search=image[top:min(image.shape[0],y2+radius),left:min(image.shape[1],x2+radius)]
        if search.shape[0]<self.patch.shape[0] or search.shape[1]<self.patch.shape[1]:return []
        response=cv2.matchTemplate(search,self.patch,cv2.TM_CCOEFF_NORMED)
        _,agreement,_,point=cv2.minMaxLoc(response)
        if agreement<self.config.get('minimum_agreement',.87):return []
        px,py=point;current=search[py:py+self.patch.shape[0],px:px+self.patch.shape[1]]
        if np.abs(current.astype(float)-self.patch.astype(float)).mean()>self.config.get('maximum_mean_error',30):return []
        dx,dy=left+px-x1,top+py-y1
        return [dict(goal_id=self.config['goal'],class_id=part['class_id'],
            xyxy=[(part['box'][0]+dx)/image.shape[1],(part['box'][1]+dy)/image.shape[0],
                  (part['box'][2]+dx)/image.shape[1],(part['box'][3]+dy)/image.shape[0]],
            geometry_source='reviewed_template_current_pixel_agreement',
            geometry_review_status='assistant_drawn_for_user_confirmed_pin',
            evidence_frame=self.config['start_frame'],template_agreement=round(agreement,4)) for part in self.config['parts']]
