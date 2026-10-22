"""Video-specific opening-position alliance seeds, with conservative ID repair.

This is motion association, not team-number recognition or permanent ReID.
"""
from math import hypot


def center(box):
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)


class OpeningAllianceTracker:
    def __init__(self, anchors, fps):
        self.fps=fps
        self.slots={key:dict(value,track_id=None,box=None,last_frame=-1) for key,value in anchors.items()}
        self.initialized=False

    def update(self, detections, frame):
        """Input xyxy normalized to 0..1; unknown associations remain unlabelled."""
        result=[dict(obj,alliance=None,team=None,opening_slot=None,identity_source='unconfirmed') for obj in detections]
        if not self.initialized:
            # Seed only at the opening; never classify by later screen position.
            if frame>self.fps*2:return result
            options=[]
            for key,slot in self.slots.items():
                for i,obj in enumerate(detections):
                    if obj.get('track_id') is not None:
                        cx,cy=center(obj['xyxy'])
                        distance=hypot(cx-slot['anchor'][0],cy-slot['anchor'][1])
                        if distance<.10:options.append((distance,key,i))
            assigned={};used=set()
            for _,key,i in sorted(options):
                if key not in assigned and i not in used:assigned[key]=i;used.add(i)
            if len(assigned)!=4:return result
            self.initialized=True
        else:
            assigned={};used=set()
            # Existing IDs are preferred, but large jumps and long gaps are rejected.
            for key,slot in self.slots.items():
                if frame-slot['last_frame']>self.fps*2:continue
                for i,obj in enumerate(detections):
                    if i in used or obj.get('track_id') is None:continue
                    if obj['track_id']==slot['track_id'] and hypot(*(a-b for a,b in zip(center(obj['xyxy']),center(slot['box']))))<.12:
                        assigned[key]=i;used.add(i);break
            # Reconnect a newly issued ID only when a nearby match is unambiguous.
            candidates=[]
            for key,slot in self.slots.items():
                if key in assigned or frame-slot['last_frame']>self.fps:continue
                for i,obj in enumerate(detections):
                    if i in used or obj.get('track_id') is None:continue
                    distance=hypot(*(a-b for a,b in zip(center(obj['xyxy']),center(slot['box']))))
                    if distance<.045:candidates.append((distance,key,i))
            for distance,key,i in sorted(candidates):
                if key in assigned or i in used:continue
                alternatives=[d for d,k,j in candidates if (k==key and j!=i) or (j==i and k!=key)]
                if alternatives and min(alternatives)-distance<.025:continue
                assigned[key]=i;used.add(i)
        for key,i in assigned.items():
            slot=self.slots[key];slot.update(track_id=detections[i]['track_id'],box=detections[i]['xyxy'],last_frame=frame)
            result[i].update(alliance=slot['alliance'],team=slot.get('team'),opening_slot=key,identity_source='opening_position_motion_tracking')
        return result
