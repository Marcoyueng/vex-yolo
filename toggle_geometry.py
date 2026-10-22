"""Infer an inward Toggle face from a calibrated exterior-face strip.

The official starting orientations differ on red and blue field quadrants.
This estimates color only: seating and robot contact remain unverified.
"""
import cv2
import numpy as np

EXTERIOR_TO_INTERIOR={
    'red':{'red':'yellow','yellow':'blue','blue':'red'},
    'blue':{'blue':'yellow','yellow':'red','red':'blue'},
}


def inward_color(exterior,assembly):
    return EXTERIOR_TO_INTERIOR[assembly].get(exterior,'unknown')


def exterior_face(image,roi):
    h,w=image.shape[:2]
    x1,y1,x2,y2=[round(a*b) for a,b in zip(roi,[w,h,w,h])]
    crop=image[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    if min(crop.shape[:2],default=0)<5:return dict(color='unknown',coverage=0.)
    hue,sat,val=cv2.split(cv2.cvtColor(crop,cv2.COLOR_BGR2HSV))
    masks=np.stack([((hue<12)|(hue>168))&(sat>100)&(val>45),
        (hue>85)&(hue<130)&(sat>100)&(val>45),
        (hue>18)&(hue<42)&(sat>100)&(val>70)])
    # In this calibrated bottom-wall view, the exterior face is the lowest
    # substantial horizontal color band. The upper face must not win by area.
    widths=masks.mean(axis=2)
    candidates=[]
    for c in range(3):
        valid=widths[c]>=.60
        start=None
        for y in range(len(valid)+1):
            if y<len(valid) and valid[y]:
                if start is None:start=y
            elif start is not None:
                if y-start>=4:candidates.append((y-1,start,c))
                start=None
    if not candidates:return dict(color='unknown',coverage=0.)
    end,start,c=max(candidates)
    band=masks[c,max(start,end-5):end+1]
    # Require horizontal evidence across the bar, not a nearby colored object.
    segment_coverage=[float(a.mean()) for a in np.array_split(band,5,axis=1)]
    if sum(v>=.5 for v in segment_coverage)<4:return dict(color='unknown',coverage=float(band.mean()))
    return dict(color=('red','blue','yellow')[c],coverage=float(band.mean()),
                band_xyxy=[x1,y1+max(start,end-5),x2,y1+end+1])


class FrontFaceReader:
    def __init__(self,roi,assembly,confirmations=3):
        self.roi=roi;self.assembly=assembly;self.confirmations=confirmations
        self.candidate=None;self.count=0

    def read(self,image):
        observed=exterior_face(image,self.roi);c=observed['color']
        self.count=self.count+1 if c==self.candidate else 1
        self.candidate=c
        stable=c!='unknown' and self.count>=self.confirmations
        return dict(color=inward_color(c,self.assembly) if stable else 'unknown',
            exterior_color=c,exterior_coverage=observed['coverage'],
            exterior_band_xyxy=observed.get('band_xyxy'),geometry_color_stable=stable,
            confidence=None,scoring_validity='unconfirmed',
            method='calibrated_exterior_face_to_interior_geometry',assembly=self.assembly)
