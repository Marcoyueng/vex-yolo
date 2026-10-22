"""Small fixed-view Toggle ROI color classifier. No scoring-validity inference."""
import cv2
import numpy as np
import torch
from torch import nn

SITES=('left','top','right','bottom')
LABELS=('red','blue','yellow')


def features(image,box,site):
    h,w=image.shape[:2]
    x1,y1,x2,y2=[round(n*d) for n,d in zip(box,[w,h,w,h])]
    crop=image[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    if crop.size==0:raise ValueError('Empty toggle ROI')
    hsv=cv2.cvtColor(cv2.resize(crop,(128,64)),cv2.COLOR_BGR2HSV)
    parts=[]
    for row in range(2):
        for col in range(4):
            block=hsv[row*32:(row+1)*32,col*32:(col+1)*32]
            mask=(block[:,:,1]>75)&(block[:,:,2]>45)
            hist=np.histogram(block[:,:,0][mask],bins=18,range=(0,180))[0].astype(np.float32)/1024
            parts.extend(hist.tolist());parts.extend([float(block[:,:,1].mean()/255),float(block[:,:,2].mean()/255)])
    parts.extend(float(site==s) for s in SITES)
    return np.array(parts,dtype=np.float32)


class ColorNet(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(164,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,x):return self.network(x)


class ColorReader:
    def __init__(self,path):
        self.model=ColorNet();self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));self.model.eval()
        self.face_readers={}
    def set_face_geometry(self,calibrations):
        """Reset per-video exterior readers; do not reuse one camera's mapping."""
        from toggle_geometry import FrontFaceReader
        self.face_readers={site:FrontFaceReader(c['exterior_roi'],c['assembly']) for site,c in calibrations.items()}
    def read(self,image,rois):
        keys=list(rois)
        data=torch.tensor(np.stack([features(image,rois[key],key) for key in keys]))
        with torch.no_grad():probabilities=self.model(data).softmax(1).numpy()
        result={key:{'color':LABELS[int(np.argmax(p))] if float(np.max(p))>=.65 else 'unknown',
                     'confidence':float(np.max(p)),'scoring_validity':'unconfirmed',
                     'method':'fixed_view_roi_color_classifier'} for key,p in zip(keys,probabilities)}
        for site,reader in self.face_readers.items():
            if site in result:
                original=result[site]
                result[site]=dict(reader.read(image),original_model_state=original)
        return result
