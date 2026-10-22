import json
import os
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
from render_toggle_pilot import calibration
from toggle_color import features,ColorNet,SITES

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    torch.set_num_threads(4);torch.manual_seed(42)
    active=json.loads((ROOT/'runs/active_toggle_pilot.json').read_text());target=Path(active['directory'])
    split=json.loads((target/'color-data-split.json').read_text(encoding='utf-8'));rois=calibration(target,split)
    (target/'color-rois.json').write_text(json.dumps(rois,indent=2),encoding='utf-8')
    data={};counts={}
    for phase,key in [('train','train_images'),('val','validation_images')]:
        x=[];y=[]
        for name in split[key]:
            image=cv2.imread(str(ROOT/'data/unlabelled'/name))
            observed={}
            for line in (ROOT/'data/annotations-v3'/(Path(name).stem+'.txt')).read_text().splitlines():
                cls,cx,cy,w,h=map(float,line.split())
                if int(cls) not in (2,3,4):continue
                site=min(rois,key=lambda s:((rois[s][0]+rois[s][2])/2-cx)**2+((rois[s][1]+rois[s][3])/2-cy)**2)
                if site in observed:raise ValueError('Conflicting Toggle labels in an image')
                observed[site]=int(cls)-2
            for site,label in observed.items():x.append(features(image,rois[site],site));y.append(label)
        data[phase]=(torch.tensor(np.stack(x)),torch.tensor(y));counts[phase]=dict(Counter(y))
    model=ColorNet();optimizer=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.02)
    count=torch.bincount(data['train'][1],minlength=3).float();criterion=torch.nn.CrossEntropyLoss(weight=count.sum()/count.clamp(min=1))
    best=float('inf');best_state=None
    for epoch in range(1200):
        model.train();optimizer.zero_grad();loss=criterion(model(data['train'][0]),data['train'][1]);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            logits=model(data['val'][0]);validation=float(torch.nn.functional.cross_entropy(logits,data['val'][1]))
        if validation<best:best=validation;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        predictions=model(data['val'][0]).argmax(1);correct=int((predictions==data['val'][1]).sum());total=len(predictions)
    torch.save(model.state_dict(),target/'toggle-color-classifier.pt')
    report={'train_counts':counts['train'],'validation_counts':counts['val'],'correct':correct,'total':total,
            'validation':'same-video middle temporal holdout with 20-second buffers; not independent-event accuracy',
            'method':'calibrated fixed-view spatial HSV histograms and small MLP'}
    (target/'color-classifier-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
