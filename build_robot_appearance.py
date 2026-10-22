"""Extract tracklet appearance samples locally, without reading team OCR labels."""
import os,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,torch
from torchvision.models import resnet18,ResNet18_Weights

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME',str(ROOT/'runs/torch-cache'))


if __name__=='__main__':
    torch.set_num_threads(8)
    active=json.loads((ROOT/'runs/active_toggle_pilot.json').read_text());target=Path(active['directory'])
    prior=Path(json.loads((ROOT/'runs/active_robot_pilot.json').read_text())['directory'])
    summary=json.loads((prior/'render-summary.json').read_text(encoding='utf-8'));width,height=summary['width'],summary['height']
    frames=[];tracks=defaultdict(list)
    with (prior/'robot-detections.jsonl').open() as f:
        for line in f:
            row=json.loads(line);frames.append(row)
            for obj in row['detections']:
                if obj['track_id'] is not None:tracks[obj['track_id']].append((row['frame'],obj))
    samples=defaultdict(list)
    for track,rows in tracks.items():
        for index in np.unique(np.linspace(0,len(rows)-1,min(12,len(rows))).astype(int)):
            frame,obj=rows[index];samples[frame].append((track,obj))
    model=resnet18(weights=ResNet18_Weights.DEFAULT);model.fc=torch.nn.Identity();model.eval()
    cap=cv2.VideoCapture(summary['source']);batch=[];keys=[];embeddings=[];identifiers=[]
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    def flush():
        if not batch:return
        with torch.inference_mode():
            output=model((torch.tensor(np.stack(batch)).permute(0,3,1,2).float()/255-mean)/std)
            output=torch.nn.functional.normalize(output,dim=1).numpy()
        embeddings.extend(output);identifiers.extend(keys);batch.clear();keys.clear()
    try:
        for frame in range(len(frames)):
            if frame not in samples:cap.grab();continue
            ok,image=cap.read()
            if not ok:raise ValueError('Source decode failed')
            for track,obj in samples[frame]:
                x1,y1,x2,y2=map(round,obj['xyxy']);crop=image[max(0,y1):min(height,y2),max(0,x1):min(width,x2)]
                if crop.size==0:continue
                h,w=crop.shape[:2];side=max(h,w);square=np.zeros((side,side,3),np.uint8)
                square[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=crop
                batch.append(cv2.cvtColor(cv2.resize(square,(224,224)),cv2.COLOR_BGR2RGB));keys.append((track,frame))
                if len(batch)>=24:flush()
            if frame%300<30:print('APPEARANCE',frame,flush=True)
        flush()
    finally:cap.release()
    np.savez_compressed(target/'appearance.npz',features=np.array(embeddings),keys=np.array(identifiers))
    seeds={1:'471A',2:'1744X',3:'2899A',4:'88S',8:'88S'}
    features=np.array(embeddings);keys=np.array(identifiers);scores={}
    for track in tracks:
        current=features[keys[:,0]==track];row={}
        for team in ('1744X','471A','88S','2899A'):
            reference=features[np.array([seeds.get(int(k[0]))==team and int(k[0])!=track for k in keys])]
            if not len(reference):reference=features[np.array([seeds.get(int(k[0]))==team for k in keys])]
            sim=current@reference.T
            row[team]=float(np.mean(np.sort(sim,axis=1)[:,-min(3,len(reference)):]))
        scores[str(track)]={'scores':row,'start':tracks[track][0][0],'end':tracks[track][-1][0],'frames':len(tracks[track])}
    (target/'appearance-scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    print(json.dumps(scores),flush=True)
