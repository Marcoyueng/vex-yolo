"""Local opening-box and appearance association, with explicit unknown teams.

Roster, seed coordinates and media are local inputs. This is not team-plate OCR.
Conflicting or weak appearance matches keep their track ID without inventing a
team number. A fresh tracker/roster is used for every video.
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME',str(ROOT/'runs/torch-cache'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);args=parser.parse_args()
    import torch
    from torchvision.models import resnet18,ResNet18_Weights
    from scipy.optimize import milp,Bounds,LinearConstraint,linear_sum_assignment
    from field_specialist import overlap
    torch.set_num_threads(8);cv2.setNumThreads(2)
    net=resnet18(weights=ResNet18_Weights.DEFAULT);net.fc=torch.nn.Identity();net.eval()
    videos=json.loads(Path(args.manifest).read_text(encoding='utf-8'))['videos']
    pointer=ROOT/'runs/latest_elims_association.json';state=dict(status='running',videos=[])
    def save():pointer.write_text(json.dumps(state,indent=2))
    save()
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    def crop(image,box):
        h,w=image.shape[:2];a,b,c,d=[round(v*z) for v,z in zip(box,[w,h,w,h])]
        region=image[max(0,b):min(h,d),max(0,a):min(w,c)]
        if not region.size:return np.zeros((224,224,3),np.uint8)
        rh,rw=region.shape[:2];side=max(rh,rw);square=np.zeros((side,side,3),np.uint8)
        square[(side-rh)//2:(side-rh)//2+rh,(side-rw)//2:(side-rw)//2+rw]=region
        return cv2.cvtColor(cv2.resize(square,(224,224)),cv2.COLOR_BGR2RGB)
    def embed(images):
        result=[]
        for first in range(0,len(images),24):
            x=torch.tensor(np.stack(images[first:first+24])).permute(0,3,1,2).float()/255
            with torch.inference_mode():features=net((x-mean)/std);features=torch.nn.functional.normalize(features,dim=1)
            result.extend(features.numpy())
        return np.array(result)
    try:
        for video in videos:
            while True:
                try:source_state=json.loads((ROOT/'runs/latest_elims_field_inference.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
                if source_state['status']=='failed':raise ValueError('Field inference failed')
                item=next((x for x in source_state['videos'] if x['id']==video['id'] and x['status']=='complete'),None)
                if item:break
                time.sleep(1)
            rows=[json.loads(line) for line in Path(item['records']).open()];tracks=defaultdict(list);apart=defaultdict(int)
            for row in rows:
                for obj in row['robots']:tracks[obj['track_id']].append((row['frame'],obj))
                for i,a in enumerate(row['robots']):
                    for b in row['robots'][i+1:]:
                        ac=np.array([(a['xyxy'][0]+a['xyxy'][2])/2,(a['xyxy'][1]+a['xyxy'][3])/2]);bc=np.array([(b['xyxy'][0]+b['xyxy'][2])/2,(b['xyxy'][1]+b['xyxy'][3])/2])
                        if np.linalg.norm(ac-bc)>.055 and overlap(a['xyxy'],b['xyxy'])<.4:apart[tuple(sorted((a['track_id'],b['track_id'])))]+=1
            wanted=defaultdict(list)
            for tid,observations in tracks.items():
                for j in np.unique(np.linspace(0,len(observations)-1,min(8,len(observations))).astype(int)):
                    frame,obj=observations[j];wanted[frame].append((tid,obj))
            for slot,seed in enumerate(video['identity_seeds']):wanted[seed['frame']].append(('seed'+str(slot),seed))
            capture=cv2.VideoCapture(video['source']);images=[];keys=[]
            for frame in range(video['frames']):
                if frame not in wanted:capture.grab();continue
                ok,image=capture.read()
                if not ok:raise ValueError('Appearance source decode failed')
                for key,obj in wanted[frame]:
                    box=obj['xyxy']
                    if isinstance(key,str):box=[v/z for v,z in zip(box,obj['coordinate_size']*2)]
                    images.append(crop(image,box));keys.append(key)
            capture.release();features=embed(images)
            ref=[features[[j for j,k in enumerate(keys) if k=='seed'+str(i)]][0] for i in range(4)]
            ids=sorted(tracks);index={tid:i for i,tid in enumerate(ids)};scores={}
            for tid in ids:
                current=features[[j for j,k in enumerate(keys) if k==tid]];scores[tid]=current@np.array(ref).T
                scores[tid]=scores[tid].mean(0)
            fixed={}
            for frame in sorted({s['frame'] for s in video['identity_seeds']}):
                seeds=[(i,s) for i,s in enumerate(video['identity_seeds']) if s['frame']==frame];observed=rows[frame]['robots']
                if not observed:continue
                matrix=np.array([[overlap([v/z for v,z in zip(s['xyxy'],s['coordinate_size']*2)],o['xyxy']) for o in observed] for _,s in seeds])
                a,b=linear_sum_assignment(-matrix)
                for aa,bb in zip(a,b):
                    if matrix[aa,bb]>=.35:fixed[observed[bb]['track_id']]=seeds[aa][0]
            size=len(ids)*5;cost=np.zeros(size);lower=np.zeros(size);upper=np.ones(size);constraints=[];lb=[];ub=[]
            for tid in ids:
                start=index[tid]*5;weight=max(.2,min(6,len(tracks[tid])/60))
                cost[start:start+4]=(1-scores[tid])*6*weight;cost[start+4]=1.8*weight
                row=np.zeros(size);row[start:start+5]=1;constraints.append(row);lb.append(1);ub.append(1)
                if tid in fixed:lower[start+fixed[tid]]=1
            for (a,b),count in apart.items():
                if count<3:continue
                for slot in range(4):
                    row=np.zeros(size);row[index[a]*5+slot]=1;row[index[b]*5+slot]=1;constraints.append(row);lb.append(0);ub.append(1)
            solved=milp(cost,integrality=np.ones(size),bounds=Bounds(lower,upper),constraints=LinearConstraint(np.array(constraints),lb,ub),options={'time_limit':20})
            if solved.x is None:raise ValueError('Identity seed/co-occurrence conflict; review required')
            mapping={}
            for tid in ids:
                slot=int(np.argmax(solved.x[index[tid]*5:index[tid]*5+5]));score=scores[tid]
                margin=float(score[slot]-max(np.delete(score,slot))) if slot<4 else 0
                valid=slot<4 and (tid in fixed or (score[slot]>=.72 and margin>=.025))
                seed=video['identity_seeds'][slot] if valid else {}
                mapping[tid]=dict(team=seed.get('team'),alliance=seed.get('alliance'),
                                  identity_source='reviewed_opening_box' if tid in fixed else 'appearance_association' if valid else 'unresolved_track',
                                  appearance_margin=margin,slot=slot if valid else None)
            target=ROOT/'runs/elims-batch'/video['id'];output=target/'match-records.jsonl';known=0;total=0
            with output.open('x') as log:
                for row in rows:
                    row['robots']=[dict(o,**mapping[o['track_id']]) for o in row['robots']]
                    known+=sum(bool(o['team']) for o in row['robots']);total+=len(row['robots'])
                    log.write(json.dumps(row)+'\n')
            (target/'identity-associations.json').write_text(json.dumps(dict(mapping=mapping,limitations='Appearance and reviewed opening boxes, not OCR; unresolved teams remain unset'),indent=2))
            entry=dict(id=video['id'],source=video['source'],records=str(output),status='complete',frames=len(rows),known_team_boxes=known,total_robot_boxes=total)
            state['videos'].append(entry);save();print('BATCH_ASSOCIATION_COMPLETE '+video['id'],flush=True)
        state.update(status='complete');save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
