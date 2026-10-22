"""Offline, one-team-per-distinct-robot tracklet association for this recording."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint

ROOT=Path(__file__).resolve().parent
TEAMS=('1744X','471A','88S','2899A')


def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


if __name__=='__main__':
    target=Path(json.loads((ROOT/'runs/active_toggle_pilot.json').read_text())['directory'])
    prior=Path(json.loads((ROOT/'runs/active_robot_pilot.json').read_text())['directory'])
    scores=json.loads((target/'appearance-scores.json').read_text());tracks={int(k):v for k,v in scores.items()}
    dimensions=json.loads((prior/'render-summary.json').read_text(encoding='utf-8'));w,h=dimensions['width'],dimensions['height']
    records=[];positions=defaultdict(list);apart=defaultdict(int);shared=defaultdict(int)
    with (prior/'robot-detections.jsonl').open() as f:
        for line in f:
            record=json.loads(line);objects=[o for o in record['detections'] if o['track_id'] is not None]
            for obj in objects:positions[obj['track_id']].append((record['frame'],obj['xyxy']))
            for i,a in enumerate(objects):
                for b in objects[i+1:]:
                    key=tuple(sorted((a['track_id'],b['track_id'])));shared[key]+=1
                    ac=np.array([(a['xyxy'][0]+a['xyxy'][2])/2/w,(a['xyxy'][1]+a['xyxy'][3])/2/h])
                    bc=np.array([(b['xyxy'][0]+b['xyxy'][2])/2/w,(b['xyxy'][1]+b['xyxy'][3])/2/h])
                    if np.linalg.norm(ac-bc)>.055 and iou(a['xyxy'],b['xyxy'])<.4:apart[key]+=1
    ids=sorted(tracks);index={track:i for i,track in enumerate(ids)};n=len(ids)*5
    cost=np.zeros(n);lower=np.zeros(n);upper=np.ones(n);rows=[];lb=[];ub=[]
    def constraint(values,minimum,maximum):
        row=np.zeros(n)
        for key,value in values.items():row[key]=value
        rows.append(row);lb.append(minimum);ub.append(maximum)
    seeds={1:'471A',2:'1744X',3:'2899A',4:'88S',8:'88S'}
    for track in ids:
        start=index[track]*5;weight=max(.1,min(10,tracks[track]['frames']/100))
        best=max(tracks[track]['scores'].values())
        for j,team in enumerate(TEAMS):cost[start+j]=(best-tracks[track]['scores'][team])*6*weight
        cost[start+4]=(2 if tracks[track]['frames']>=100 else .65)*weight
        constraint({start+j:1 for j in range(5)},1,1)
        if track in seeds:lower[start+TEAMS.index(seeds[track])]=1
    # Short spatially continuous transitions, reviewed against their endpoints.
    # Some overlapping IDs are duplicate partial boxes for the same machine.
    links=[(8,40),(3,42),(42,50),(26,52),(31,34),(34,37),(37,44),(15,26)]
    for a,b in links:
        for team in range(5):constraint({index[a]*5+team:1,index[b]*5+team:-1},0,0)
    conflicts=[]
    for pair,count in apart.items():
        if pair in links or tuple(reversed(pair)) in links:continue
        if count>=min(8,max(2,round(min(tracks[pair[0]]['frames'],tracks[pair[1]]['frames'])*.1))):
            conflicts.append(pair)
            for team in range(4):constraint({index[pair[0]]*5+team:1,index[pair[1]]*5+team:1},0,1)
    matrix=np.array(rows);linear=LinearConstraint(matrix,np.array(lb),np.array(ub))
    solution=milp(cost,integrality=np.ones(n),bounds=Bounds(lower,upper),constraints=linear,options={'time_limit':30})
    if solution.x is None:raise ValueError('Association constraints are inconsistent')
    mapping={}
    for track in ids:
        selected=int(np.argmax(solution.x[index[track]*5:index[track]*5+5]));team=TEAMS[selected] if selected<4 else None
        margin=None
        if selected<4 and track not in seeds and tracks[track]['frames']>=30:
            alternative_upper=upper.copy();alternative_upper[index[track]*5+selected]=0
            alt=milp(cost,integrality=np.ones(n),bounds=Bounds(lower,alternative_upper),constraints=linear,options={'time_limit':5})
            margin=float(alt.fun-solution.fun) if alt.x is not None else 99.0
        mapping[str(track)]={'team':team,'alliance':'red' if team in ('1744X','471A') else 'blue' if team else None,
                             'method':'opening_seed' if track in seeds else 'offline_appearance_and_cooccurrence',
                             'alternative_cost_margin':margin,**tracks[track]}
    (target/'identity-associations.json').write_text(json.dumps({'mapping':mapping,'conflicts':conflicts,'motion_links':links,
          'limitations':'not OCR; recording-specific seed/appearance/one-robot-per-team associations require visual review'},indent=2),encoding='utf-8')
    for track,entry in mapping.items():
        if entry['frames']>=30:print(track,entry['team'],entry['start'],entry['end'],'margin',entry['alternative_cost_margin'])
