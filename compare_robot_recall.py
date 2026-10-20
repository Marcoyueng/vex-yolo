"""Compare old/new detectors at the workbench's fixed confidence threshold.

Small, previously used, same-video holdout. This is a pilot comparison only.
No automatic promotion: increased box recall does not validate team identity.
"""
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','8')
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def main():
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    state=json.loads((ROOT/'runs/latest_robot_recall.json').read_text(encoding='utf-8'))
    if state['status']!='complete':raise ValueError('Training has not completed')
    previous=json.loads((ROOT/'runs/active_robot_pilot.json').read_text(encoding='utf-8'))
    dataset=Path(previous['directory'])/'dataset'
    directory=Path(state['directory'])
    report=dict(validation='same-video temporal holdout, previously used; not independent-match accuracy',
                confidence=.35,iou=.5,experiments={})
    for name,weights in [('old',state['source_weights']),('new',state['weights'])]:
        model=YOLO(weights)
        for size in (640,960):
            tp=fp=fn=0;frames=[]
            for image in sorted((dataset/'images/val').glob('*.jpg')):
                result=model.predict(str(image),imgsz=size,conf=.35,device='cpu',classes=[0],verbose=False)[0]
                targets=[]
                for line in (dataset/'labels/val'/image.with_suffix('.txt').name).read_text().splitlines():
                    cls,x,y,w,h=map(float,line.split())
                    if cls==0:targets.append([x-w/2,y-h/2,x+w/2,y+h/2])
                boxes=result.boxes.xyxyn.cpu().tolist();scores=result.boxes.conf.cpu().tolist()
                matched=set();hits=0
                for index in sorted(range(len(boxes)),key=lambda j:scores[j],reverse=True):
                    options=[(iou(boxes[index],target),j) for j,target in enumerate(targets) if j not in matched]
                    overlap,j=max(options,default=(0,-1))
                    if overlap>=.5:matched.add(j);hits+=1
                tp+=hits;fp+=len(boxes)-hits;fn+=len(targets)-hits
                frames.append(dict(image=image.name,truth=len(targets),detected=len(boxes),matched=hits))
                if name=='new' and size==960:
                    preview=directory/'validation-previews';preview.mkdir(exist_ok=True)
                    result.save(filename=str(preview/image.name))
            report['experiments'][f'{name}_{size}']=dict(weights=weights,imgsz=size,true_positive=tp,false_positive=fp,
                false_negative=fn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,frames=frames)
    (directory/'fixed-threshold-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    baseline=report['experiments']['old_640'];candidate=report['experiments']['new_960']
    state['improved_max_f1_summary']=state.pop('recommended_on_pilot',False)
    state['recommended_on_pilot']=(candidate['recall']>baseline['recall'] and candidate['precision']>=baseline['precision']-.03)
    state['fixed_threshold_comparison']=str(directory/'fixed-threshold-comparison.json')
    state['adoption']='Not promoted: candidate has more misses at confidence 0.35; keep the existing model' if not state['recommended_on_pilot'] else 'Candidate needs full-match review before adoption'
    payload=json.dumps(state,ensure_ascii=False,indent=2)
    (directory/'summary.json').write_text(payload,encoding='utf-8')
    pointer=ROOT/'runs/latest_robot_recall.json';temporary=pointer.with_suffix('.tmp')
    temporary.write_text(payload,encoding='utf-8');temporary.replace(pointer)
    print(json.dumps({name:{k:v for k,v in row.items() if k not in ('frames','weights')} for name,row in report['experiments'].items()},indent=2),flush=True)


if __name__=='__main__':main()
