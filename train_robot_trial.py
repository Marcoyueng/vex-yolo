"""Robot-only pilot using reviewed labels and an explicit temporal holdout.

This is not independent-event validation. Original labels are never modified.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def prepare():
    from schema import SCHEMA, CLASSES
    from datetime import datetime
    target=ROOT/'runs'/('robot-pilot-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    target.mkdir(parents=True)
    labels=ROOT/'data/annotations-v3'
    if json.loads((labels/'classes.json').read_text(encoding='utf-8'))!=SCHEMA:
        raise ValueError('Wrong annotation schema')
    manifests={}
    for path in (ROOT/'data/unlabelled').glob('*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);manifests[row['image']]=row
    samples=[];seen=set();duplicates=[]
    for label in sorted(labels.glob('*.txt')):
        image=ROOT/'data/unlabelled'/(label.stem+'.jpg')
        if not image.is_file() or image.name not in manifests:continue
        rows=[line for line in label.read_text().splitlines() if line.split()[0]=='0']
        if not rows:continue
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen:duplicates.append(image.name);continue
        seen.add(digest)
        samples.append({'image':image,'label':label,'rows':rows,**manifests[image.name]})
        samples[-1]['image_path']=image
    samples.sort(key=lambda x:x['seconds'])
    if len(samples)<10:raise ValueError('Too few reviewed samples for pilot')
    # Both imported prefixes reference the same recording. Never pretend these
    # are independent source videos; reserve a final block and a 20-second gap.
    validation=samples[-max(4,round(len(samples)*.2)):]
    cutoff=validation[0]['seconds']-20
    training=[s for s in samples if s['seconds']<cutoff]
    if len(training)<5:raise ValueError('Temporal holdout leaves too few training images')
    for split,items in [('train',training),('val',validation)]:
        for kind in ('images','labels'):(target/'dataset'/kind/split).mkdir(parents=True)
        for sample in items:
            image=sample['image_path']
            shutil.copy2(image,target/'dataset/images'/split/image.name)
            (target/'dataset/labels'/split/(image.stem+'.txt')).write_text('\n'.join(sample['rows'])+'\n',encoding='utf-8')
    config=target/'dataset/dataset.yaml'
    config.write_text('path: '+json.dumps((target/'dataset').as_posix())+'\ntrain: images/train\nval: images/val\nnames:\n'
                      +''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)),encoding='utf-8')
    info={'scope':['robot'],'validation':'same-recording temporal holdout; NOT independent-event accuracy',
          'gap_seconds':20,'train_images':[s['image_path'].name for s in training],
          'validation_images':[s['image_path'].name for s in validation],
          'excluded_near_boundary':[s['image_path'].name for s in samples if s not in training and s not in validation],
          'duplicates_removed':duplicates,'sources':sorted({r['source'] for r in manifests.values() if 'starting is ez' in r['source']})}
    (target/'split.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'runs/active_robot_pilot.json').write_text(json.dumps({'directory':str(target)},ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'directory':str(target),'train':len(training),'val':len(validation),'duplicates':len(duplicates)}),flush=True)
    return target,config


if __name__=='__main__':
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    target,config=prepare()
    model=YOLO('yolo11n.pt')
    model.train(data=str(config),epochs=50,imgsz=640,batch=4,device='cpu',workers=0,
                project=str(target),name='model',seed=42,patience=15,plots=True,amp=False,
                mosaic=.5,close_mosaic=10,fliplr=.5,cache=False)
    weights=Path(model.trainer.best)
    (ROOT/'runs/active_robot_pilot.json').write_text(json.dumps({'directory':str(target),'weights':str(weights)}),encoding='utf-8')
    print('PILOT_COMPLETE '+str(weights),flush=True)
