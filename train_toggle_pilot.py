"""Extend the robot pilot with reviewed Toggle visual-color labels."""
import json
import os
from pathlib import Path
import shutil
from collections import Counter
from datetime import datetime

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


if __name__=='__main__':
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    from schema import CLASSES,SCHEMA
    previous=json.loads((ROOT/'runs/active_robot_pilot.json').read_text(encoding='utf-8'))
    split=json.loads((Path(previous['directory'])/'split.json').read_text(encoding='utf-8'))
    labels=ROOT/'data/annotations-v3'
    assert json.loads((labels/'classes.json').read_text(encoding='utf-8'))==SCHEMA
    target=ROOT/'runs'/('robot-toggle-pilot-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    counts={}
    for phase,key in [('train','train_images'),('val','validation_images')]:
        for kind in ('images','labels'):(target/'dataset'/kind/phase).mkdir(parents=True)
        count=Counter()
        for name in split[key]:
            image=ROOT/'data/unlabelled'/name
            label=labels/(image.stem+'.txt')
            rows=[line for line in label.read_text().splitlines() if int(line.split()[0]) in (0,2,3,4)]
            count.update(int(line.split()[0]) for line in rows)
            shutil.copy2(image,target/'dataset/images'/phase/name)
            (target/'dataset/labels'/phase/label.name).write_text('\n'.join(rows)+'\n',encoding='utf-8')
        counts[phase]=dict(count)
    for cls in (0,2,3,4):
        if not counts['train'].get(cls):raise ValueError(f'No training examples for {CLASSES[cls]}')
    split.update(scope=['robot','toggle_red','toggle_blue','toggle_yellow'],object_counts=counts,
                 limitations='Visual colors only; SC4 settled and contact conditions are not inferred')
    (target/'split.json').write_text(json.dumps(split,indent=2,ensure_ascii=False),encoding='utf-8')
    config=target/'dataset/dataset.yaml'
    config.write_text('path: '+json.dumps((target/'dataset').as_posix())+'\ntrain: images/train\nval: images/val\nnames:\n'
                      +''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)),encoding='utf-8')
    (ROOT/'runs/active_toggle_pilot.json').write_text(json.dumps({'directory':str(target)}),encoding='utf-8')
    print(json.dumps({'directory':str(target),'counts':counts}),flush=True)
    model=YOLO(previous['weights'])
    model.train(data=str(config),epochs=80,imgsz=768,batch=4,device='cpu',workers=0,
                project=str(target),name='robot-toggle',seed=42,patience=25,plots=True,amp=False,
                mosaic=0,close_mosaic=0,fliplr=0,scale=.1,translate=.05,cache=False,
                hsv_h=0,hsv_s=.2,hsv_v=.2,optimizer='AdamW',lr0=.001,cos_lr=True)
    active={'directory':str(target),'weights':str(model.trainer.best),'robot_weights':previous['weights']}
    (ROOT/'runs/active_toggle_pilot.json').write_text(json.dumps(active),encoding='utf-8')
    print('TOGGLE_TRAINING_COMPLETE '+active['weights'],flush=True)
