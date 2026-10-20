"""One cumulative robot fine-tuning stage per locally reviewed video.

Only explicit reviewed boxes are labels. Raw automatic detections are not
silently reused as truth. Each video retains a held-out reviewed frame.
"""
import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
import cv2
from schema import CLASSES,SCHEMA

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review',required=True);parser.add_argument('--epochs',type=int,default=15)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(8);cv2.setNumThreads(2)
    from ultralytics import YOLO
    review=json.loads(Path(args.review).read_text(encoding='utf-8'))
    accepted=json.loads((ROOT/'runs/active_robot_pilot.json').read_text())
    base_dataset=Path(accepted['directory'])/'dataset';weights=accepted['weights']
    target=ROOT/'runs'/('video-batch-training-'+datetime.now().strftime('%Y%m%d-%H%M%S'));target.mkdir()
    pointer=ROOT/'runs/latest_elims_training.json'
    state=dict(status='training',directory=str(target),review=str(Path(args.review).resolve()),stages=[],
               limitations='Assistant-reviewed small robot-only samples; per-video temporal holdout, not independent match accuracy')
    def save():pointer.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        for number,video in enumerate(review['videos'],1):
            vid=video['id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+',vid):raise ValueError('Invalid local video ID')
            stage=target/f'{number:02d}-{vid}';dataset=stage/'dataset';shutil.copytree(base_dataset,dataset)
            samples=[s for s in review['samples'] if s['video']==vid]
            for sample in samples:
                image=cv2.imread(sample['image'])
                if image is None or list(image.shape[1::-1])!=sample['coordinate_size']:raise ValueError('Reviewed sample unavailable or resized')
                h,w=image.shape[:2];split=sample['split'];name=f'{vid}__{sample["frame"]:08d}'
                if split not in ('train','val'):raise ValueError('Invalid split')
                cv2.imwrite(str(dataset/'images'/split/(name+'.jpg')),image)
                labels=[]
                for obj in sample['objects']:
                    if obj['class_id']!=0:continue
                    a,b,c,d=obj['xyxy']
                    if not 0<=a<c<=w or not 0<=b<d<=h:raise ValueError('Invalid reviewed box')
                    labels.append(f'0 {(a+c)/2/w:.7f} {(b+d)/2/h:.7f} {(c-a)/w:.7f} {(d-b)/h:.7f}')
                (dataset/'labels'/split/(name+'.txt')).write_text('\n'.join(labels)+'\n')
            config=dataset/'dataset.yaml'
            config.write_text('path: '+json.dumps(dataset.as_posix())+'\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)),encoding='utf-8')
            (dataset/'classes.json').write_text(json.dumps(SCHEMA))
            entry=dict(video=video,directory=str(stage),status='training',initial_weights=str(weights),reviewed_frames=len(samples))
            state['stages'].append(entry);save();print('VIDEO_TRAIN_START '+vid,flush=True)
            model=YOLO(weights)
            model.train(data=str(config),epochs=args.epochs,imgsz=640,batch=4,device='cpu',workers=0,
                        project=str(stage),name='robot',seed=42,patience=0,plots=True,amp=False,
                        mosaic=0,close_mosaic=0,fliplr=0,scale=.1,translate=.05,cache=False,
                        optimizer='AdamW',lr0=.0003,cos_lr=True)
            weights=str(model.trainer.best);base_dataset=dataset
            Path(weights).with_suffix('.scope.json').write_text(json.dumps(dict(scope=['robot'],source='assistant-reviewed cumulative local-video training')))
            entry.update(status='complete',weights=weights,epochs_completed=int(model.trainer.epoch)+1,
                         metrics={k:float(v) for k,v in model.trainer.metrics.items()});save()
            (stage/'training-summary.json').write_text(json.dumps(entry,indent=2),encoding='utf-8')
            print('VIDEO_TRAIN_COMPLETE '+vid,flush=True)
        state.update(status='complete',weights=weights);save()
        (target/'summary.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
