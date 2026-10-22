"""Continue the tiny robot pilot with restrained augmentation."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')

if __name__=='__main__':
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    active=json.loads((ROOT/'runs/active_robot_pilot.json').read_text(encoding='utf-8'))
    target=Path(active['directory'])
    model=YOLO(active['weights'])
    model.train(data=str(target/'dataset/dataset.yaml'),epochs=80,imgsz=640,batch=4,device='cpu',workers=0,
                project=str(target/'refinement'),name='robot',seed=42,patience=0,plots=True,amp=False,
                mosaic=0,close_mosaic=0,fliplr=0,scale=.1,translate=.05,cache=False,
                optimizer='AdamW',lr0=.001,cos_lr=True)
    active['weights']=str(model.trainer.best)
    (ROOT/'runs/active_robot_pilot.json').write_text(json.dumps(active),encoding='utf-8')
    print('REFINEMENT_COMPLETE '+active['weights'],flush=True)
