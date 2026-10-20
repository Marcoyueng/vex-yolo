"""Train a stack specialist candidate without replacing the accepted model."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review',required=True)
    parser.add_argument('--epochs',type=int,default=35)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    from prepare_stack_refinement import prepare
    accepted=json.loads((ROOT/'runs/active_goal_pilot.json').read_text(encoding='utf-8'))
    target,config=prepare(args.review,accepted['directory'])
    pointer=ROOT/'runs/latest_stack_training.json'
    state=dict(directory=str(target),status='training',initial_weights=accepted['weights'],epochs_requested=args.epochs,
               validation='Same-video reviewed crops only; not independent match validation')
    def save():pointer.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        model=YOLO(accepted['weights'])
        model.train(data=str(config),epochs=args.epochs,imgsz=320,batch=16,device='cpu',workers=0,
                    project=str(target),name='specialist',seed=42,patience=12,plots=True,amp=False,
                    mosaic=0,close_mosaic=0,fliplr=0,flipud=0,scale=.12,translate=.06,
                    hsv_h=0,hsv_s=.12,hsv_v=.2,cache=False,optimizer='AdamW',lr0=.0003,cos_lr=True)
        state.update(status='complete',weights=str(model.trainer.best),
                     metrics={key:float(value) for key,value in model.trainer.metrics.items()})
        (target/'training-summary.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        save();print('STACK_TRAINING_COMPLETE '+json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
