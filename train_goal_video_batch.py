"""Cumulative Goal specialist stages using reviewed local crop inputs."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--epochs',type=int,default=5)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    from prepare_stack_refinement import prepare
    videos=json.loads(Path(args.manifest).read_text(encoding='utf-8'))['videos']
    accepted=json.loads((ROOT/'runs/active_goal_pilot.json').read_text());base=accepted['directory'];weights=accepted['weights']
    state=dict(status='training',stages=[],validation='Previous reviewed crop holdout; new videos are not yet independently validated')
    pointer=ROOT/'runs/latest_elims_goal_training.json'
    def save():pointer.write_text(json.dumps(state,indent=2))
    save()
    try:
        for video in videos:
            target,config=prepare(video['goal_review'],base)
            entry=dict(video=video['id'],directory=str(target),status='training',initial_weights=weights)
            state['stages'].append(entry);save();print('BATCH_GOAL_START '+video['id'],flush=True)
            model=YOLO(weights)
            model.train(data=str(config),epochs=args.epochs,imgsz=320,batch=16,device='cpu',workers=0,
                        project=str(target),name='specialist',seed=42,patience=0,plots=True,amp=False,
                        mosaic=0,close_mosaic=0,fliplr=0,scale=.1,translate=.05,hsv_h=0,hsv_s=.12,hsv_v=.15,
                        cache=False,optimizer='AdamW',lr0=.0002,cos_lr=True)
            weights=str(model.trainer.best);base=str(target)
            Path(weights).with_suffix('.scope.json').write_text(json.dumps(dict(scope=['goal','pin_half_red','pin_half_blue','pin_half_yellow','cup'],inference='Calibrated crop specialist; per-video calibration required',scoring_validity='unconfirmed')))
            entry.update(status='complete',weights=weights,epochs_completed=int(model.trainer.epoch)+1,
                         metrics={k:float(v) for k,v in model.trainer.metrics.items()});save()
            print('BATCH_GOAL_COMPLETE '+video['id'],flush=True)
        state.update(status='complete',weights=weights);save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
