"""Small fixed-view Goal/Pin/Cup crop experiment; not an automatic scorer."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default=str(ROOT/'yolo11n.pt'))
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--curation',help='Local assistant/user curation JSON; not published with code')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    from prepare_goal_pilot import prepare
    target,config=prepare(args.curation)
    pointer=ROOT/'runs/active_goal_pilot.json'
    state=json.loads(pointer.read_text());state.update(status='training',initial_weights=args.base,epochs_requested=args.epochs);pointer.write_text(json.dumps(state))
    try:
        model=YOLO(args.base)
        model.train(data=str(config),epochs=args.epochs,imgsz=320,batch=16,device='cpu',workers=0,
                    project=str(target),name='specialist',seed=42,patience=15,plots=True,amp=False,
                    mosaic=0,close_mosaic=0,fliplr=0,flipud=0,scale=.2,translate=.1,
                    hsv_h=0,hsv_s=.15,hsv_v=.2,cache=False,optimizer='AdamW',lr0=.0003 if args.base!=str(ROOT/'yolo11n.pt') else .001,cos_lr=True)
        weights=str(model.trainer.best)
        result=YOLO(weights).val(data=str(config),imgsz=320,batch=16,device='cpu',workers=0,
                               project=str(target),name='validation',plots=True,verbose=False)
        state.update(status='complete',weights=weights,metrics=dict(precision=float(result.box.mp),recall=float(result.box.mr),map50=float(result.box.map50)),
                     limitations='assistant partial-region labels; same-video crop holdout; no Placed / nesting / scoring validity')
        (target/'training-summary.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        pointer.write_text(json.dumps(state,indent=2),encoding='utf-8')
        print('GOAL_PILOT_COMPLETE '+json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));pointer.write_text(json.dumps(state));raise
