"""Compare a 960px robot fine-tune on the existing temporal pilot split.

This experiment does not train team identity or Goal scoring. Original labels,
accepted replay files and default weights are never overwritten.
"""
import json
import os
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    active=json.loads((ROOT/'runs/active_robot_pilot.json').read_text(encoding='utf-8'))
    source=Path(active['weights'])
    config=Path(active['directory'])/'dataset/dataset.yaml'
    if not source.is_file() or not config.is_file():
        raise ValueError('Existing robot pilot weights / dataset are missing')
    target=ROOT/'runs'/('robot-recall-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    target.mkdir(parents=True)
    pointer=ROOT/'runs/latest_robot_recall.json'
    state=dict(directory=str(target),status='validating_baseline',source_weights=str(source),
               scope=['robot'],validation='same-video temporal holdout; previously used; not independent-match accuracy',
               limitations='No team identity labels, no Goal/Pin annotations; cannot train identity or scoring')
    def save():
        payload=json.dumps(state,ensure_ascii=False,indent=2)
        (target/'summary.json').write_text(payload,encoding='utf-8')
        temporary=pointer.with_suffix('.tmp');temporary.write_text(payload,encoding='utf-8');temporary.replace(pointer)
    def metrics(result):
        return dict(precision=float(result.box.mp),recall=float(result.box.mr),map50=float(result.box.map50),map50_95=float(result.box.map))
    save()
    try:
        validation=dict(data=str(config),imgsz=960,batch=2,device='cpu',workers=0,
                        project=str(target),plots=False,verbose=False)
        model=YOLO(str(source))
        state['baseline']=metrics(model.val(**validation,name='baseline'))
        state['status']='training';save()
        print('BASELINE '+json.dumps(state['baseline']),flush=True)
        model.train(data=str(config),epochs=40,imgsz=960,batch=2,device='cpu',workers=0,
                    project=str(target),name='model',seed=42,patience=12,plots=True,amp=False,
                    mosaic=0,close_mosaic=0,fliplr=0,flipud=0,degrees=3,scale=.15,translate=.06,
                    hsv_h=.01,hsv_s=.2,hsv_v=.2,cache=False,optimizer='AdamW',lr0=.0003,cos_lr=True)
        weights=Path(model.trainer.best)
        state.update(status='validating_candidate',weights=str(weights));save()
        state['candidate']=metrics(YOLO(str(weights)).val(**validation,name='candidate'))
        before,after=state['baseline'],state['candidate']
        recommended=(after['recall']>before['recall'] and after['map50']>=before['map50'] and after['precision']>=before['precision']-.03)
        state.update(status='complete',recommended_on_pilot=recommended,
                     adoption='Candidate kept separately; full-match identity and missed-detection review still required')
        save();print('TRAINING_COMPLETE '+json.dumps(state,ensure_ascii=False),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
