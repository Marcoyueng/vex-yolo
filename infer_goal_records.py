"""Rerun a trained stack specialist on a video, preserving accepted identities.

Writes a new local record file. The accepted model is not changed by this job.
"""
import argparse
import json
import os
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',required=True)
    parser.add_argument('--interval',type=float,default=1.)
    args=parser.parse_args()
    if not 0<args.interval<=2:raise ValueError('Sampling interval must be in (0, 2] seconds')
    import torch
    torch.set_num_threads(8);cv2.setNumThreads(2)
    from field_specialist import Specialist
    from goal_observer import observe_goal_stacks
    accepted=json.loads((ROOT/'runs/active_goal_pilot.json').read_text(encoding='utf-8'))
    target=Path(args.weights).resolve().parents[2]
    output=target/'field-detections.jsonl'
    specialist=Specialist(args.weights,max_estimate_seconds=args.interval+.15)
    rois={g:[b[0]/2048,b[1]/1152,b[2]/2048,b[3]/1152] for g,b in specialist.goals.items()}
    cap=cv2.VideoCapture(accepted['source']);fps=cap.get(5);total=round(cap.get(7))
    if not cap.isOpened() or fps<=0:raise ValueError('Source video unavailable')
    pointer=ROOT/'runs/latest_stack_inference.json'
    state=dict(status='running',weights=args.weights,directory=str(target),source=accepted['source'],
               input_records=accepted['preview_jsonl'],records=str(output),frames=total,interval_seconds=args.interval)
    def save():pointer.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save();written=0
    try:
        with Path(accepted['preview_jsonl']).open(encoding='utf-8') as source,output.open('x',encoding='utf-8') as log:
            for index,line in enumerate(source):
                row=json.loads(line);ok,frame=cap.read()
                if not ok or row['frame']!=index:raise ValueError('Video/records mismatch')
                image=cv2.resize(frame,(2048,1152))
                if index%max(1,round(fps*args.interval))==0:
                    objects=specialist.detect(image,row['seconds'],row.get('robots',[]))
                else:objects=specialist.propagate(image,row['seconds'])
                row.update(field_objects=objects,goal_rois=rois,field_scoring_validity='unconfirmed')
                row['goal_estimate']=observe_goal_stacks(row)
                log.write(json.dumps(row)+'\n');written+=1
                if index%300==0:print(f'STACK_INFERENCE {index}/{total}',flush=True)
        if written!=total:raise ValueError('Incomplete inference')
        state.update(status='complete',written_frames=written);save()
        (target/'inference-summary.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    except BaseException as error:
        state.update(status='failed',error=str(error),written_frames=written);save();raise
    finally:cap.release()


if __name__=='__main__':main()
