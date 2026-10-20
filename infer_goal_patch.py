"""Apply a supplementary specialist only to selected calibrated Goal columns."""
import argparse
import json
from pathlib import Path
import cv2
from field_specialist import Specialist
from goal_observer import observe_goal_stacks,intersection_fraction

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--goals',nargs='+',required=True)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(8);cv2.setNumThreads(2)
    state=json.loads((ROOT/'runs/active_goal_pilot.json').read_text())
    training=json.loads((ROOT/'runs/latest_stack_training.json').read_text())
    if training['status']!='complete':raise ValueError('Training incomplete')
    target=Path(training['directory']);model=Specialist(training['weights'],max_estimate_seconds=1.15)
    model.goals={g:model.goals[g] for g in args.goals}
    normalized={g:[b[0]/2048,b[1]/1152,b[2]/2048,b[3]/1152] for g,b in model.goals.items()}
    def in_column(obj):
        box=obj['xyxy'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        for base in normalized.values():
            if obj['class_id']==6 and intersection_fraction(base,box)>.2:return True
            gw=base[2]-base[0];gh=base[3]-base[1];gx=(base[0]+base[2])/2
            if abs(cx-gx)<gw*1.5 and base[1]-gh*5<cy<base[1]+gh*.6:return True
        return False
    cap=cv2.VideoCapture(state['source']);fps=cap.get(5);total=round(cap.get(7));written=0
    output=target/'patched-field-detections.jsonl'
    try:
        with Path(state['preview_jsonl']).open() as source,output.open('x') as log:
            for index,line in enumerate(source):
                row=json.loads(line);ok,frame=cap.read()
                if not ok or row['frame']!=index:raise ValueError('Video/records mismatch')
                image=cv2.resize(frame,(2048,1152))
                objects=model.detect(image,row['seconds'],row['robots']) if index%round(fps)==0 else model.propagate(image,row['seconds'])
                # This local supplementary training covers Goal/yellow/Cup.
                classes=(6,9,10)
                kept=[o for o in row['field_objects'] if o['class_id'] not in classes or not in_column(o)]
                kept += [o for o in objects if o['class_id'] in classes and in_column(o)]
                row['field_objects']=kept;row['goal_rois'].update(normalized)
                row['goal_estimate']=observe_goal_stacks(row)
                log.write(json.dumps(row)+'\n');written+=1
                if index%600==0:print(f'GOAL_PATCH {index}/{total}',flush=True)
        if written!=total:raise ValueError('Incomplete Goal patch')
        report=dict(status='complete',weights=training['weights'],directory=str(target),records=str(output),
                    source=state['source'],frames=total,goals=args.goals,base_records=state['preview_jsonl'],
                    inference='learned specialist plus current-pixel yellow/neutral connector proposals; unconfirmed')
        (ROOT/'runs/latest_goal_patch.json').write_text(json.dumps(report,indent=2))
        print('GOAL_PATCH_COMPLETE',flush=True)
    finally:cap.release()


if __name__=='__main__':main()
