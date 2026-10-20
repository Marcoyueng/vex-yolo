"""Calibrated per-video field observations, reusing complete robot records."""
import argparse
import json
import os
import time
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--interval',type=float,default=2.)
    args=parser.parse_args()
    if not 0<args.interval<=2:raise ValueError('Use an interval in (0,2] seconds')
    import torch
    torch.set_num_threads(8);cv2.setNumThreads(2)
    from field_specialist import Specialist
    from toggle_color import ColorReader
    from goal_observer import observe_goal_stacks
    training=json.loads((ROOT/'runs/latest_elims_goal_training.json').read_text())
    if training['status']!='complete':raise ValueError('Goal batch training incomplete')
    videos=json.loads(Path(args.manifest).read_text(encoding='utf-8'))['videos']
    toggle_dir=Path(json.loads((ROOT/'runs/active_toggle_pilot.json').read_text())['directory'])
    colors=ColorReader(toggle_dir/'toggle-color-classifier.pt')
    pointer=ROOT/'runs/latest_elims_field_inference.json';state=dict(status='running',weights=training['weights'],videos=[])
    def save():pointer.write_text(json.dumps(state,indent=2))
    save()
    try:
        for video in videos:
            while True:
                try:robot_state=json.loads((ROOT/'runs/latest_elims_robot_inference.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
                if robot_state['status']=='failed':raise ValueError('Robot inference failed')
                item=next((x for x in robot_state['videos'] if x['id']==video['id'] and x['status']=='complete'),None)
                if item:break
                time.sleep(1)
            model=Specialist(training['weights'],max_estimate_seconds=args.interval+.15,
                             calibration=video['calibration'],search_columns=(0,))
            goal_rois={g:[b[0]/2048,b[1]/1152,b[2]/2048,b[3]/1152] for g,b in model.goals.items()}
            target=ROOT/'runs/elims-batch'/video['id'];output=target/'field-records.jsonl'
            entry=dict(id=video['id'],source=video['source'],records=str(output),status='running',frames=video['frames'])
            state['videos'].append(entry);save();capture=cv2.VideoCapture(video['source']);count=0
            with Path(item['records']).open() as source,output.open('x') as log:
                for index,line in enumerate(source):
                    row=json.loads(line);ok,frame=capture.read()
                    if not ok or row['frame']!=index:raise ValueError('Video/robot frame mismatch')
                    image=cv2.resize(frame,(2048,1152))
                    if index%max(1,round(video['fps']*args.interval))==0:
                        objects=model.detect(image,row['seconds'],row['robots'])
                    else:objects=model.propagate(image,row['seconds'])
                    row.update(field_objects=objects,goal_rois=goal_rois,toggles=colors.read(image,video['toggle_rois']),
                               toggle_rois=video['toggle_rois'],scoring_validity='unconfirmed')
                    row['goal_estimate']=observe_goal_stacks(row)
                    log.write(json.dumps(row)+'\n');count+=1
                    if index%600==0:print(f'BATCH_FIELD {video["id"]} {index}/{video["frames"]}',flush=True)
            capture.release()
            if count!=video['frames']:raise ValueError('Incomplete field observations')
            entry.update(status='complete',written_frames=count);save()
        state.update(status='complete');save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
