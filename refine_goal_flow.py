"""Replay saved detector samples with bounded, pixel-verified propagation.

No training or detector samples change; this allows tracking fixes to be
checked across a full source video without rerunning the detector.
"""
import json
from pathlib import Path
import cv2
from field_specialist import Specialist
from goal_observer import observe_goal_stacks

ROOT=Path(__file__).resolve().parent


def main():
    cv2.setNumThreads(2)
    state=json.loads((ROOT/'runs/latest_stack_inference.json').read_text(encoding='utf-8'))
    if state['status']!='complete':raise ValueError('Detector inference is not complete')
    output=Path(state['directory'])/'stable-field-detections.jsonl'
    model=Specialist.__new__(Specialist)
    model.max_estimate_seconds=state['interval_seconds']+.15
    cap=cv2.VideoCapture(state['source']);fps=cap.get(5)
    interval=max(1,round(fps*state['interval_seconds']))
    written=0
    try:
        with Path(state['records']).open(encoding='utf-8') as source,output.open('x',encoding='utf-8') as log:
            for index,line in enumerate(source):
                row=json.loads(line);ok,frame=cap.read()
                if not ok or row['frame']!=index:raise ValueError('Video/records mismatch')
                image=cv2.resize(frame,(2048,1152))
                if index%interval==0:
                    objects=row['field_objects']
                    model.objects=[dict(o) for o in objects]
                    model.previous=cv2.cvtColor(cv2.resize(image,(1024,576)),cv2.COLOR_BGR2GRAY)
                    model.sample_seconds=row['seconds']
                else:objects=model.propagate(image,row['seconds'])
                row['field_objects']=objects;row['goal_estimate']=observe_goal_stacks(row)
                log.write(json.dumps(row)+'\n');written+=1
                if index%600==0:print(f'STABLE_STACK_FLOW {index}/{state["frames"]}',flush=True)
        if written!=state['frames']:raise ValueError('Incomplete propagation')
        report=dict(state,status='complete',records=str(output),detector_records=state['records'],
                    propagation='bounded optical flow and stationary pixel agreement; estimates, not fresh detections')
        (ROOT/'runs/latest_stack_flow.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('STABLE_STACK_FLOW_COMPLETE',flush=True)
    finally:cap.release()


if __name__=='__main__':main()
