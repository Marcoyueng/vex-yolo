"""Detect every decoded frame in a locally reviewed video batch.

ByteTrack runs in frame order after batched CPU inference. Source videos remain
unchanged. Identity assignment is a separate reviewed association step.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(8);cv2.setNumThreads(2)
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    training=json.loads((ROOT/'runs/latest_elims_training.json').read_text())
    if training['status']!='complete':raise ValueError('Robot batch training is incomplete')
    videos=json.loads(Path(args.manifest).read_text(encoding='utf-8'))['videos']
    state=dict(status='running',weights=training['weights'],videos=[])
    pointer=ROOT/'runs/latest_elims_robot_inference.json'
    def save():pointer.write_text(json.dumps(state,indent=2))
    save()
    try:
        for video in videos:
            target=ROOT/'runs/elims-batch'/video['id'];target.mkdir(exist_ok=True)
            path=target/'robot-raw.jsonl'
            model=YOLO(training['weights'])
            tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                                               track_buffer=round(video['fps']*2),match_thresh=.8,fuse_score=True))
            polygon=np.float32(video['calibration']['field_polygon'])*np.float32([1280/2048,720/1152])
            capture=cv2.VideoCapture(video['source']);index=0;batch=[]
            entry=dict(id=video['id'],source=video['source'],records=str(path),status='running',frames=video['frames'])
            state['videos'].append(entry);save()
            with path.open('x',encoding='utf-8') as log:
                while True:
                    ok,frame=capture.read()
                    if ok:batch.append(cv2.resize(frame,(1280,720)))
                    if len(batch)<8 and ok:continue
                    if batch:
                        predictions=model.predict(batch,imgsz=640,conf=.12,iou=.45,classes=[0],device='cpu',batch=8,verbose=False)
                        for image,result in zip(batch,predictions):
                            boxes=result.boxes.cpu().numpy()
                            keep=[]
                            for b in boxes.xyxy:
                                point=(float((b[0]+b[2])/2),float(b[1]+.8*(b[3]-b[1])))
                                keep.append(cv2.pointPolygonTest(polygon,point,False)>=0)
                            boxes=boxes[np.array(keep,dtype=bool)]
                            tracks=tracker.update(boxes,image)
                            robots=[]
                            for item in sorted(tracks,key=lambda a:a[5],reverse=True)[:4]:
                                x,y,x2,y2,tid,confidence,cls,det=item
                                robots.append(dict(track_id=int(tid),xyxy=[float(x/1280),float(y/720),float(x2/1280),float(y2/720)],
                                                   confidence=float(confidence),team=None,alliance=None,identity_source='track_id_only'))
                            row=dict(frame=index,seconds=index/video['fps'],robots=robots,
                                     roster={t:c for c in ('red','blue') for t in video[c]},match=video['match'])
                            log.write(json.dumps(row)+'\n');index+=1
                            if index%300==0:print(f'BATCH_ROBOTS {video["id"]} {index}/{video["frames"]}',flush=True)
                        batch=[]
                    if not ok:break
            capture.release()
            if index!=video['frames']:raise ValueError('Incomplete decode')
            entry.update(status='complete',written_frames=index);save()
        state['status']='complete';save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
