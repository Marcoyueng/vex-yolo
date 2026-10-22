"""Same-recording pilot: robot team seeds plus visual Toggle colors.

Field anchors are calibrated from reviewed training boxes for this fixed view.
They are not universal quadrants or official scoring decisions.
"""
import json
import os
from pathlib import Path
import subprocess
from statistics import median
import cv2
from opening_alliance import OpeningAllianceTracker,center

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')
STATE_COLORS={'red':(40,40,240),'blue':(250,130,30),'yellow':(30,230,250),'unknown':(170,170,170)}


def calibration(target,split):
    points={'left':(.25,.31),'top':(.52,.17),'right':(.78,.34),'bottom':(.52,.79)}
    groups={key:[] for key in points}
    for name in split['train_images']:
        label=ROOT/'data/annotations-v3'/(Path(name).stem+'.txt')
        for line in label.read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            if int(cls) not in (2,3,4):continue
            key=min(points,key=lambda key:(points[key][0]-x)**2+(points[key][1]-y)**2)
            groups[key].append([x-w/2,y-h/2,x+w/2,y+h/2])
    return {key:[median(row[i] for row in values) for i in range(4)] for key,values in groups.items() if values}


def main():
    import torch
    torch.set_num_threads(8)
    from toggle_color import ColorReader
    active=json.loads((ROOT/'runs/active_toggle_pilot.json').read_text(encoding='utf-8'))
    target=Path(active['directory']);split=json.loads((target/'split.json').read_text(encoding='utf-8'))
    config=json.loads((target/'opening-config.json').read_text(encoding='utf-8'))
    source=next(Path(p) for p in split['sources'] if Path(p).is_file())
    cap=cv2.VideoCapture(str(source));width=round(cap.get(3));height=round(cap.get(4));fps=cap.get(5);frames=round(cap.get(7));cap.release()
    rois=json.loads((target/'color-rois.json').read_text(encoding='utf-8'))
    (target/'toggle-rois.json').write_text(json.dumps(rois,indent=2),encoding='utf-8')
    tracker=OpeningAllianceTracker(config['anchors'],fps)
    reader=ColorReader(target/'toggle-color-classifier.pt')
    previous=json.loads((ROOT/'runs/active_robot_pilot.json').read_text(encoding='utf-8'))
    robot_report=Path(previous['directory'])/'robot-detections.jsonl'
    output=target/'robots-teams-toggles.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
         '-r',str(fps),'-i','pipe:0','-i',str(source),'-map','0:v:0','-map','1:a?',
         '-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-c:a','copy','-movflags','+faststart',str(output)]
    count=0
    with (target/'ffmpeg.log').open('w') as errors,(target/'detections.jsonl').open('x',encoding='utf-8') as log,robot_report.open(encoding='utf-8') as robot_log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=errors,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        capture=cv2.VideoCapture(str(source))
        try:
            for index,line in enumerate(robot_log):
                record=json.loads(line)
                if record['frame']!=index:raise ValueError('Robot records are not sequential')
                ok,image=capture.read()
                if not ok:raise ValueError('Original video ended early')
                annotated=image.copy()
                raw=[{'xyxy':[obj['xyxy'][0]/width,obj['xyxy'][1]/height,obj['xyxy'][2]/width,obj['xyxy'][3]/height],
                      'track_id':obj['track_id'],'confidence':obj['confidence']} for obj in record['detections']]
                teams=tracker.update(raw,index)
                for obj in teams:
                    color=STATE_COLORS.get(obj['alliance'],STATE_COLORS['unknown'])
                    x1,y1,x2,y2=[round(n*dim) for n,dim in zip(obj['xyxy'],[width,height,width,height])]
                    label=f'{obj["team"]} {obj["alliance"].upper()}' if obj['team'] and obj['alliance'] else 'ROBOT - UNCONFIRMED'
                    label+=f' ID{obj["track_id"]} {obj["confidence"]:.2f}'
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)
                    cv2.putText(annotated,label,(x1,max(30,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.72,color,2,cv2.LINE_AA)
                states=reader.read(image,rois)
                for key,state in states.items():
                    color=STATE_COLORS[state['color']]
                    x1,y1,x2,y2=[round(n*d) for n,d in zip(rois[key],[width,height,width,height])]
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)
                    caption=f'Toggle {key}: {state["color"].upper()}'
                    if state['confidence'] is not None:caption+=f' {state["confidence"]:.2f}'
                    cv2.putText(annotated,caption,(x1,max(30,y1-9)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
                cv2.putText(annotated,'Toggle = visual color only; validity not checked',(24,height-24),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),2,cv2.LINE_AA)
                process.stdin.write(annotated.tobytes())
                log.write(json.dumps({'frame':index,'seconds':index/fps,'robots':teams,'toggles':states})+'\n')
                count+=1
                if index in (0,round(fps*30),round(fps*90),frames-1):cv2.imwrite(str(target/f'preview-{index:06d}.jpg'),annotated)
                if index%100==0:print(f'RENDER {index}/{frames}; confirmed identities {sum(o["team"] is not None for o in teams)}; toggle colors {sum(s["confidence"] is not None for s in states.values())}',flush=True)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('FFmpeg failed')
        except BaseException:
            if process.poll() is None:process.kill();process.wait()
            raise
        finally:
            capture.release()
    if count!=frames:raise ValueError('Incomplete source decode')
    (target/'render-summary.json').write_text(json.dumps({'video':str(output),'frames':count,'source':str(source),
        'limitations':'same-video pilot; teams based on user opening seeds and motion; toggle visual colors only'},indent=2),encoding='utf-8')
    print('RENDER_COMPLETE '+str(output),flush=True)


if __name__=='__main__':main()
