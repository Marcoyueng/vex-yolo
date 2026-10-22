"""Render the entire original pilot recording, preserving its audio if present."""
import json
import os
from pathlib import Path
import subprocess
import cv2

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))
os.environ.setdefault('OMP_NUM_THREADS','8')


def main():
    import torch
    torch.set_num_threads(8)
    from ultralytics import YOLO
    active=json.loads((ROOT/'runs/active_robot_pilot.json').read_text(encoding='utf-8'))
    target=Path(active['directory'])
    split=json.loads((target/'split.json').read_text(encoding='utf-8'))
    source=next((Path(p) for p in split['sources'] if Path(p).is_file()),None)
    if source is None:raise ValueError('Original recording is unavailable')
    cap=cv2.VideoCapture(str(source))
    width,height=round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS);frames=round(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    output=target/'robot-tracked.mp4'
    if output.exists():raise ValueError('Output exists; refusing to overwrite')
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24',
             '-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-i',str(source),
             '-map','0:v:0','-map','1:a?','-c:v','libx264','-threads','4','-preset','veryfast','-crf','20',
             '-c:a','copy','-movflags','+faststart',str(output)]
    model=YOLO(active['weights'])
    ffmpeg_log=(target/'render-ffmpeg.log').open('w',encoding='utf-8')
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=ffmpeg_log,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    count=0;detected=0
    try:
        with (target/'robot-detections.jsonl').open('x',encoding='utf-8') as log:
            for index,result in enumerate(model.track(source=str(source),stream=True,save=False,
                     tracker='bytetrack.yaml',classes=[0],device='cpu',imgsz=640,conf=.35,iou=.45,verbose=False)):
                annotated=result.plot(line_width=2,font_size=12,color_mode='instance')
                process.stdin.write(annotated.tobytes())
                boxes=result.boxes
                ids=boxes.id.cpu().tolist() if boxes.id is not None else [None]*len(boxes)
                objects=[{'track_id':int(t) if t is not None else None,'xyxy':xyxy,'confidence':conf,'class':'robot'}
                         for xyxy,conf,t in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),ids)]
                log.write(json.dumps({'frame':index,'seconds':index/fps,'detections':objects})+'\n')
                count+=1
                if objects:detected+=1
                if index in (0,round(fps*30),round(fps*90)):
                    cv2.imwrite(str(target/f'preview-{index:06d}.jpg'),annotated)
                if index%100==0:print(f'RENDER {index}/{frames} frames; {len(objects)} robots',flush=True)
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg failed; inspect render-ffmpeg.log')
    except BaseException:
        if process.poll() is None:process.kill();process.wait()
        raise
    finally:
        ffmpeg_log.close()
    if count!=frames:raise ValueError(f'Incomplete decode: {count}/{frames}')
    summary={'source':str(source),'output':str(output),'frames':count,'frames_with_detections':detected,
             'fps':fps,'width':width,'height':height,'scope':'robot only; IDs are not team numbers',
             'validation':'same recording pilot; generalization not verified'}
    (target/'render-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('RENDER_COMPLETE '+str(output),flush=True)


if __name__=='__main__':main()
