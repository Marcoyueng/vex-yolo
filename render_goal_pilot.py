"""Render the original full video with accepted robots/Toggles and field drafts.

Original audio is copied. The specialist runs ~once/two seconds, with bounded optical
flow estimates between samples. No Goal score is manufactured from detections.
"""
import json,os,subprocess
from pathlib import Path
import cv2
from field_specialist import Specialist,PALETTE
from overlay_style import draw,caption
from goal_observer import observe_goal_stacks

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/config'))


def dashed(image,box,color):
    x,y,x2,y2=box
    for a,b,c,d in ((x,y,x2,y),(x,y2,x2,y2),(x,y,x,y2),(x2,y,x2,y2)):
        length=max(abs(c-a),abs(d-b))
        for offset in range(0,length,14):
            end=min(length,offset+8)
            if length:
                p=(round(a+(c-a)*offset/length),round(b+(d-b)*offset/length))
                q=(round(a+(c-a)*end/length),round(b+(d-b)*end/length))
                cv2.line(image,p,q,color,2)


def render():
    import torch
    torch.set_num_threads(8)
    goal_pointer=ROOT/'runs/active_goal_pilot.json'
    state=json.loads(goal_pointer.read_text(encoding='utf-8'))
    if state['status']!='complete':raise ValueError('Goal specialist training has not finished')
    target=Path(state['directory'])
    goals=json.loads((target/'manifest.json').read_text(encoding='utf-8'))['goals']
    accepted=Path(json.loads((ROOT/'runs/active_toggle_pilot.json').read_text())['directory'])
    source=json.loads((accepted/'render-summary.json').read_text())['source']
    rois=json.loads((accepted/'toggle-rois.json').read_text())
    output=target/'miko-yolo-goals-pins.mp4';records_path=target/'field-detections.jsonl'
    if output.exists() or records_path.exists():raise ValueError('Output already exists; choose another run rather than overwrite')
    capture=cv2.VideoCapture(source)
    width,height=round(capture.get(3)),round(capture.get(4));fps=capture.get(5);total=round(capture.get(7))
    if not capture.isOpened() or fps<=0:raise ValueError('Video unavailable')
    specialist=Specialist(state['weights'],max_estimate_seconds=2.1)
    cv2.setNumThreads(2)
    goal_rois={key:[b[0]/2048,b[1]/1152,b[2]/2048,b[3]/1152] for key,b in goals.items()}
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
             '-r',str(fps),'-i','pipe:0','-i',source,'-map','0:v:0','-map','1:a?',
             '-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-c:a','copy','-movflags','+faststart',str(output)]
    index=0
    try:
        with (target/'render-ffmpeg.log').open('w') as errors,(accepted/'corrected-identities.jsonl').open(encoding='utf-8') as previous,records_path.open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=errors,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                for index,line in enumerate(previous):
                    row=json.loads(line);ok,frame=capture.read()
                    if not ok or row['frame']!=index:raise ValueError('Video / record mismatch')
                    review=cv2.resize(frame,(2048,1152))
                    sampled=index%round(fps*2)==0
                    objects=specialist.detect(review,row['seconds'],row['robots']) if sampled else specialist.propagate(review,row['seconds'])
                    image=draw(frame,row['robots'],row['toggles'],rois)
                    for key,box in goal_rois.items():
                        pixel=[round(v*d) for v,d in zip(box,[width,height,width,height])]
                        dashed(image,pixel,PALETTE[6]);caption(image,key+' ROI',(pixel[0],pixel[3]+20),PALETTE[6],.55)
                    goal_observation=observe_goal_stacks(dict(row,field_objects=objects,goal_rois=goal_rois))
                    for obj in goal_observation['objects']:
                        cls=obj['class_id']
                        if cls==6:continue # separate calibrated Goal outlines avoid duplicate boxes
                        pixel=[round(v*d) for v,d in zip(obj['xyxy'],[width,height,width,height])]
                        color=PALETTE[cls]
                        if obj['observation']=='optical_flow_estimate':dashed(image,pixel,color)
                        else:cv2.rectangle(image,tuple(pixel[:2]),tuple(pixel[2:]),color,2)
                        label={7:'Pin-R',8:'Pin-B',9:'Pin-Y',10:'Cup'}[cls]
                        if obj['observation']=='optical_flow_estimate':label+=' ~'
                        caption(image,label,(pixel[0],pixel[1]-4),color,.45)
                    caption(image,'Goal ROI / Pin-Cup drafts; scoring unconfirmed',(24,height-62),(240,240,240),.62)
                    process.stdin.write(image.tobytes())
                    row.update(field_objects=[o for o in objects if o['class_id']==6]+goal_observation['objects'],goal_rois=goal_rois,
                               goal_estimate=goal_observation,field_scoring_validity='unconfirmed')
                    log.write(json.dumps(row)+'\n')
                    if index in (0,900,1800,2700,3600,4500,total-1):cv2.imwrite(str(target/f'preview-{index:06d}.jpg'),image)
                    if index%300==0:print(f'RENDER {index}/{total}; {len(objects)} field drafts',flush=True)
                process.stdin.close()
                if process.wait()!=0:raise RuntimeError('FFmpeg failed')
                if index+1!=total:raise ValueError('Incomplete video')
            except BaseException:
                if process.poll() is None:process.kill();process.wait()
                raise
    finally:capture.release()
    # Check complete video and original audio before publishing the pointer.
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(output)],text=True))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    if int(video['nb_read_frames'])!=total:raise ValueError('Encoded frame count mismatch')
    state.update(video=str(output),preview_jsonl=str(records_path),source=source,frames=total,goal_rois=goal_rois,
                 field_scoring_validity='unconfirmed',preview_method='0.5Hz crop specialist; max 2.1s optical-flow estimates between samples; calibrated Goal ROIs',
                 audio_streams=sum(s['codec_type']=='audio' for s in probe['streams']))
    (target/'render-summary.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    temporary=goal_pointer.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2),encoding='utf-8');temporary.replace(goal_pointer)
    print('RENDER_COMPLETE '+str(output),flush=True)


if __name__=='__main__':render()
