"""Revise robot identities while reusing the accepted Toggle frame records exactly."""
import json,subprocess
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent
COLORS={'red':(40,40,240),'blue':(250,130,30),None:(180,180,180),'yellow':(30,230,250),'unknown':(170,170,170)}


def caption(image,text,position,color,scale=.75):
    x,y=position
    (w,h),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
    x=max(0,min(image.shape[1]-w-8,x));y=max(h+5,min(image.shape[0]-baseline-3,y))
    cv2.rectangle(image,(x-2,y-h-4),(x+w+4,y+baseline+2),(15,15,15),-1)
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)


def resolve(objects,mapping):
    result=[]
    for obj in objects:
        entry=mapping.get(str(obj.get('track_id')))
        if not entry or entry['frames']<30:continue
        team=entry['team']
        margin=entry.get('alternative_cost_margin')
        if entry['method']!='opening_seed' and margin is not None and margin<.5:team=None
        row=dict(obj,team=team,alliance=('red' if team in ('1744X','471A') else 'blue') if team else None,
                 identity_source=entry['method'] if team else 'unconfirmed')
        result.append(row)
    # Duplicate detector boxes for one physical machine may carry different IDs.
    filtered=[]
    for team in ('1744X','471A','88S','2899A'):
        candidates=[o for o in result if o['team']==team]
        if candidates:
            filtered.append(max(candidates,key=lambda o:(o['xyxy'][2]-o['xyxy'][0])*(o['xyxy'][3]-o['xyxy'][1])*o['confidence']))
    filtered.extend(o for o in result if o['team'] is None)
    return filtered


def draw(image,robots,toggles,rois):
    image=image.copy();h,w=image.shape[:2]
    for obj in robots:
        color=COLORS[obj['alliance']]
        x1,y1,x2,y2=[round(n*d) for n,d in zip(obj['xyxy'],[w,h,w,h])]
        cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
        text=f'{obj["team"]} | {obj["alliance"].upper()}' if obj['team'] else 'Team unconfirmed'
        caption(image,text,(x1+5,y1+29),color,.8)
    for key,state in toggles.items():
        x1,y1,x2,y2=[round(n*d) for n,d in zip(rois[key],[w,h,w,h])];color=COLORS[state['color']]
        cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
        caption(image,f'Toggle {key}: {state["color"].upper()}',(x1,y1-8),color,.65)
    caption(image,'miko yolo',(24,h-24),(245,245,245),.85)
    return image


if __name__=='__main__':
    target=Path(json.loads((ROOT/'runs/active_toggle_pilot.json').read_text())['directory'])
    mapping=json.loads((target/'identity-associations.json').read_text())['mapping']
    summary=json.loads((target/'render-summary.json').read_text());source=summary['source']
    rois=json.loads((target/'toggle-rois.json').read_text())
    capture=cv2.VideoCapture(source);w=round(capture.get(3));h=round(capture.get(4));fps=capture.get(5);total=round(capture.get(7))
    output=target/'miko-yolo-teams-corrected.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
             '-r',str(fps),'-i','pipe:0','-i',source,'-map','0:v:0','-map','1:a?',
             '-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-c:a','copy','-movflags','+faststart',str(output)]
    with (target/'corrected-ffmpeg.log').open('w') as errors,(target/'detections.jsonl').open(encoding='utf-8') as records,(target/'corrected-identities.jsonl').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=errors,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        count=0
        try:
            for index,line in enumerate(records):
                record=json.loads(line);ok,image=capture.read()
                if not ok or record['frame']!=index:raise ValueError('Frame mismatch')
                robots=resolve(record['robots'],mapping)
                annotated=draw(image,robots,record['toggles'],rois)
                process.stdin.write(annotated.tobytes())
                record['robots']=robots;record['watermark']='miko yolo'
                log.write(json.dumps(record)+'\n');count+=1
                if index in (0,900,2900,4000,total-1):cv2.imwrite(str(target/f'corrected-preview-{index:06d}.jpg'),annotated)
                if index%200==0:print(f'RENDER {index}/{total}; identified {sum(o["team"] is not None for o in robots)}',flush=True)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('FFmpeg failed')
        except BaseException:
            if process.poll() is None:process.kill();process.wait()
            raise
        finally:capture.release()
    if count!=total:raise ValueError('Incomplete video')
    (target/'corrected-summary.json').write_text(json.dumps({'video':str(output),'frames':count,
        'alliances':{'red':['1744X','471A'],'blue':['88S','2899A']},'watermark':'miko yolo',
        'toggles':'reused accepted values without modification','limitations':'recording-specific identity association; ambiguous tracklets remain unconfirmed'},indent=2),encoding='utf-8')
    print('RENDER_COMPLETE '+str(output),flush=True)
