"""Assistant-reviewed object crops from six frames of the user's fixed view.

These are partial, assistant-made annotations, not human-certified full-scene
ground truth. Only the selected crop regions may be used for this experiment.
No Placed, visible-for-scoring, nesting, contact or score labels are inferred.
"""
import json, hashlib
from pathlib import Path
from datetime import datetime
import cv2
from schema import SCHEMA, CLASSES

ROOT=Path(__file__).resolve().parent


def prepare(curation_path=None):
    if curation_path is None:
        pointer=ROOT/'runs/goal_curation_source.json'
        if not pointer.exists():raise ValueError('请提供本地 curation JSON；视频标注不随代码发布')
        curation_path=json.loads(pointer.read_text(encoding='utf-8'))['path']
    curation=json.loads(Path(curation_path).read_text(encoding='utf-8'))
    goals=curation['goals']
    common={int(k):v for k,v in curation['common'].items()}
    extra={int(k):v for k,v in curation['extra'].items()}
    target=ROOT/'runs'/('goal-pin-pilot-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    target.mkdir(parents=True)
    frames=tuple(curation['frames'])
    source_folder=Path(curation['images'])
    manifest={'scope':CLASSES[6:],'annotation_source':'assistant visual review of selected regions; partial labels',
              'limitations':'six frames, same camera, same video; crop validation only; NOT scoring-state or identity training',
              'goals':goals,'field_polygon':curation['field_polygon'],'curation_source':str(Path(curation_path).resolve()),'train_frames':list(frames[:4]),'validation_frames':list(frames[4:]),'samples':[]}
    counts={'train':{},'val':{}}
    for frame in frames:
        image_path=next(source_folder.glob(f'*{frame:08d}.jpg'))
        image=cv2.resize(cv2.imread(str(image_path)),(2048,1152))
        phase='train' if frame in frames[:4] else 'val'
        objects=[]
        for goal,box in goals.items():
            if goal in curation.get('skip_goals',{}).get(str(frame),[]):continue
            objects.append((6,box,goal))
        for cls,boxes in common.items():
            # These four midline assemblies have been removed later in the match.
            if frame>=curation.get('common_until_frame',float('inf')) and cls in (7,8,10):continue
            for box in boxes:
                if [cls,box] in curation.get('removed_common',{}).get(str(frame),[]):continue
                objects.append((cls,box,None))
        for cls_key,boxes in extra[frame].items():
            cls=int(cls_key)
            for box in boxes:objects.append((cls,box,None))
        # Record partial annotations explicitly; never present as complete scene labels.
        annotation={'frame':frame,'source':str(image_path),'source_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    'review_size':[2048,1152],'complete_scene':False,'objects':[dict(class_id=c,xyxy=b,goal=g) for c,b,g in objects]}
        (target/f'partial-review-{frame:08d}.json').write_text(json.dumps(annotation,indent=2),encoding='utf-8')
        for index,(cls,box,_) in enumerate(objects):
            x1,y1,x2,y2=box
            # Only annotated local regions; originals remain untouched.
            side=144 if cls==6 else 112
            cx,cy=(x1+x2)//2,(y1+y2)//2
            left=max(0,min(2048-side,cx-side//2));top=max(0,min(1152-side,cy-side//2))
            crop=image[top:top+side,left:left+side]
            labels=[]
            for other,bb,_ in objects:
                ax,ay,bx,by=bb
                xx1,yy1,xx2,yy2=max(ax,left),max(ay,top),min(bx,left+side),min(by,top+side)
                if xx2<=xx1 or yy2<=yy1:continue
                retained=(xx2-xx1)*(yy2-yy1)/((bx-ax)*(by-ay))
                if retained<.5:continue
                labels.append(f'{other} {(xx1+xx2-2*left)/2/side:.7f} {(yy1+yy2-2*top)/2/side:.7f} {(xx2-xx1)/side:.7f} {(yy2-yy1)/side:.7f}')
            name=f'frame{frame:08d}__object{index:03d}'
            for kind in ('images','labels'):(target/'dataset'/kind/phase).mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(target/'dataset/images'/phase/(name+'.jpg')),crop)
            (target/'dataset/labels'/phase/(name+'.txt')).write_text('\n'.join(labels)+'\n',encoding='utf-8')
            counts[phase][CLASSES[cls]]=counts[phase].get(CLASSES[cls],0)+1
            manifest['samples'].append(dict(frame=frame,name=name,split=phase,focus=CLASSES[cls],crop_xyxy=[left,top,left+side,top+side]))
    (target/'dataset/classes.json').write_text(json.dumps(SCHEMA),encoding='utf-8')
    for phase in ('train','val'):
        (target/'dataset/labels'/phase/'classes.json').write_text(json.dumps(SCHEMA),encoding='utf-8')
    config=target/'dataset/dataset.yaml'
    config.write_text('path: '+json.dumps((target/'dataset').as_posix())+'\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)),encoding='utf-8')
    manifest['focus_counts']=counts
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (ROOT/'runs/active_goal_pilot.json').write_text(json.dumps({'directory':str(target),'status':'prepared','scope':CLASSES[6:]}),encoding='utf-8')
    print(json.dumps({'directory':str(target),'focus_counts':counts}),flush=True)
    return target,config


if __name__=='__main__':prepare()
