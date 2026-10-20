"""Add locally reviewed stack crops to a previous specialist dataset.

Media paths and annotation coordinates are supplied by an ignored local JSON.
Only explicitly reviewed regions are cropped; originals are never changed.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
import cv2
from schema import CLASSES

ROOT=Path(__file__).resolve().parent


def prepare(review_path,base_run):
    review=json.loads(Path(review_path).read_text(encoding='utf-8'))
    base=Path(base_run)
    target=ROOT/'runs'/('stack-refinement-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    shutil.copytree(base/'dataset',target/'dataset')
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(parent_run=str(base),stack_review=str(Path(review_path).resolve()),
                    limitations='Assistant-reviewed partial crops; same-video validation, not independent match accuracy.')
    manifest['stack_samples']=[]
    for sample in review['samples']:
        image=cv2.imread(sample['image'])
        if image is None:raise ValueError('Reviewed image unavailable')
        if list(image.shape[1::-1])!=review['coordinate_size']:raise ValueError('Annotation coordinate size mismatch')
        split=sample['split']
        if split not in ('train','val'):raise ValueError('Invalid split')
        for index,obj in enumerate(sample['objects']):
            box=obj['xyxy'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            side=112
            left=max(0,min(image.shape[1]-side,round(cx-side/2)))
            top=max(0,min(image.shape[0]-side,round(cy-side/2)))
            # Do not use unreviewed neighboring pixels as unlabeled negatives.
            crop=image[top:top+side,left:left+side].copy()
            region=sample['review_region']
            mask=cv2.rectangle(__import__('numpy').zeros(crop.shape[:2],dtype='uint8'),
                (max(0,region[0]-left),max(0,region[1]-top)),
                (min(side-1,region[2]-left),min(side-1,region[3]-top)),255,-1)
            crop[mask==0]=0
            labels=[]
            for other in sample['objects']:
                a,b,c,d=other['xyxy'];x1,y1,x2,y2=max(a,left),max(b,top),min(c,left+side),min(d,top+side)
                if x2<=x1 or y2<=y1 or (x2-x1)*(y2-y1)/((c-a)*(d-b))<.5:continue
                labels.append(f"{other['class_id']} {(x1+x2-2*left)/2/side:.7f} {(y1+y2-2*top)/2/side:.7f} {(x2-x1)/side:.7f} {(y2-y1)/side:.7f}")
            name=f"stack_{sample['id']}__{index:03d}"
            cv2.imwrite(str(target/'dataset/images'/split/(name+'.jpg')),crop)
            (target/'dataset/labels'/split/(name+'.txt')).write_text('\n'.join(labels)+'\n')
            manifest['stack_samples'].append(dict(name=name,split=split,source=sample['image'],class_id=obj['class_id']))
    config=target/'dataset/dataset.yaml'
    config.write_text('path: '+json.dumps((target/'dataset').as_posix())+'\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)),encoding='utf-8')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return target,config
