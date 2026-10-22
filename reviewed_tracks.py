"""Merge only explicitly reviewed, segment-scoped parts of the same robot."""
from copy import deepcopy
from collections import Counter
import math


def split_reviewed_robots(robots, reviews, roster):
    """Apply explicit current-frame geometry; never extrapolate a split."""
    result=deepcopy(robots)
    for review in reviews:
        selected=[o for o in result if o.get('identity_segment')==review['segment']
                  and o['track_id']==review['parent_track_id']]
        if not selected:continue
        if len(selected)!=1 or selected[0].get('source_track_ids'):
            raise ValueError('Split requires one original, unmerged detection')
        parts=review['parts'];parent=selected[0]
        if len(parts)!=2 or len({p['team'] for p in parts})!=2 or len({p['track_id'] for p in parts})!=2:
            raise ValueError('Split requires two distinct reviewed robots')
        children=[]
        for part in parts:
            box=part['xyxy'];team=part['team']
            if (team not in roster or len(box)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in box)
                    or box[2]<=box[0] or box[3]<=box[1]):
                raise ValueError('Invalid reviewed split box or team')
            if any(o['track_id']==part['track_id'] or o.get('team')==team for o in result if o is not parent):
                raise ValueError('Split would duplicate a known robot')
            child=deepcopy(parent)
            for key in ('identity_review_label','identity_review_note','identity_anchor_frame'):
                child.pop(key,None)
            child.update(track_id=part['track_id'],team=team,alliance=roster[team],xyxy=list(box),
                identity_segment=review['segment']+'/'+str(part['track_id']),
                source_track_ids=[parent['track_id']],source_identity_segments=[review['segment']],
                reviewed_split=True,split_review_id=review['review_id'],
                geometry_source='assistant_reviewed_split_current_pixel_match',
                identity_source='user_confirmed_front_rear_split',
                confidence_scope='original_combined_detection_not_individual_robot',
                review_pixel_correlation=part['pixel_correlation'])
            children.append(child)
        result=[o for o in result if o is not parent]+children
    return result


def validate_reviewed_provenance(robots, original, reviews):
    """Count a reviewed two-child split once, without losing any source IDs."""
    expected={r['review_id']:r for r in reviews};groups={};seen=[]
    for obj in robots:
        if obj.get('reviewed_split'):
            key=obj['split_review_id']
            if key not in expected:raise ValueError('Unregistered split')
            groups.setdefault(key,[]).append(obj)
        else:seen.extend(obj.get('source_track_ids',[obj['track_id']]))
    for key,children in groups.items():
        review=expected[key]
        if len(children)!=2 or {o['track_id'] for o in children}!={p['track_id'] for p in review['parts']}:
            raise ValueError('Missing or duplicated split child')
        if any(o['source_track_ids']!=[review['parent_track_id']] for o in children):
            raise ValueError('Incorrect split provenance')
        seen.append(review['parent_track_id'])
    if Counter(seen)!=Counter(o['track_id'] for o in original):
        raise ValueError('Source detections were lost or duplicated')


def merge_reviewed_parts(robots,groups):
    result=deepcopy(robots)
    for group in groups:
        selected=[o for o in result if o.get('identity_segment') in group['segments']]
        if len(selected)<2:continue
        if any(o.get('team')!=group['team'] for o in selected):
            raise ValueError('A merge cannot override a different team')
        primary=max(selected,key=lambda o:(o['track_id']==group.get('preferred_id'),o['confidence']))
        merged=deepcopy(primary)
        boxes=[o['xyxy'] for o in selected]
        if group['mode']=='union':
            box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            if box[2]-box[0]>group.get('max_width',.3) or box[3]-box[1]>group.get('max_height',.55):
                continue # evidence has drifted too far; do not bridge distant robots
            merged['xyxy']=box
        elif group['mode']=='best':
            best=max(selected,key=lambda o:o['confidence'])
            merged=deepcopy(best)
        else:raise ValueError('Unknown reviewed merge mode')
        merged.update(source_track_ids=sorted(o['track_id'] for o in selected),
            source_identity_segments=sorted(o['identity_segment'] for o in selected),
            reviewed_merge=True,geometry_source='reviewed_same_robot_'+group['mode'],
            identity_source='reviewed_same_robot_parts')
        result=[o for o in result if o not in selected]+[merged]
    return result
