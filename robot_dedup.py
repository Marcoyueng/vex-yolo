"""Conservative removal of nested robot detections sharing the same image-space base.

This operates on detector output, never on ground-truth labels. It does not infer
team identities or force the number of robots to four. Distinct robots can still
be ambiguous in one frame; callers should retain the suppression audit.
"""
from math import isfinite


def deduplicate_robot_boxes(objects):
    """Return (kept_objects, audit) without changing input objects.

    Only lower-confidence boxes are suppressed. Require near-complete nesting,
    similar widths and aligned bases; ordinary overlaps remain separate.
    Audit indices are zero-based indices into the input sequence.
    """
    for obj in objects:
        x1, y1, x2, y2 = obj['xyxy']
        if not all(isfinite(v) for v in (x1, y1, x2, y2, obj['confidence'])):
            raise ValueError('Non-finite robot detection')
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Robot detection must have positive width and height')
    order = sorted(range(len(objects)), key=lambda i: (-objects[i]['confidence'], i))
    removed = set()
    audit = []
    for position, i in enumerate(order):
        if i in removed:
            continue
        a = objects[i]['xyxy']
        aw, ah = a[2]-a[0], a[3]-a[1]
        for j in order[position+1:]:
            if j in removed or objects[j]['confidence'] == objects[i]['confidence']:
                continue
            b = objects[j]['xyxy']
            bw, bh = b[2]-b[0], b[3]-b[1]
            intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
            nested_fraction = intersection/min(aw*ah,bw*bh)
            base_difference = abs(a[3]-b[3])/max(ah,bh)
            center_difference = abs((a[0]+a[2])-(b[0]+b[2]))/2/max(aw,bw)
            width_ratio = min(aw,bw)/max(aw,bw)
            if nested_fraction >= .95 and base_difference <= .05 and center_difference <= .15 and width_ratio >= .6:
                removed.add(j)
                audit.append(dict(kept_index=i, removed_index=j, reason='nested_same_base',
                    nested_fraction=nested_fraction, base_difference=base_difference,
                    center_difference=center_difference, width_ratio=width_ratio))
    return [obj for i,obj in enumerate(objects) if i not in removed], audit
