"""Conservative 2-D Goal stack proposals. Geometry is not SC2/SC3 proof.

Only objects connected upward from a Goal mouth remain visible. Loose ground
objects and floating items in a tall rectangular search area are excluded.
Any automatic score here is explicitly provisional, never referee-confirmed.
"""
from match_state import match_score
from schema import REGIONS

SITES={'red1':'left','red2':'bottom','blue1':'right','blue2':'top'}


def center(box):
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)


def intersection_fraction(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1e-12,(a[2]-a[0])*(a[3]-a[1]))


def rooted(box,goal):
    cx,cy=center(box);gx,_=center(goal)
    gw,gh=goal[2]-goal[0],goal[3]-goal[1]
    return (abs(cx-gx)<=max(gw*.38,(box[2]-box[0])*.25) and
            goal[1]-gh*.25<=box[3]<=goal[1]+gh*.55 and cy<=goal[1]+gh*.25 and
            box[1]<=goal[1]-min(gh*.06,(box[3]-box[1])*.15))


def above(child,parent):
    cx,cy=center(child);px,py=center(parent)
    ch,pw,ph=child[3]-child[1],parent[2]-parent[0],parent[3]-parent[1]
    cw=child[2]-child[0]
    return (cy<py-.001 and abs(cx-px)<=max(pw*.65,cw*.65,.008) and
            parent[1]-.28*max(ph,ch)-.003<=child[3]<=py+ph*.3)


def unique_halves(objects):
    """Collapse a duplicate tight box inside a larger box for the same half.

    Different classes remain separate (a Pin may overlap its Cup). Adjacent
    same-color halves are retained when neither box contains the other.
    """
    kept=[]
    for obj in sorted(objects,key=lambda o:o.get('confidence',0),reverse=True):
        box=obj['xyxy'];cx,cy=center(box)
        duplicate=False
        for other in kept:
            if obj['class_id']!=other['class_id']:continue
            parent=other['xyxy'];px,py=center(parent)
            containment=max(intersection_fraction(box,parent),intersection_fraction(parent,box))
            if containment>.85 and abs(cx-px)<max(box[2]-box[0],parent[2]-parent[0])*.55 and abs(cy-py)<max(box[3]-box[1],parent[3]-parent[1])*.45:
                duplicate=True;break
        if not duplicate:kept.append(obj)
    return kept


def observe_goal_stacks(row,goal_rois=None):
    rois=goal_rois if goal_rois is not None else row.get('goal_rois',{})
    rois={g:list(b) for g,b in rois.items() if g in {f'G{i}' for i in range(9)}}
    objects=unique_halves([dict(o) for o in row.get('field_objects',[]) if o.get('class_id') in (7,8,9,10)])
    reached={}
    for goal,base in rois.items():
        found={i for i,o in enumerate(objects) if rooted(o['xyxy'],base)}
        changed=True
        while changed:
            changed=False
            for i,obj in enumerate(objects):
                if i not in found and any(above(obj['xyxy'],objects[j]['xyxy']) for j in found):
                    found.add(i);changed=True
        reached[goal]=found
    memberships={i:[g for g,s in reached.items() if i in s] for i in range(len(objects))}
    visible=[];goals={};counts={}
    for index,obj in enumerate(objects):
        if len(memberships[index])==1:
            visible.append(dict(obj,goal_id=memberships[index][0],placement='2d_stack_candidate',scoring_validity='unconfirmed'))
    for index in range(9):
        goal=f'G{index}';base=rois.get(goal)
        members=[o for o in visible if o['goal_id']==goal]
        candidate={color:sum(o['class_id']==cls for o in members) for color,cls in (('red',7),('blue',8),('yellow',9))}
        ambiguous=any(len(memberships[i])>1 for i in reached.get(goal,()))
        occluded=base is not None and any(intersection_fraction(base,o['xyxy'])>.2 for o in row.get('robots',[]))
        # A reference ROI alone cannot establish that the Goal is visible.
        observed=base is not None and any(o.get('class_id')==6 and intersection_fraction(base,o['xyxy'])>.25 for o in row.get('field_objects',[]))
        unknown=base is None or not observed or occluded or ambiguous
        # A robot covering the base does not erase already visible connected
        # halves (SC2 also permits robot contact). Preserve a known subtotal,
        # while keeping the unseen remainder explicitly incomplete.
        has_visible_half=any(candidate.values())
        counts[goal]=dict.fromkeys(('red','blue','yellow'),None) if unknown and not has_visible_half else candidate
        goals[goal]=dict(counts=candidate,objects=members,status='occluded' if occluded else 'ambiguous' if ambiguous else 'unobserved' if not observed else 'estimate',
                         partial=bool(unknown),scoring_validity='unconfirmed')
    regions={region:('neutral' if row.get('toggles',{}).get(site,{}).get('color')=='yellow' else
                     row.get('toggles',{}).get(site,{}).get('color','unknown')) for region,site in SITES.items()}
    regions={r:c if c in ('red','blue','neutral') else 'unknown' for r,c in regions.items()}
    result=match_score(counts,regions)
    for goal,view in goals.items():
        result['goals'][goal]['partial']=view['partial']
        if view['partial']:result['complete']=False
    # The visible colors are only a hypothesis of ownership; SC4 is not observed.
    return dict(goals=goals,objects=visible,score=result,regions=regions,
                mode='provisional',scoring_validity='unconfirmed',seconds=row['seconds'],
                excluded_ground_or_unconnected=len(objects)-len(visible))
