"""Persistent team state and reviewed SC3/SC5 score observations.

Prediction keeps identity visible; it is never evidence of a new identification.
Coordinates are normalized. A goal observation counts visible halves of Placed
Pins, not bounding boxes. Missing observations remain unknown.
"""
import math
from scoring import count


class TeamMemory:
    def __init__(self, roster, prediction_seconds=1.5):
        if len(roster) != 4 or list(roster.values()).count('red') != 2 or list(roster.values()).count('blue') != 2:
            raise ValueError('需要四个不同队号，红蓝各两队')
        self.roster = dict(roster)
        self.prediction_seconds = prediction_seconds
        self.tracks = {}
        self.last_time = None
        self.remaining_candidate = None

    def update(self, seconds, robots):
        if not math.isfinite(seconds) or (self.last_time is not None and seconds < self.last_time):
            raise ValueError('时间必须递增；跳转前请重置跟踪状态')
        self.last_time = seconds
        self.reconnect(seconds, robots)
        seen = set()
        for team in self.roster:
            candidates = [r for r in robots if r.get('team') == team]
            if not candidates:
                continue
            obj = max(candidates, key=lambda r: r.get('confidence', 0))
            box = list(obj['xyxy'])
            if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('机器人坐标必须是有效归一化框')
            old = self.tracks.get(team)
            velocity = [0.] * 4
            if old and 0 < seconds-old['last_seen'] <= .5:
                dt = seconds-old['last_seen']
                # Bound motion; do not let a single noisy detection launch a box.
                velocity = [max(-.25, min(.25, (a-b)/dt)) for a, b in zip(box, old['xyxy'])]
            continuous = old and old.get('track_id') == obj.get('track_id') and seconds-old['last_seen'] <= .5
            self.tracks[team] = dict(team=team, alliance=self.roster[team], xyxy=box,
                                     last_seen=seconds, velocity=velocity, track_id=obj.get('track_id'),
                                     stable_since=old.get('stable_since',seconds) if continuous else seconds,
                                     identity_source=obj.get('identity_source', 'associated'))
            seen.add(team)
        result = []
        for team, alliance in self.roster.items():
            previous = self.tracks.get(team)
            if previous is None:
                result.append(dict(team=team, alliance=alliance, status='waiting', xyxy=None, age=None))
                continue
            row = dict(previous)
            age = seconds-row['last_seen']
            row.update(age=age, status='observed' if team in seen else 'predicted' if age <= self.prediction_seconds else 'lost')
            if row['status'] == 'predicted':
                # Translate the box, preserving its size at image boundaries.
                x1,y1,x2,y2=row['xyxy']; v=row['velocity']
                dx=max(-x1,min(1-x2,(v[0]+v[2])/2*age))
                dy=max(-y1,min(1-y2,(v[1]+v[3])/2*age))
                row['xyxy']=[x1+dx,y1+dy,x2+dx,y2+dy]
            if row['status'] == 'lost':
                row['xyxy'] = None
            result.append(row)
        return result

    def reconnect(self, seconds, robots):
        self._spatial_reconnect(seconds, robots)
        self._remaining_reconnect(seconds, robots)

    def _remaining_reconnect(self, seconds, robots):
        """Recover the sole missing team from a sustained four-robot observation.

        This is a roster/continuity hypothesis. Three stable peers and one
        distinct, confident robot are required; no new box is synthesized.
        Unlike spatial recovery, elapsed occlusion time is not a disqualifier.
        """
        pending = self.remaining_candidate
        self.remaining_candidate = None
        if len(robots) != 4:
            return
        known = [o for o in robots if o.get('team') in self.roster]
        unknown = [o for o in robots if o.get('team') is None]
        present = {o['team'] for o in known}
        if len(known) != 3 or len(present) != 3 or len(unknown) != 1:
            return
        team = next(iter(set(self.roster)-present))
        if team not in self.tracks:
            return
        obj = unknown[0]
        if obj.get('track_id') is None or obj.get('confidence',0) < .35:
            return
        if obj.get('field_validated') is False or obj.get('class_id',0) != 0:
            return
        if obj.get('identity_source') in ('reviewed_overlap_uncertain','superseded_review','identity_conflict'):
            return
        if obj.get('alliance') not in (None,self.roster[team]):
            return
        box = obj.get('xyxy',[])
        if len(box)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in box) or box[2]<=box[0] or box[3]<=box[1]:
            return
        if len({o.get('track_id') for o in robots}) != 4:
            return
        for peer in known:
            old = self.tracks.get(peer['team'])
            if old is None or seconds-old['last_seen']>.5 or old.get('track_id')!=peer.get('track_id'):
                return
            if seconds-old.get('stable_since',seconds)<.25:
                return
            b=peer['xyxy']
            intersection=max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1]))
            area=(box[2]-box[0])*(box[3]-box[1]); other=(b[2]-b[0])*(b[3]-b[1])
            if intersection/max(1e-12,area+other-intersection)>.65:
                return
            # A small duplicate crop inside a known robot is not a fourth car.
            if intersection/max(1e-12,min(area,other))>.9:
                return
        signature=(team,obj['track_id'],tuple(sorted((p['team'],p['track_id']) for p in known)))
        if pending is not None:
            previous=pending['box']
            displacement=math.hypot((box[0]+box[2]-previous[0]-previous[2])/2,
                                    (box[1]+box[3]-previous[1]-previous[3])/2)
            if displacement>.06+.4*max(0,seconds-pending['last']):
                pending=None
        if pending is None or pending['signature']!=signature or not 0<seconds-pending['last']<=.15:
            pending=dict(signature=signature,start=seconds,last=seconds,frames=1,box=list(box))
        else:
            pending=dict(pending,last=seconds,frames=pending['frames']+1,box=list(box))
        self.remaining_candidate=pending
        if pending['frames']>=3 and seconds-pending['start']>=.1-1e-9:
            obj.update(team=team,alliance=self.roster[team],identity_source='unique_remaining_robot_reconnect',
                       identity_evidence='Three stable known peers and one distinct robot over multiple frames')
            self.remaining_candidate=None

    def _spatial_reconnect(self, seconds, robots):
        """Recover a recent lost identity only with three independently visible peers.

        Updates the current detection for both overlays and cards. A spatial
        recovery remains a tracking hypothesis, never a reviewed team label.
        """
        present = {r.get('team') for r in robots if r.get('team') in self.roster}
        missing = set(self.roster)-present
        if len(missing) != 1:
            return
        team = next(iter(missing)); old = self.tracks.get(team)
        if old is None or not 0 <= seconds-old['last_seen'] <= 2:
            return
        # A peer which just changed tracker ID is not a reliable spatial anchor.
        for peer in present:
            previous = self.tracks.get(peer)
            current = [r for r in robots if r.get('team') == peer]
            if len(current) != 1 or previous is None or seconds-previous['last_seen'] > .5:
                return
            if previous.get('track_id') is None or current[0].get('track_id') != previous['track_id']:
                return
        box = old['xyxy']; age = seconds-old['last_seen']; velocity = old['velocity']
        cx=(box[0]+box[2])/2+(velocity[0]+velocity[2])/2*age
        cy=(box[1]+box[3])/2+(velocity[1]+velocity[3])/2*age
        choices=[]
        for obj in robots:
            if obj.get('team') is not None or obj.get('track_id') is None:
                continue
            if obj.get('field_validated') is False or obj.get('class_id',0)!=0:
                continue
            if obj.get('identity_source') in ('reviewed_overlap_uncertain','superseded_review','identity_conflict'):
                continue
            b=obj.get('xyxy',[])
            if len(b)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in b) or b[2]<=b[0] or b[3]<=b[1]:
                continue
            if obj.get('alliance') not in (None,self.roster[team]):
                continue
            if any(t!=team and obj['track_id']==s.get('track_id') for t,s in self.tracks.items()):
                continue
            area=(b[2]-b[0])*(b[3]-b[1]); previous_area=(box[2]-box[0])*(box[3]-box[1])
            if not .3 <= area/previous_area <= 3:
                continue
            # A new crop of a visible peer is not the missing robot, even if
            # it is close to the lost position. This matters during overlaps.
            duplicate_peer=False
            for peer in robots:
                if peer.get('team') not in present:continue
                p=peer['xyxy'];other=(p[2]-p[0])*(p[3]-p[1])
                intersection=max(0,min(b[2],p[2])-max(b[0],p[0]))*max(0,min(b[3],p[3])-max(b[1],p[1]))
                if intersection/max(1e-12,area+other-intersection)>.65 or intersection/max(1e-12,min(area,other))>.9:
                    duplicate_peer=True;break
            if duplicate_peer:continue
            distance=math.hypot((b[0]+b[2])/2-cx,(b[1]+b[3])/2-cy)
            if distance <= min(.18,.08+.05*age):choices.append((distance,obj))
        choices.sort(key=lambda pair:pair[0])
        if not choices or (len(choices)>1 and choices[1][0]-choices[0][0]<.04):
            return
        choices[0][1].update(team=team,alliance=self.roster[team],identity_source='recent_lost_spatial_reconnect')


def goal_scores(goals, owners):
    """Per-goal red/blue subtotals; owner is reviewed red/blue/neutral/unknown.

    G0 midfield; G1-2 red1, G3-4 red2, G5-6 blue1, G7-8 blue2.
    End-midfield robot points and autonomous bonus are separate match items.
    """
    keys = {f'G{i}' for i in range(9)}
    if set(goals) != keys or set(owners) != keys:
        raise ValueError('需要 G0–G8 九个 Goal')
    result = dict(red=0, blue=0, complete=True, goals={})
    for goal in sorted(keys):
        observation = goals[goal]
        owner = owners[goal]
        if owner not in ('red', 'blue', 'neutral', 'unknown'):
            raise ValueError('无效黄色半段归属')
        values = [observation.get(c) for c in ('red','blue','yellow')]
        for value in values:
            if value is not None:
                count(value)
        red, blue, yellow = values
        yellow_points = 0 if owner == 'neutral' or yellow == 0 else None if owner == 'unknown' or yellow is None else 10*yellow
        row = dict(red=None if red is None or (owner in ('red','unknown') and yellow_points is None) else red*5+(yellow_points if owner == 'red' else 0),
                   blue=None if blue is None or (owner in ('blue','unknown') and yellow_points is None) else blue*5+(yellow_points if owner == 'blue' else 0),
                   owner=owner, counts=dict(zip(('red','blue','yellow'),values)))
        for color in ('red','blue'):
            if row[color] is None:
                result['complete'] = False
            # Keep guaranteed contributions even when the yellow portion is unknown.
            own_halves = observation.get(color)
            result[color] += (0 if own_halves is None else own_halves*5)
            if owner == color and yellow_points is not None:
                result[color] += yellow_points
        result['goals'][goal] = row
    return result


def match_score(goals, regions, midfield_red=None, midfield_blue=None, phase='end', bonus_red=None, bonus_blue=None):
    """Standard V5RC estimates. Bonus is reviewed separately (SC7 violations).

    Live robot centers cannot establish the infinitely tall midfield volume.
    Bonus inputs are explicit reviewed points; they are never inferred from a
    partial autonomous subtotal. The current period subtotal remains incomplete
    when required end-state observations or bonuses are missing.
    """
    from schema import REGIONS
    if set(regions) != set(REGIONS) or phase not in ('end','autonomous'):
        raise ValueError('需要四个区域及正确比赛阶段')
    if phase == 'end' and bonus_red is not None and bonus_blue is not None and (bonus_red,bonus_blue) not in ((12,0),(0,12),(6,6),(0,0)):
        raise ValueError('请核查自动奖励：单方 12 分、平局各 6 分，或均无奖励')
    for number in (midfield_red, midfield_blue):
        if number is not None:
            count(number, 2)
    middle = ('neutral' if phase == 'autonomous' else 'unknown' if midfield_red is None or midfield_blue is None else
              'red' if midfield_red > midfield_blue else 'blue' if midfield_blue > midfield_red else 'neutral')
    owners = {f'G{i}': middle if i == 0 else regions[REGIONS[(i-1)//2]] for i in range(9)}
    result = goal_scores(goals, owners)
    result['goal_red'], result['goal_blue'] = result['red'], result['blue']
    for color, robots, bonus in (('red', midfield_red, bonus_red), ('blue', midfield_blue, bonus_blue)):
        if bonus is not None and (type(bonus) is not int or bonus not in (0,6,12)):
            raise ValueError('自动奖励必须为 0、6、12 或未知')
        parking = 0 if phase == 'autonomous' else None if robots is None else robots*8
        applied_bonus = 0 if phase == 'autonomous' else bonus
        result[f'{color}_midfield'] = parking
        result[f'{color}_bonus'] = applied_bonus
        if parking is None or applied_bonus is None:
            result['complete'] = False
        result[color] += (parking or 0)+(applied_bonus or 0)
    return result
