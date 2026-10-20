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

    def update(self, seconds, robots):
        if not math.isfinite(seconds) or (self.last_time is not None and seconds < self.last_time):
            raise ValueError('时间必须递增；跳转前请重置跟踪状态')
        self.last_time = seconds
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
            self.tracks[team] = dict(team=team, alliance=self.roster[team], xyxy=box,
                                     last_seen=seconds, velocity=velocity, track_id=obj.get('track_id'),
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
