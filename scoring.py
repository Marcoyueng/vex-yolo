"""V5RC head-to-head yellow Pin Half attribution, SC2-SC7 (2.1).

Inputs are reviewed observations, never raw detector box counts.
"""
from schema import REGIONS


def toggle_owner(color, settled, robot_contact):
    if color not in ('red', 'blue', 'yellow', 'unknown'):
        raise ValueError('Invalid toggle color')
    if settled not in (True, False, None) or robot_contact not in (True, False, None):
        raise ValueError('Invalid toggle validity')
    if settled is False or robot_contact is True:
        return 'neutral'
    if color == 'unknown' or settled is None or robot_contact is None:
        return 'unknown'
    return color if color in ('red', 'blue') else 'neutral'


def count(value, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError('计数必须是非负整数；标准 V5RC 每方最多两台机器人。')
    return value


def yellow_score(toggles, halves_by_goal, midfield_red=None, midfield_blue=None, phase='end'):
    """Nine project-local goals: G0 midfield; paired G1-G8 follow REGIONS.

    None is unknown, not zero. Counts include only visible yellow halves on
    confirmed Placed Pins. No autonomous bonus, penalties, or skills/VURC rules.
    """
    if phase not in ('end', 'autonomous') or set(toggles) != set(REGIONS):
        raise ValueError('必须指定四个区域及正确比赛阶段。')
    if set(halves_by_goal) != {f'G{i}' for i in range(9)}:
        raise ValueError('必须指定 G0–G8；未知数量请填 None。')
    owners = {region: toggle_owner(**toggles[region]) for region in REGIONS}
    for robots in (midfield_red, midfield_blue):
        if robots is not None:
            count(robots, 2)
    midfield = ('neutral' if phase == 'autonomous' else
                'unknown' if midfield_red is None or midfield_blue is None else
                'red' if midfield_red > midfield_blue else 'blue' if midfield_blue > midfield_red else 'neutral')
    result = {'red': 0, 'blue': 0, 'complete': True, 'goals': {}, 'owners': owners}
    for i in range(9):
        goal = f'G{i}'
        owner = midfield if i == 0 else owners[REGIONS[(i-1)//2]]
        halves = halves_by_goal[goal]
        if halves is not None:
            count(halves)
        points = 0 if owner == 'neutral' or halves == 0 else None if owner == 'unknown' or halves is None else halves*10
        if points is None:
            result['complete'] = False
        elif owner in ('red', 'blue'):
            result[owner] += points
        result['goals'][goal] = {'owner': owner, 'yellow_halves': halves, 'points': points}
    return result
