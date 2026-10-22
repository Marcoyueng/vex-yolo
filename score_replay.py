"""Chronological replay estimates with explicit, time-scoped human corrections."""
from copy import deepcopy
from goal_memory import GoalStackMemory, detector_sample, retain_goal_counts, gate_scoring
from match_state import match_score


def reviewed_object_is_excluded(detail, obj, frame):
    """Exclude reviewed ineligible colors without hiding other stack halves."""
    if detail.get('reviewed_scoring_state') in ('nested_not_scored','ground_pins_not_scored','confirmed_no_scoring_halves'):
        return True
    color={7:'red',8:'blue',9:'yellow'}.get(obj['class_id'])
    review=detail.get('reviewed_counts',{}).get(color,{})
    return (review.get('value')==0 and review.get('count_mode','exact')=='exact'
            and review.get('frame',float('inf'))<=frame<=review.get('hold_until_frame',-1))


class ReplayScoring:
    def __init__(self, config):
        self.config=deepcopy(config)
        self.start=config['start_frame']
        self.auto_end=config['autonomous_end_frame']
        self.driver=config['driver_start_frame']
        if not 0<=self.start<self.auto_end<self.driver:
            raise ValueError('Invalid period boundaries')
        self.events=sorted(config.get('goal_reviews',[]),key=lambda e:e['frame'])
        if any(e['frame']<self.start for e in self.events):
            raise ValueError('Scoring corrections cannot precede match start')
        if any(e.get('hold_until_frame',e['frame'])<e['frame'] for e in self.events):
            raise ValueError('A reviewed interval cannot end before it starts')
        for event in self.events:
            if event.get('event_kind') in ('nested_not_scored','ground_pins_not_scored'):
                # These statuses describe a Goal with no eligible halves.
                # Mixed stacks need ordinary per-color count reviews instead.
                if ('hold_until_frame' not in event or
                        any(set(counts)!={'red','blue','yellow'} or any(v!=0 for v in counts.values())
                            for counts in event['goals'].values())):
                    raise ValueError('Goal-wide exclusion needs a reviewed interval and all three eligible counts set to zero')
            for goal,modes in event.get('count_modes',{}).items():
                for color,mode in modes.items():
                    if mode not in ('exact','minimum') or event['goals'].get(goal,{}).get(color) is None:
                        raise ValueError('Count mode requires a reviewed count and must be exact or minimum')
        self.memory=GoalStackMemory()
        self.event_index=0
        self.last_frame=-1
        self.reviews={}
        self.active_holds={}
        self.history=[]
        self.last_view=None

    def update(self,row):
        frame=row['frame']
        if frame<=self.last_frame:
            raise ValueError('Replay scoring must be rebuilt before seeking backward')
        self.last_frame=frame
        active=frame>=self.start
        phase=('prestart' if not active else 'autonomous' if frame<self.auto_end
               else 'intermission' if frame<self.driver else 'driver')
        valid=row.get('field_overlay_valid',True)
        if not valid and self.last_view is not None:
            # A broadcast cut is not evidence that goals or ownership changed.
            view=deepcopy(self.last_view)
            view['scene_valid']=False
            return view
        observation=row['goal_estimate']
        if active and valid:
            sample=detector_sample(row)
            # A cached sample captured before the start is not post-start evidence.
            if sample is not None and sample>=self.config['start_seconds']:
                self.memory.update(observation,sample)
            while self.event_index<len(self.events) and self.events[self.event_index]['frame']<=frame:
                event=self.events[self.event_index]
                corrected=deepcopy(event['goals'])
                snapshot=self.memory.snapshot()
                for goal,counts in corrected.items():
                    for color,value in counts.items():
                        if event.get('count_modes',{}).get(goal,{}).get(color)=='minimum':
                            counts[color]=max(value,snapshot[goal][color] or 0)
                self.memory.correct(corrected)
                for goal,counts in event['goals'].items():
                    self.reviews.setdefault(goal,{}).update({c:dict(value=v,frame=event['frame'],
                        source=event['source'],event_kind=event.get('event_kind','count_correction'),
                        count_mode=event.get('count_modes',{}).get(goal,{}).get(c,'exact'),
                        hold_until_frame=event.get('hold_until_frame',event['frame']),
                        rule_reference=event.get('rule_reference')) for c,v in counts.items() if v is not None})
                    for color,value in counts.items():
                        if value is not None:
                            mode=event.get('count_modes',{}).get(goal,{}).get(color,'exact')
                            self.active_holds[goal,color]=(event.get('hold_until_frame',event['frame']),value,mode)
                self.event_index+=1
            # A reviewed interval overrides noisy counts and clears pending
            # evidence. Afterwards, two fresh observations can add new halves.
            held={}
            snapshot=self.memory.snapshot()
            for (goal,color),(end,value,mode) in self.active_holds.items():
                if frame<=end:
                    if mode=='minimum' and (snapshot[goal][color] or 0)>=value:continue
                    held.setdefault(goal,{})[color]=value
            if held:self.memory.correct(held)
        view=retain_goal_counts(observation,self.memory.snapshot())
        bonus=self.config.get('bonus_review',{})
        bonus_ready=active and frame>=self.driver and frame>=bonus.get('frame',float('inf'))
        score=match_score(self.memory.snapshot(),observation['regions'],
            phase='autonomous' if frame<self.driver else 'end',
            bonus_red=bonus.get('red') if bonus_ready else None,
            bonus_blue=bonus.get('blue') if bonus_ready else None)
        for goal,detail in view['goals'].items():
            detail['reviewed_counts']=deepcopy(self.reviews.get(goal,{}))
            active_states={r['event_kind'] for r in detail['reviewed_counts'].values()
                           if frame<=r['hold_until_frame']}
            detail['reviewed_scoring_state']=next((state for state in
                ('nested_not_scored','ground_pins_not_scored','confirmed_no_scoring_halves','visible_scoring_halves')
                if state in active_states),None)
            detail['count_semantics']='visible_placed_halves_before_yellow_ownership'
            excluded=[obj for obj in detail.get('objects',[]) if reviewed_object_is_excluded(detail,obj,frame)]
            if excluded or detail['reviewed_scoring_state'] in ('nested_not_scored','ground_pins_not_scored','confirmed_no_scoring_halves'):
                detail['excluded_candidates']=deepcopy(excluded)
                detail['objects']=[obj for obj in detail.get('objects',[]) if obj not in excluded]
            detail['minimum_counts']={c:r['value'] for c,r in detail['reviewed_counts'].items() if r.get('count_mode')=='minimum'}
            if detail['minimum_counts']:detail['partial']=True
            score['goals'][goal]['minimum_counts']=deepcopy(detail['minimum_counts'])
            score['goals'][goal]['partial']=detail['partial']
            if detail['partial']:score['complete']=False
        view['score']=score
        view=gate_scoring(view,frame,self.start)
        view['match_timing'].update(period=phase,
            autonomous_end_frame=self.auto_end,driver_start_frame=self.driver)
        view['bonus_source']=bonus.get('source') if bonus_ready else None
        view['scoring_validity']='unconfirmed'
        view['scene_valid']=valid
        view['reviewed_incidents']=[deepcopy(e) for e in self.config.get('reviewed_incidents',[]) if e['frame']<=frame]
        # Scores are estimates: visual Toggle faces do not establish valid control.
        if active:view['score']['complete']=False
        point=dict(seconds=row['seconds'],red=view['score']['red'],blue=view['score']['blue'])
        if not self.history or point['seconds']-self.history[-1]['seconds']>=.2 or any(
                point[c]!=self.history[-1][c] for c in ('red','blue')):
            self.history.append(point)
        self.last_view=deepcopy(view)
        return view
