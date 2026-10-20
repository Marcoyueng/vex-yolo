"""Optional legal-match persistence assumption for already placed halves.

Two independent detector samples are required before accepting an increase.
This is an estimate, not a referee/DQ decision. Manual review may correct it.
"""
from copy import deepcopy
from collections import deque
from bisect import bisect_right
from match_state import match_score

COLORS=('red','blue','yellow')


def detector_sample(row):
    times=[o.get('sample_seconds',row['seconds']) for o in row.get('field_objects',[])
           if o.get('observation')=='draft_model_detection']
    return max(times) if times else None


class GoalStackMemory:
    def __init__(self,confirmations=2,max_gap=3.):
        if confirmations<2:raise ValueError('At least two independent samples are required')
        self.confirmations=confirmations;self.max_gap=max_gap
        self.counts={f'G{i}':dict.fromkeys(COLORS,None) for i in range(9)}
        self.pending={(g,c):deque(maxlen=confirmations) for g in self.counts for c in COLORS}
        self.last_sample=None

    def update(self,observation,sample):
        if sample is None:return
        if self.last_sample is not None and sample<self.last_sample:raise ValueError('Rebuild memory before seeking backward')
        if sample==self.last_sample:return # propagated frames are not new evidence
        self.last_sample=sample
        for goal,view in observation['goals'].items():
            for color in COLORS:
                value=view['counts'][color]
                if value is None or (view.get('partial',False) and value==0):continue
                if type(value) is not int or value<0:raise ValueError('Invalid half count')
                evidence=self.pending[goal,color]
                if evidence and sample-evidence[-1][0]>self.max_gap:evidence.clear()
                evidence.append((sample,value))
                if len(evidence)==self.confirmations:
                    accepted=min(v for _,v in evidence)
                    previous=self.counts[goal][color]
                    self.counts[goal][color]=accepted if previous is None else max(previous,accepted)

    def correct(self,goals):
        """Explicit reviewed correction may lower an erroneous retained count."""
        for goal,values in goals.items():
            if goal not in self.counts:continue
            for color in COLORS:
                value=values.get(color)
                if value is None:continue
                if type(value) is not int or value<0:raise ValueError('Invalid correction')
                self.counts[goal][color]=value;self.pending[goal,color].clear()

    def snapshot(self):return deepcopy(self.counts)


def retain_goal_counts(observation,counts):
    result=dict(observation,goals={})
    for goal,view in observation['goals'].items():
        values=dict(counts[goal])
        held=any(v is not None and v>=(view['counts'].get(c) or 0) and
                 (v>(view['counts'].get(c) or 0) or view.get('partial')) for c,v in values.items())
        pending=any(v is None or (view['counts'].get(c) or 0)>v for c,v in values.items())
        partial=view.get('partial',False) or pending
        status='remembered' if held else 'pending' if pending else 'stable'
        result['goals'][goal]=dict(view,counts=values,observed_counts=dict(view['counts']),
                                  status=status,partial=partial,memory_held=held)
    score=match_score(counts,observation['regions'])
    for goal,view in result['goals'].items():
        score['goals'][goal]['partial']=view['partial']
        if view['partial']:score['complete']=False
    result.update(score=score,persistence=dict(enabled=True,assumption='placed_halves_remain',scoring_validity='unconfirmed'))
    return result


class GoalMemoryTimeline:
    """Small change-point snapshots, always looked up at or before the frame."""
    def __init__(self,records,fps,goal_rois=None,reviews=(),resets=()):
        from goal_observer import observe_goal_stacks
        keeper=GoalStackMemory();self.frames=[-1];self.snapshots=[keeper.snapshot()];self.history=[]
        reviews={r['frame']:r for r in reviews};resets=set(resets)
        for index in range(len(records.offsets)):
            row=records.get(index);sample=detector_sample(row)
            chart=index%max(1,round(fps))==0
            if sample is None and not chart and index not in reviews and index not in resets and not row.get('reviewed_goal_counts'):continue
            observation=row.get('goal_estimate') if not row.get('goal_rois') and not goal_rois else None
            if observation is None:observation=observe_goal_stacks(row,goal_rois)
            if index in resets:keeper=GoalStackMemory()
            keeper.update(observation,sample)
            if row.get('reviewed_goal_counts') and index not in resets:
                keeper.correct(row['reviewed_goal_counts'])
            if index in reviews:keeper.correct(reviews[index]['input']['goals'])
            snapshot=keeper.snapshot()
            if snapshot!=self.snapshots[-1]:
                self.frames.append(index);self.snapshots.append(snapshot)
            if chart:
                score=retain_goal_counts(observation,snapshot)['score']
                self.history.append(dict(seconds=row['seconds'],red=score['red'],blue=score['blue']))

    def view(self,frame,observation):
        counts=self.snapshots[bisect_right(self.frames,frame)-1]
        return retain_goal_counts(observation,counts)
