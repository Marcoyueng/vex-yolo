import unittest
from types import SimpleNamespace
from goal_memory import GoalStackMemory,retain_goal_counts,GoalMemoryTimeline,gate_scoring


def observation(yellow=1,red=1,blue=1,owner='red',partial=False):
    counts={f'G{i}':dict(red=0,blue=0,yellow=0) for i in range(9)}
    counts['G1']=dict(red=red,blue=blue,yellow=yellow)
    return dict(seconds=1,goals={g:dict(counts=c,partial=partial,status='estimate',objects=[]) for g,c in counts.items()},
                regions={r:owner for r in ('red1','red2','blue1','blue2')},objects=[],scoring_validity='unconfirmed')


class GoalMemoryTests(unittest.TestCase):
    def test_prestart_false_positive_cannot_seed_live_score(self):
        rows=[]
        for i,n in enumerate((5,5,1,1)):
            o=observation(yellow=n,red=0,blue=0)
            rows.append(dict(frame=i,seconds=i,goal_estimate=o,field_objects=[dict(observation='draft_model_detection',sample_seconds=i)]))
        records=SimpleNamespace(offsets=list(range(4)),get=lambda i:rows[i])
        timeline=GoalMemoryTimeline(records,1,start_frame=2)
        before=timeline.view(1,rows[1]['goal_estimate'])
        self.assertEqual(before['score']['red'],0)
        self.assertEqual(before['match_timing']['state'],'prestart')
        self.assertEqual(timeline.view(3,rows[3]['goal_estimate'])['goals']['G1']['counts']['yellow'],1)
        self.assertEqual(timeline.history[1]['red'],0)
        unset=GoalMemoryTimeline(records,1,start_frame=None)
        self.assertEqual(unset.view(3,rows[3]['goal_estimate'])['score']['red'],0)
        self.assertIsNone(unset.snapshots[-1]['G1']['yellow'])

    def test_one_false_detection_and_propagated_copies_cannot_lock_count(self):
        m=GoalStackMemory();m.update(observation(yellow=5),0)
        m.update(observation(yellow=5),0);m.update(observation(yellow=5),None)
        m.update(observation(yellow=1),1)
        self.assertEqual(m.snapshot()['G1']['yellow'],1)

    def test_missing_halves_do_not_remove_points_but_toggle_reassigns_yellow(self):
        m=GoalStackMemory();m.update(observation(yellow=5),0);m.update(observation(yellow=5),1)
        missing=observation(yellow=0,red=0,blue=0,partial=True)
        m.update(missing,2);m.update(missing,3)
        red=retain_goal_counts(missing,m.snapshot())
        self.assertEqual(red['score']['goals']['G1']['red'],55)
        self.assertEqual(red['score']['goals']['G1']['blue'],5)
        blue=retain_goal_counts(observation(yellow=0,red=0,blue=0,owner='blue',partial=True),m.snapshot())
        self.assertEqual(blue['score']['goals']['G1']['red'],5)
        self.assertEqual(blue['score']['goals']['G1']['blue'],55)
        self.assertEqual(blue['goals']['G1']['counts']['yellow'],5)
        self.assertEqual(blue['scoring_validity'],'unconfirmed')

    def test_manual_correction_may_lower_a_bad_count(self):
        m=GoalStackMemory();m.update(observation(yellow=5),0);m.update(observation(yellow=5),1)
        m.correct({'G1':dict(yellow=3)})
        self.assertEqual(m.snapshot()['G1']['yellow'],3)
        m.update(observation(yellow=5),2)
        self.assertEqual(m.snapshot()['G1']['yellow'],3)

    def test_new_match_has_no_memory_and_reverse_updates_are_rejected(self):
        m=GoalStackMemory();m.update(observation(),1)
        with self.assertRaises(ValueError):m.update(observation(),0)
        self.assertIsNone(GoalStackMemory().snapshot()['G1']['yellow'])

    def test_timeline_seek_cannot_use_future_counts(self):
        rows=[]
        for i,n in enumerate((1,1,5,5)):
            o=observation(yellow=n)
            rows.append(dict(frame=i,seconds=i,goal_estimate=o,field_objects=[dict(observation='draft_model_detection',sample_seconds=i)]))
        records=SimpleNamespace(offsets=list(range(4)),get=lambda i:rows[i])
        timeline=GoalMemoryTimeline(records,1)
        self.assertEqual(timeline.view(3,observation())['goals']['G1']['counts']['yellow'],5)
        self.assertEqual(timeline.view(1,observation())['goals']['G1']['counts']['yellow'],1)
        self.assertIsNone(timeline.view(0,observation())['goals']['G1']['counts']['yellow'])
        reset=GoalMemoryTimeline(records,1,resets=[3])
        self.assertIsNone(reset.view(3,observation())['goals']['G1']['counts']['yellow'])


if __name__=='__main__':unittest.main()
