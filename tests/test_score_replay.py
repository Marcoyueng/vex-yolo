import unittest
from score_replay import ReplayScoring, reviewed_object_is_excluded
from tests.test_goal_memory import observation


class ScoreReplayTests(unittest.TestCase):
    def config(self):
        return dict(start_frame=2,start_seconds=2,autonomous_end_frame=7,driver_start_frame=9,
            goal_reviews=[dict(frame=4,goals={'G4':dict(red=1,yellow=1)},source='user')],
            bonus_review=dict(frame=10,red=12,blue=0,source='reviewed_broadcast'))

    def row(self,i,owner='red',missing=False):
        o=observation(yellow=0,red=0,blue=0,owner=owner,partial=missing)
        o['goals']['G0']['counts']['yellow']=2
        return dict(frame=i,seconds=i,goal_estimate=o,field_objects=[
            dict(observation='draft_model_detection',sample_seconds=i)])

    def test_g4_is_time_scoped_then_persists_and_changes_yellow_owner(self):
        m=ReplayScoring(self.config());views=[m.update(self.row(i)) for i in range(5)]
        self.assertEqual(views[1]['score']['red'],0)
        self.assertFalse(views[3]['goals']['G4']['reviewed_counts'])
        self.assertEqual(views[4]['score']['goals']['G4']['red'],15)
        v=m.update(self.row(5,'blue',True))
        self.assertEqual((v['score']['goals']['G4']['red'],v['score']['goals']['G4']['blue']),(5,10))
        self.assertEqual(v['goals']['G4']['counts'],dict(red=1,blue=0,yellow=1))
        self.assertEqual(m.update(self.row(6,'neutral',True))['score']['goals']['G4']['red'],5)

    def test_autonomous_excludes_midfield_and_does_not_guess_bonus(self):
        m=ReplayScoring(self.config());vs=[m.update(self.row(i)) for i in range(11)]
        self.assertEqual(vs[6]['score']['goals']['G0']['red'],0)
        self.assertEqual(vs[6]['score']['red_midfield'],0)
        self.assertEqual(vs[8]['match_timing']['period'],'intermission')
        self.assertEqual(vs[8]['score']['red_bonus'],0)
        self.assertIsNone(vs[9]['score']['red_bonus'])
        self.assertEqual(vs[10]['score']['red_bonus'],12)
        self.assertIsNone(vs[10]['score']['red_midfield'])
        self.assertFalse(vs[10]['score']['complete'])

    def test_prestart_and_propagated_samples_do_not_seed_counts(self):
        m=ReplayScoring(self.config())
        for i in range(4):
            r=self.row(i);r['goal_estimate']['goals']['G1']['counts']['red']=9
            if i>=2:r['field_objects'][0]['sample_seconds']=1
            v=m.update(r)
        self.assertIsNone(v['goals']['G1']['counts']['red'])
        with self.assertRaises(ValueError):m.update(self.row(1))

    def test_partial_review_does_not_invent_other_color_counts(self):
        m=ReplayScoring(self.config())
        for i in range(5):v=m.update(self.row(i,missing=True))
        self.assertIsNone(v['goals']['G4']['counts']['blue'])
        self.assertEqual(v['score']['red'],15)

    def test_broadcast_cut_cannot_change_retained_score(self):
        m=ReplayScoring(self.config())
        for i in range(5):before=m.update(self.row(i))
        cut=self.row(5,'blue');cut['field_overlay_valid']=False
        after=m.update(cut)
        self.assertEqual(after['score'],before['score'])
        self.assertFalse(after['scene_valid'])

    def test_reviewed_removal_lowers_counts_and_rejects_noise_during_hold(self):
        c=self.config()
        c['goal_reviews']=[dict(frame=4,hold_until_frame=5,goals={'G5':dict(red=1,blue=1,yellow=1)},source='review'),
            dict(frame=6,hold_until_frame=8,goals={'G5':dict(red=0,blue=0,yellow=1)},source='review',event_kind='observed_removal')]
        m=ReplayScoring(c)
        for i in range(11):
            r=self.row(i,'blue')
            r['goal_estimate']['goals']['G5']['counts']=dict(red=1,blue=1,yellow=2)
            v=m.update(r)
            if i==5:self.assertEqual(v['goals']['G5']['counts'],dict(red=1,blue=1,yellow=1))
            if 6<=i<=9:self.assertEqual(v['goals']['G5']['counts'],dict(red=0,blue=0,yellow=1))
        self.assertEqual(v['goals']['G5']['counts'],dict(red=1,blue=1,yellow=2))

    def test_minimum_review_is_time_scoped_and_does_not_cap_further_growth(self):
        c=self.config()
        c['goal_reviews']=[dict(frame=4,hold_until_frame=8,goals={'G1':dict(red=4,yellow=1)},
            count_modes={'G1':dict(red='minimum')},source='user at least four')]
        m=ReplayScoring(c)
        for i in range(9):
            r=self.row(i)
            if i>=5:r['goal_estimate']['goals']['G1']['counts']['red']=5
            v=m.update(r);g=v['goals']['G1']
            if i<4:self.assertFalse(g['minimum_counts'])
            else:
                self.assertEqual(g['minimum_counts'],dict(red=4))
                self.assertTrue(g['partial'])
                self.assertEqual(g['counts']['red'],4 if i<=5 else 5)
                self.assertEqual(v['score']['goals']['G1']['red'],30 if i<=5 else 35)

    def test_minimum_does_not_lower_higher_count_and_removal_supersedes_it(self):
        c=self.config()
        c['goal_reviews']=[dict(frame=4,hold_until_frame=8,goals={'G1':dict(red=4)},
            count_modes={'G1':dict(red='minimum')},source='minimum'),
            dict(frame=6,goals={'G1':dict(red=1)},source='reviewed removal',event_kind='observed_removal')]
        m=ReplayScoring(c)
        for i in range(7):
            r=self.row(i);r['goal_estimate']['goals']['G1']['counts']['red']=6
            v=m.update(r)
            if i==4:self.assertEqual(v['goals']['G1']['counts']['red'],6)
        self.assertEqual(v['goals']['G1']['counts']['red'],1)
        self.assertFalse(v['goals']['G1']['minimum_counts'])

    def test_nested_half_stops_scoring_and_new_placement_can_score_again(self):
        c=self.config()
        c['goal_reviews']=[dict(frame=4,hold_until_frame=5,goals={'G6':dict(red=0,blue=0,yellow=0)},
            source='confirmed opaque Cup nesting',event_kind='nested_not_scored',rule_reference='SC3')]
        m=ReplayScoring(c)
        for i in range(8):
            r=self.row(i,'blue');r['goal_estimate']['goals']['G6']['counts']=dict(red=0,blue=1,yellow=1)
            v=m.update(r);g=v['goals']['G6']
            if i==3:self.assertEqual(v['score']['goals']['G6']['blue'],15)
            if i in (4,5,6):self.assertEqual(g['counts'],dict(red=0,blue=0,yellow=0))
            if i in (4,5):self.assertEqual(g['reviewed_scoring_state'],'nested_not_scored')
            if i==7:
                self.assertEqual(v['score']['goals']['G6']['blue'],15)
                self.assertIsNone(g['reviewed_scoring_state'])

    def test_camera_occlusion_alone_does_not_mean_nested(self):
        c=self.config();c['goal_reviews']=[];m=ReplayScoring(c)
        for i in range(5):
            r=self.row(i,'blue')
            r['goal_estimate']['goals']['G6']['counts']=dict(red=0,blue=0,yellow=1 if i<4 else 0)
            r['goal_estimate']['goals']['G6']['partial']=i==4
            v=m.update(r)
        self.assertEqual(v['goals']['G6']['counts']['yellow'],1)
        self.assertIsNone(v['goals']['G6']['reviewed_scoring_state'])

    def test_removed_yellow_is_excluded_without_hiding_remaining_red(self):
        c=self.config();c['goal_reviews']=[dict(frame=4,hold_until_frame=6,
            goals={'G4':dict(red=1,blue=0,yellow=0)},source='reviewed yellow fell',event_kind='observed_removal')]
        m=ReplayScoring(c)
        red=dict(class_id=7,goal_id='G4');yellow=dict(class_id=9,goal_id='G4')
        for i in range(8):
            r=self.row(i,'blue');r['goal_estimate']['goals']['G4']['objects']=[red,yellow]
            v=m.update(r);g=v['goals']['G4']
            self.assertFalse(reviewed_object_is_excluded(g,red,i))
            self.assertEqual(reviewed_object_is_excluded(g,yellow,i),4<=i<=6)
            if 4<=i<=6:
                self.assertEqual(g['objects'],[red]);self.assertEqual(g['excluded_candidates'],[yellow])
                self.assertEqual((v['score']['goals']['G4']['red'],v['score']['goals']['G4']['blue']),(5,0))

    def test_nesting_review_requires_zero_eligible_halves_and_scoped_interval(self):
        for overrides in ({'hold_until_frame':5,'goals':{'G6':{'yellow':1}}},
                          {'goals':{'G6':{'yellow':0}}},
                          {'hold_until_frame':5,'goals':{'G6':{'yellow':0}}}):
            c=self.config();c['goal_reviews']=[dict(frame=4,source='review',event_kind='nested_not_scored',**overrides)]
            with self.assertRaises(ValueError):ReplayScoring(c)

    def test_ground_pin_review_excludes_candidates_but_keeps_original_observation(self):
        c=self.config();c['goal_reviews']=[dict(frame=4,hold_until_frame=8,
            goals={'G6':dict(red=0,blue=0,yellow=0)},source='user confirms floor Pin',event_kind='ground_pins_not_scored')]
        m=ReplayScoring(c)
        for i in range(5):
            r=self.row(i,'blue');r['goal_estimate']['goals']['G6']['objects']=[dict(class_id=8,goal_id='G6')]
            v=m.update(r)
        self.assertEqual(v['goals']['G6']['reviewed_scoring_state'],'ground_pins_not_scored')
        self.assertEqual(v['goals']['G6']['objects'],[])
        self.assertEqual(len(v['goals']['G6']['excluded_candidates']),1)
        self.assertEqual(len(r['goal_estimate']['goals']['G6']['objects']),1)


if __name__=='__main__':unittest.main()
