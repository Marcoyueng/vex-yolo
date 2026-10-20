import unittest
from match_state import TeamMemory, goal_scores, match_score
from schema import REGIONS


ROSTER={'1744X':'red','471A':'red','88S':'blue','2899A':'blue'}


class MatchStateTests(unittest.TestCase):
    def goals(self):
        return {f'G{i}':dict(red=0,blue=0,yellow=0) for i in range(9)}

    def test_four_cards_survive_occlusion_without_inventing_detection(self):
        memory=TeamMemory(ROSTER)
        obj=dict(team='1744X',xyxy=[.1,.2,.2,.3],confidence=.9,track_id=1)
        self.assertEqual(memory.update(0,[obj])[0]['status'],'observed')
        states=memory.update(.2,[])
        self.assertEqual(len(states),4)
        self.assertEqual(states[0]['status'],'predicted')
        self.assertEqual(states[1]['status'],'waiting')
        states=memory.update(2,[])
        self.assertEqual(states[0]['status'],'lost')
        self.assertIsNone(states[0]['xyxy'])
        self.assertEqual(states[0]['team'],'1744X')
        self.assertEqual(states[0]['last_seen'],0)

    def test_position_never_changes_alliance_and_new_track_recovers_team(self):
        memory=TeamMemory(ROSTER)
        memory.update(0,[dict(team='88S',xyxy=[.8,.8,.9,.9],track_id=4)])
        states=memory.update(3,[dict(team='88S',xyxy=[.1,.1,.2,.2],track_id=58,alliance='red')])
        self.assertEqual(states[2]['alliance'],'blue')
        self.assertEqual(states[2]['track_id'],58)
        self.assertEqual(states[2]['status'],'observed')

    def test_prediction_stays_inside_image(self):
        memory=TeamMemory(ROSTER)
        memory.update(0,[dict(team='471A',xyxy=[.8,.8,.9,.9])])
        memory.update(.1,[dict(team='471A',xyxy=[.85,.85,.95,.95])])
        box=memory.update(1,[])[1]['xyxy']
        self.assertTrue(all(0<=v<=1 for v in box))
        self.assertAlmostEqual(box[2]-box[0],.1)
        with self.assertRaises(ValueError):memory.update(.5,[])

    def test_sc3_visible_halves_and_sc5_yellow_owner(self):
        goals=self.goals();goals['G1']=dict(red=2,blue=1,yellow=3)
        owners={g:'neutral' for g in goals};owners['G1']='blue'
        score=goal_scores(goals,owners)
        self.assertEqual((score['red'],score['blue']),(10,35))
        self.assertTrue(score['complete'])

    def test_unknown_is_not_zero_even_with_known_other_color(self):
        goals=self.goals();goals['G1']=dict(red=2,blue=1,yellow=3)
        owners={g:'neutral' for g in goals};owners['G1']='unknown'
        score=goal_scores(goals,owners)
        self.assertFalse(score['complete'])
        self.assertIsNone(score['goals']['G1']['red'])
        self.assertIsNone(score['goals']['G1']['blue'])
        self.assertEqual((score['red'],score['blue']),(10,5))
        owners['G1']='neutral'
        self.assertEqual(goal_scores(goals,owners)['goals']['G1']['red'],10)

    def test_midfield_and_autonomous_exclusion_and_bonus(self):
        goals=self.goals();goals['G0']['yellow']=3;goals['G0']['red']=1
        regions=dict.fromkeys(REGIONS,'neutral')
        result=match_score(goals,regions,2,1,'end',12,0)
        self.assertEqual((result['red'],result['blue']),(63,8))
        self.assertTrue(result['complete'])
        auto=match_score(goals,regions,2,1,'autonomous')
        self.assertEqual((auto['red'],auto['blue']),(5,0))
        self.assertTrue(auto['complete'])

    def test_unknown_midfield_and_bonus_stays_incomplete(self):
        result=match_score(self.goals(),dict.fromkeys(REGIONS,'neutral'))
        self.assertFalse(result['complete'])
        with self.assertRaises(ValueError):match_score(self.goals(),dict.fromkeys(REGIONS,'neutral'),3,0)
        with self.assertRaises(ValueError):match_score(self.goals(),dict.fromkeys(REGIONS,'neutral'),0,0,bonus_red=5)
        with self.assertRaises(ValueError):match_score(self.goals(),dict.fromkeys(REGIONS,'neutral'),0,0,bonus_red=12,bonus_blue=12)


if __name__=='__main__':unittest.main()
