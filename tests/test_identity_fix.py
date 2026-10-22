import unittest
from render_identity_fix import resolve


class IdentityFixTests(unittest.TestCase):
    def entry(self,team,margin=1):
        return {'team':team,'frames':100,'method':'offline_appearance_and_cooccurrence','alternative_cost_margin':margin}

    def box(self,track,x2=.4):
        return {'track_id':track,'xyxy':[.1,.1,x2,.4],'confidence':.9}

    def test_scoreboard_alliance_is_independent_of_screen_position(self):
        rows=resolve([self.box(i) for i in range(4)],{str(i):self.entry(team) for i,team in enumerate(['1744X','471A','88S','2899A'])})
        self.assertEqual([o['alliance'] for o in rows],['red','red','blue','blue'])

    def test_duplicate_boxes_cannot_duplicate_a_confirmed_team(self):
        rows=resolve([self.box(1),self.box(2,.5)],{'1':self.entry('1744X'),'2':self.entry('1744X')})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['track_id'],2)

    def test_ambiguous_identity_is_not_published_as_a_team(self):
        rows=resolve([self.box(1)],{'1':self.entry('88S',.1)})
        self.assertIsNone(rows[0]['team'])
        self.assertIsNone(rows[0]['alliance'])
