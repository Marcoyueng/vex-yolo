import unittest
from opening_alliance import OpeningAllianceTracker


class AllianceTests(unittest.TestCase):
    def setUp(self):
        self.anchors={'left':{'anchor':(.2,.5),'alliance':'blue','team':'1744X'},'bottom':{'anchor':(.5,.8),'alliance':'blue'},
                      'top':{'anchor':(.5,.2),'alliance':'red'},'right':{'anchor':(.8,.5),'alliance':'red'}}
        self.tracker=OpeningAllianceTracker(self.anchors,30)
        self.objects=[{'track_id':i,'xyxy':[x-.02,y-.02,x+.02,y+.02]} for i,(x,y) in enumerate([(.2,.5),(.5,.8),(.5,.2),(.8,.5)])]

    def test_seeds_and_carries_alliance_when_robot_moves(self):
        self.assertEqual([o['alliance'] for o in self.tracker.update(self.objects,0)],['blue','blue','red','red'])
        for frame in range(1,8):
            self.objects[0]['xyxy']=[n+.08 if i in (0,2) else n for i,n in enumerate(self.objects[0]['xyxy'])]
            result=self.tracker.update(self.objects,frame)
        self.assertEqual(result[0]['alliance'],'blue')
        self.assertEqual(result[0]['team'],'1744X')
        self.assertGreater(self.objects[0]['xyxy'][0],.7)

    def test_short_id_change_can_reconnect(self):
        self.tracker.update(self.objects,0)
        self.objects[0]['track_id']=99
        self.assertEqual(self.tracker.update(self.objects,1)[0]['alliance'],'blue')

    def test_long_gap_is_unknown(self):
        self.tracker.update(self.objects,0)
        self.assertIsNone(self.tracker.update(self.objects,100)[0]['alliance'])

    def test_late_initialization_does_not_use_current_position(self):
        self.assertTrue(all(o['alliance'] is None for o in self.tracker.update(self.objects,100)))
