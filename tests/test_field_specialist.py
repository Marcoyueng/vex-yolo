import unittest
import cv2
import numpy as np
from field_specialist import Specialist,suppress,field_mask


class FieldSpecialistTests(unittest.TestCase):
    def test_nested_cup_pin_and_goals_are_not_merged_between_classes(self):
        obj=dict(xyxy=[.2,.2,.4,.4],confidence=.9,class_id=7)
        rows=suppress([obj,dict(obj,confidence=.6),dict(obj,class_id=10)])
        self.assertEqual(len(rows),2)
        self.assertEqual({row['class_id'] for row in rows},{7,10})

    def test_field_mask_excludes_scoreboard_and_spectators(self):
        mask=field_mask((100,200,3),[[20,20],[180,20],[180,80],[20,80]])
        self.assertEqual(mask[50,100],255)
        self.assertEqual(mask[50,190],0)
        self.assertEqual(mask[10,80],0)

    def test_short_flow_estimate_is_not_observation_or_score_evidence(self):
        model=Specialist.__new__(Specialist)
        rng=np.random.default_rng(2)
        image=np.zeros((1152,2048,3),np.uint8)
        image[450:550,900:1000]=rng.integers(0,256,(100,100,3),dtype=np.uint8)
        model.previous=cv2.cvtColor(cv2.resize(image,(1024,576)),cv2.COLOR_BGR2GRAY)
        model.objects=[dict(class_id=7,xyxy=[900/2048,450/1152,1000/2048,550/1152],
                            confidence=.9,scoring_validity='unconfirmed',sample_seconds=0)]
        model.sample_seconds=0
        moved=cv2.warpAffine(image,np.float32([[1,0,4],[0,1,2]]),(2048,1152))
        result=model.propagate(moved,.1)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['observation'],'optical_flow_estimate')
        self.assertEqual(result[0]['scoring_validity'],'unconfirmed')
        self.assertAlmostEqual(result[0]['xyxy'][0],904/2048,delta=.002)
        self.assertEqual(model.propagate(moved,1.2),[])


if __name__=='__main__':unittest.main()
