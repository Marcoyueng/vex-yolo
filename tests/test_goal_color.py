import unittest
import cv2
import numpy as np
from goal_color import corroborate_yellow
from goal_observer import observe_goal_stacks


class GoalColorTests(unittest.TestCase):
    def fixture(self):
        image=np.full((600,800,3),240,np.uint8)
        cv2.rectangle(image,(390,300),(410,325),(0,255,255),-1)
        cv2.rectangle(image,(392,325),(408,355),(55,55,55),-1)
        cv2.rectangle(image,(389,355),(411,380),(0,255,255),-1)
        cv2.rectangle(image,(392,380),(408,402),(55,55,55),-1)
        goals={'G1':[380,400,420,450]}
        objects=[dict(class_id=6,xyxy=[380/800,400/600,420/800,450/600],confidence=.9),
                 dict(class_id=9,xyxy=[390/800,300/600,410/800,325/600],confidence=.9)]
        return image,goals,objects

    def test_current_yellow_and_neutral_pixels_can_complete_a_proposal(self):
        image,goals,objects=self.fixture()
        extra=corroborate_yellow(image,goals,objects,1)
        row=dict(seconds=1,robots=[],toggles={},goal_rois={'G1':[380/800,400/600,420/800,450/600]},field_objects=objects+extra)
        self.assertEqual(observe_goal_stacks(row)['goals']['G1']['counts']['yellow'],2)
        self.assertTrue(all(o['scoring_validity']=='unconfirmed' for o in extra))

    def test_floor_yellow_or_missing_gray_connector_cannot_fill_a_stack(self):
        image,goals,objects=self.fixture()
        image[325:355]=240
        cv2.rectangle(image,(389,470),(411,495),(0,255,255),-1)
        self.assertEqual(corroborate_yellow(image,goals,objects,1),[])
        self.assertEqual(corroborate_yellow(image,goals,objects[1:],1),[])


if __name__=='__main__':unittest.main()
