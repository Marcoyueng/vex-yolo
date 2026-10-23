import unittest
from types import SimpleNamespace
from goal_target_detector import GoalTargetDetector,target_window


class TensorList:
    def __init__(self,values):self.values=values
    def cpu(self):return self
    def tolist(self):return self.values


class StubModel:
    def predict(self,crops,**kwargs):
        self.crops=crops;self.options=kwargs
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=TensorList([[10,20,30,40]]*3),
            cls=TensorList([0,1,2]),conf=TensorList([.7,.8,.9]))) for _ in crops]


class GoalTargetTests(unittest.TestCase):
    def test_window_includes_tall_stack_above_floor(self):
        x,y,x2,y2=target_window([.45,.6,.5,.65],(1000,2000,3))
        self.assertLess(y,200)
        self.assertLess(x,900)
        self.assertGreater(x2,1000)
        self.assertGreaterEqual(y2,650)

    def test_edge_crop_is_clipped_to_real_image(self):
        x,y,x2,y2=target_window([.01,.1,.06,.16],(1000,2000,3))
        self.assertEqual((x,y),(0,0))
        self.assertLessEqual(x2,2000);self.assertLessEqual(y2,1000)

    def test_invalid_goal_is_not_used_for_crop(self):
        with self.assertRaises(ValueError):target_window([.4,.4,.3,.5],(100,200,3))
        with self.assertRaises(ValueError):target_window([-2,.4,-1.9,.5],(100,200,3))

    def test_model_output_maps_from_crop_to_source_and_stays_provisional(self):
        import numpy as np
        image=np.zeros((1000,2000,3),dtype=np.uint8)
        model=GoalTargetDetector.__new__(GoalTargetDetector);model.model=StubModel()
        roi=[.45,.6,.5,.65];x,y,_,_=target_window(roi,image.shape)
        result=model.detect(image,{'G1':roi},seconds=7.25)
        self.assertEqual([o['class_id'] for o in result['G1']],[7,8,9])
        self.assertEqual(result['G1'][0]['xyxy'],[(x+10)/2000,(y+20)/1000,(x+30)/2000,(y+40)/1000])
        self.assertTrue(all(o['goal_id']=='G1' and o['scoring_validity']=='unconfirmed' for o in result['G1']))
        self.assertTrue(all(o['observation']=='goal_conditioned_model_detection' for o in result['G1']))
        self.assertEqual(model.model.options['conf'],.25)

    def test_empty_goals_does_not_invent_boxes(self):
        import numpy as np
        model=GoalTargetDetector.__new__(GoalTargetDetector);model.model=StubModel()
        self.assertEqual(model.detect(np.zeros((100,200,3),dtype=np.uint8),{}),{})
        self.assertFalse(hasattr(model.model,'crops'))

if __name__=='__main__':unittest.main()
