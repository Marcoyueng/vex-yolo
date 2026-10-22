import unittest
import cv2
import numpy as np
from reviewed_pin_overlay import ReviewedPinOverlay


class ReviewedPinTests(unittest.TestCase):
    def test_geometry_requires_current_pixels_and_cannot_appear_before_placement(self):
        rng=np.random.default_rng(7);im=rng.integers(0,256,(60,60,3),dtype=np.uint8)
        tracker=ReviewedPinOverlay.__new__(ReviewedPinOverlay)
        tracker.config=dict(start_frame=10,context_box=[20,20,40,40],goal='G4',
            parts=[dict(class_id=9,box=[25,25,30,30])])
        tracker.patch=im[20:40,20:40].copy()
        self.assertEqual(tracker.observe(im,9),[])
        self.assertEqual(len(tracker.observe(im,10)),1)
        occluded=im.copy();occluded[15:45,15:45]=0
        self.assertEqual(tracker.observe(occluded,11),[])


if __name__=='__main__':unittest.main()
