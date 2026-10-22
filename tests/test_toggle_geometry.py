import unittest
import numpy as np
from toggle_geometry import exterior_face,inward_color,FrontFaceReader


class ToggleGeometryTests(unittest.TestCase):
    def test_three_face_correspondence_and_mirrored_installation(self):
        self.assertEqual([inward_color(c,'red') for c in ('red','blue','yellow')],['yellow','red','blue'])
        self.assertEqual(inward_color('blue','blue'),'yellow')
        self.assertEqual(inward_color('yellow','blue'),'red')
        self.assertEqual(inward_color('unknown','red'),'unknown')

    def test_front_band_wins_over_larger_upper_face(self):
        im=np.zeros((40,200,3),np.uint8)
        im[:25]=(0,255,255);im[25:35]=(0,0,255)
        self.assertEqual(exterior_face(im,[0,0,1,1])['color'],'red')
        im[:25]=(0,0,255);im[25:35]=(255,150,0)
        self.assertEqual(exterior_face(im,[0,0,1,1])['color'],'blue')

    def test_occlusion_and_transition_do_not_retain_false_color(self):
        reader=FrontFaceReader([0,0,1,1],'red')
        im=np.full((40,200,3),(0,255,255),np.uint8)
        self.assertEqual(reader.read(im)['color'],'unknown')
        reader.read(im)
        self.assertEqual(reader.read(im)['color'],'blue')
        im[:]=0;im[:,20:50]=(0,0,255)
        self.assertEqual(reader.read(im)['color'],'unknown')

if __name__=='__main__':unittest.main()
