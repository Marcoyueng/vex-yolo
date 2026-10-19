import unittest
from desktop import encode_boxes, decode_boxes


class AnnotationTests(unittest.TestCase):
    def test_round_trip_at_image_edges(self):
        boxes = [(0, 0, 0, 1280, 720), (2, 100, 200, 350, 410)]
        restored = decode_boxes(encode_boxes(boxes, 1280, 720), 1280, 720)
        for expected, actual in zip(boxes, restored):
            for a, b in zip(expected, actual):
                self.assertAlmostEqual(a, b, places=4)

    def test_negative_image(self):
        self.assertEqual(encode_boxes([], 1280, 720), '')
        self.assertEqual(decode_boxes('', 1280, 720), [])

    def test_invalid_class_and_bounds(self):
        for row in ('3 .5 .5 .2 .2', '0 nan .5 .2 .2', '1 .9 .9 .5 .5', '1.5 .5 .5 .2 .2'):
            with self.subTest(row=row), self.assertRaises(ValueError):
                decode_boxes(row, 1280, 720)
