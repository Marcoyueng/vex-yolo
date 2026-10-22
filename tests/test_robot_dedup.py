import copy
import unittest
from robot_dedup import deduplicate_robot_boxes


def box(coords, score):
    return {'xyxy': coords, 'confidence': score}


class RobotDedupTests(unittest.TestCase):
    def test_nested_body_duplicate_keeps_confident_full_robot_without_mutation(self):
        objects=[box([8,80,92,200],.6),box([0,0,100,200],.8)]
        before=copy.deepcopy(objects)
        kept,audit=deduplicate_robot_boxes(objects)
        self.assertEqual(kept,[objects[1]])
        self.assertEqual((audit[0]['kept_index'],audit[0]['removed_index']),(1,0))
        self.assertEqual(objects,before)

    def test_two_overlapping_robots_with_separate_bases_are_not_merged(self):
        objects=[box([0,0,100,200],.9),box([10,30,90,160],.8)]
        self.assertEqual(deduplicate_robot_boxes(objects),(objects,[]))

    def test_side_by_side_robots_keep_separate_boxes(self):
        objects=[box([0,0,100,200],.9),box([35,20,135,200],.8)]
        self.assertEqual(deduplicate_robot_boxes(objects),(objects,[]))

    def test_equal_confidence_ambiguous_pair_is_preserved(self):
        objects=[box([0,0,100,200],.8),box([8,80,92,200],.8)]
        self.assertEqual(deduplicate_robot_boxes(objects),(objects,[]))

    def test_box_count_is_not_forced_to_four(self):
        objects=[box([i*100,0,i*100+50,50],.9) for i in range(6)]
        self.assertEqual(deduplicate_robot_boxes(objects),(objects,[]))


if __name__=='__main__':unittest.main()
