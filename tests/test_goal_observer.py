import unittest
from goal_observer import observe_goal_stacks


class GoalObserverTests(unittest.TestCase):
    def row(self):
        return dict(seconds=1,robots=[],toggles={s:dict(color='red') for s in ('left','right','top','bottom')},
                    goal_rois={'G1':[.4,.5,.5,.6]},field_objects=[dict(class_id=6,xyxy=[.4,.5,.5,.6])])

    def test_ground_and_floating_objects_are_not_drawn_or_counted(self):
        row=self.row()
        row['field_objects'] += [dict(class_id=9,xyxy=[.44,.61,.47,.65]),
                                 dict(class_id=7,xyxy=[.44,.2,.47,.24])]
        result=observe_goal_stacks(row)
        self.assertEqual(result['objects'],[])
        self.assertEqual(result['goals']['G1']['counts'],dict(red=0,blue=0,yellow=0))
        self.assertEqual(result['excluded_ground_or_unconnected'],2)

    def test_upward_cup_chain_can_join_goal_without_counting_cup(self):
        row=self.row()
        row['field_objects'] += [dict(class_id=10,xyxy=[.43,.43,.47,.5]),
                                 dict(class_id=9,xyxy=[.44,.39,.465,.435])]
        result=observe_goal_stacks(row)
        self.assertEqual(len(result['objects']),2)
        self.assertEqual(result['goals']['G1']['counts']['yellow'],1)
        self.assertEqual(result['score']['goals']['G1']['red'],10)
        self.assertEqual(result['scoring_validity'],'unconfirmed')

    def test_robot_occlusion_yields_unknown_not_zero(self):
        row=self.row();row['robots']=[dict(xyxy=[.39,.48,.52,.65])]
        result=observe_goal_stacks(row)
        self.assertEqual(result['goals']['G1']['status'],'occluded')
        self.assertIsNone(result['score']['goals']['G1']['red'])

    def test_visible_color_does_not_become_confirmed_ownership(self):
        result=observe_goal_stacks(self.row())
        self.assertEqual(result['mode'],'provisional')
        self.assertEqual(result['scoring_validity'],'unconfirmed')
        self.assertFalse(result['score']['complete'])

    def test_same_half_seen_in_two_crop_windows_counts_once(self):
        row=self.row()
        row['field_objects'] += [dict(class_id=9,xyxy=[.44,.46,.46,.5],confidence=.95),
                                 dict(class_id=9,xyxy=[.441,.462,.459,.48],confidence=.7)]
        result=observe_goal_stacks(row)
        self.assertEqual(result['goals']['G1']['counts']['yellow'],1)


if __name__=='__main__':unittest.main()
