import unittest
from scoring import toggle_owner, yellow_score
from schema import REGIONS


class ScoringTests(unittest.TestCase):
    def inputs(self, owner='red'):
        toggles = {r: dict(color=owner, settled=True, robot_contact=False) for r in REGIONS}
        return toggles, {f'G{i}': 0 for i in range(9)}

    def test_toggle_contact_and_unsettled_are_neutral(self):
        self.assertEqual(toggle_owner('red', True, True), 'neutral')
        self.assertEqual(toggle_owner('blue', False, False), 'neutral')
        self.assertEqual(toggle_owner('red', None, False), 'unknown')

    def test_sc5_color_changes_ownership_not_multiplier(self):
        t, h = self.inputs()
        h['G1'] = 3
        a = yellow_score(t, h, 0, 0)
        self.assertEqual((a['red'], a['blue']), (30, 0))
        t['red1']['color'] = 'blue'
        b = yellow_score(t, h, 0, 0)
        self.assertEqual((b['red'], b['blue']), (0, 30))

    def test_region_controls_two_goals(self):
        t, h = self.inputs('yellow')
        t['blue2']['color'] = 'red'
        h.update(G7=2, G8=1, G6=4)
        self.assertEqual(yellow_score(t, h, 0, 0)['red'], 30)

    def test_midfield_majority_tie_and_autonomous(self):
        t, h = self.inputs()
        h['G0'] = 4
        self.assertEqual(yellow_score(t, h, 1, 2)['blue'], 40)
        self.assertEqual(yellow_score(t, h, 2, 2)['red'], 0)
        self.assertEqual(yellow_score(t, h, 2, 0, 'autonomous')['red'], 0)

    def test_unknown_is_not_zero(self):
        t, h = self.inputs()
        h['G1'] = None
        result = yellow_score(t, h, 0, 0)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['goals']['G1']['points'])

    def test_unknown_contact_blocks_claim(self):
        t, h = self.inputs()
        h['G1'] = 3
        t['red1']['robot_contact'] = None
        self.assertFalse(yellow_score(t, h, 0, 0)['complete'])

    def test_invalid_counts(self):
        t, h = self.inputs()
        for value in (-1, 1.5, True):
            h['G1'] = value
            with self.assertRaises(ValueError):
                yellow_score(t, h, 0, 0)
        h['G1'] = 0
        with self.assertRaises(ValueError):
            yellow_score(t, h, 3, 0)
