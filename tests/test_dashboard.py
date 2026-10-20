import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from live_dashboard import Dashboard, RecordIndex


class DashboardTests(unittest.TestCase):
    def test_prefilled_candidates_cannot_be_saved_without_review(self):
        panel=SimpleNamespace(current={'frame':0},pending_candidate_review=True,
                              valid=SimpleNamespace(get=lambda:False),session={'reviews':[]})
        with patch('live_dashboard.messagebox.showerror') as error:
            Dashboard.save_review(panel)
        error.assert_called_once()
        self.assertEqual(panel.session['reviews'],[])

    def test_index_rejects_missing_identity_contract_and_discontinuous_frames(self):
        for rows in ([dict(frame=0,seconds=0,detections=[])],
                     [dict(frame=1,seconds=0,robots=[])]):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                path=Path(folder)/'input.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):RecordIndex(path)

    def test_index_random_access_and_cross_match_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'input.jsonl'
            rows=[dict(frame=i,seconds=i/30,robots=[dict(team='1744X',alliance='red')]) for i in range(3)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            index=RecordIndex(path)
            try:
                self.assertEqual(index.get(2)['frame'],2)
                self.assertEqual(index.get(0)['frame'],0)
                self.assertEqual(index.roster,{'1744X':'red'})
            finally:index.close()
            rows[2]['robots'][0]['alliance']='blue'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):RecordIndex(path)

    def test_binding_time_scope_newest_wins_and_raw_rows_are_preserved(self):
        panel=SimpleNamespace(roster={'A':'red','B':'blue'},session=dict(bindings=[
            dict(frame=20,track_id=2,team='A'),dict(frame=10,track_id=1,team='A')]))
        original=dict(frame=15,robots=[dict(track_id=1,team=None),dict(track_id=2,team=None)])
        result=Dashboard.corrected(panel,original)
        self.assertEqual(result['robots'][0]['team'],'A')
        self.assertIsNone(original['robots'][0]['team'])
        result=Dashboard.corrected(panel,dict(original,frame=25))
        self.assertIsNone(result['robots'][0]['team'])
        self.assertEqual(result['robots'][1]['team'],'A')
        result=Dashboard.corrected(panel,dict(original,frame=5))
        self.assertTrue(all(o['team'] is None for o in result['robots']))


if __name__=='__main__':unittest.main()
