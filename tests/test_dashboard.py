import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from live_dashboard import Dashboard, RecordIndex


class DashboardTests(unittest.TestCase):
    def test_overlap_review_can_unset_identity_only_inside_its_frame_range(self):
        panel=SimpleNamespace(roster={'A':'red','B':'blue'},session=dict(bindings=[dict(frame=10,end_frame=20,track_id=1,team=None)]))
        row=dict(frame=15,robots=[dict(track_id=1,team='A',alliance='red')])
        self.assertIsNone(Dashboard.corrected(panel,row)['robots'][0]['team'])
        self.assertEqual(Dashboard.corrected(panel,dict(row,frame=21))['robots'][0]['team'],'A')

    def test_same_robot_duplicate_box_does_not_return_as_unknown_after_review(self):
        panel=SimpleNamespace(roster={'A':'red','B':'blue'},session=dict(bindings=[dict(frame=0,track_id=2,team='A')]))
        row=dict(frame=1,robots=[dict(track_id=1,team='A',alliance='red',xyxy=[.1,.1,.3,.4]),
                                 dict(track_id=2,team=None,xyxy=[.1,.1,.3,.4])])
        result=Dashboard.corrected(panel,row)
        self.assertEqual(len(result['robots']),1);self.assertEqual(result['robots'][0]['track_id'],2)

    def test_speed_change_reanchors_without_a_video_time_jump(self):
        panel=SimpleNamespace(frame=210,playing=True)
        with patch('live_dashboard.time.monotonic',return_value=12.):Dashboard.set_playback_rate(panel,.5)
        self.assertEqual((panel.play_start_frame,panel.play_started,panel.playback_rate),(210,12.,.5))
        shown=[]
        panel.timer=None;panel.fps=30;panel.records=SimpleNamespace(offsets=range(900));panel.display_fps=15
        panel.advance_to=lambda i:shown.append(i);panel.after=lambda *args:None;panel.tick=lambda:None
        with patch('live_dashboard.time.monotonic',return_value=14.):Dashboard.tick(panel)
        self.assertEqual(shown,[240])
        with patch('live_dashboard.time.monotonic',return_value=14.):Dashboard.set_playback_rate(panel,2.)
        with patch('live_dashboard.time.monotonic',return_value=16.):Dashboard.tick(panel)
        self.assertEqual(shown[-1],330)

    def test_roster_and_manual_binding_prevent_duplicate_or_wrong_alliance(self):
        panel=SimpleNamespace(roster={'R1':'red','R2':'red','B1':'blue','B2':'blue'},session=dict(bindings=[
            dict(frame=10,end_frame=20,track_id=7,team='B1')]))
        robots=[dict(track_id=7,team='R1',alliance='red',confidence=.8),dict(track_id=8,team='B1',alliance='blue',confidence=.9),
                dict(track_id=9,team='R2',alliance='blue',confidence=.9)]
        result=Dashboard.corrected(panel,dict(frame=15,robots=robots))
        self.assertEqual((result['robots'][0]['team'],result['robots'][0]['alliance']),('B1','blue'))
        self.assertIsNone(result['robots'][1]['team'])
        self.assertEqual(result['robots'][2]['alliance'],'red')
        after=Dashboard.corrected(panel,dict(frame=21,robots=robots))
        self.assertEqual(after['robots'][0]['team'],'R1')

    def test_playback_follows_wall_clock_when_paint_is_late(self):
        shown=[];scheduled=[]
        panel=SimpleNamespace(timer=None,playing=True,frame=100,play_start_frame=100,play_started=10.,fps=30,
                              records=SimpleNamespace(offsets=range(500)),display_fps=15,
                              advance_to=lambda i:shown.append(i),after=lambda ms,cb:scheduled.append(ms),tick=lambda:None)
        with patch('live_dashboard.time.monotonic',return_value=12.):Dashboard.tick(panel)
        self.assertEqual(shown,[160])
        self.assertLessEqual(scheduled[0],67)

    def test_skipped_display_frames_still_update_identity_memory(self):
        consumed=[];shown=[];grabbed=[]
        panel=SimpleNamespace(frame=2,capture=SimpleNamespace(grab=lambda:grabbed.append(True) or True),
                              records=SimpleNamespace(get=lambda i:dict(seconds=i,robots=[dict(track_id=i)])),
                              corrected=lambda r:r,memory=SimpleNamespace(update=lambda t,objects:consumed.append(t)),
                              show_frame=lambda i,read_sequential:shown.append((i,read_sequential)))
        Dashboard.advance_to(panel,6)
        self.assertEqual(consumed,[3,4,5]);self.assertEqual(len(grabbed),3)
        self.assertEqual(shown,[(6,True)])

    def test_explicit_roster_keeps_unreadable_robot_team_unset(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'input.jsonl'
            path.write_text(json.dumps(dict(frame=0,seconds=0,robots=[dict(team=None,alliance=None)],roster={'A':'red','B':'blue'}))+'\n')
            records=RecordIndex(path)
            self.assertEqual(records.roster,{'A':'red','B':'blue'})
            self.assertIsNone(records.get(0)['robots'][0]['team']);records.close()

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
