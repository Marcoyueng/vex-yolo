import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from types import SimpleNamespace
import cv2
import numpy as np
from schema import CLASSES, migrate_v2, ensure_schema
from event_project import (add_media, import_matches, connect, CSV_FIELDS, bind_team,
                           resolve_team, export_identities, seconds, remove_binding, project_lock)
from event_worker import run, detect_match


class EventTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.project=self.root/'project'
        self.video=self.root/'test.avi'
        writer=cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
        self.assertTrue(writer.isOpened())
        for i in range(40):writer.write(np.full((120,160,3),i*5,dtype=np.uint8))
        writer.release()
        self.source_bytes=self.video.read_bytes()
        add_media(self.project,[self.video,self.video])
        self.csv=self.root/'matches.csv'
        self.write_csv([['Q1',self.video.name,0,2,'100A','200B','300C','400D'],
                        ['Q2',self.video.name,2,4,'300C','400D','100A','200B']])
        import_matches(self.project,self.csv)

    def write_csv(self,rows):
        with self.csv.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(CSV_FIELDS);writer.writerows(rows)

    def test_catalog_does_not_copy_video(self):
        with connect(self.project) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM media').fetchone()[0],1)
        self.assertFalse(list(self.project.rglob('*.avi')))
        self.assertEqual(self.source_bytes,self.video.read_bytes())

    def test_csv_import_is_atomic(self):
        self.write_csv([['Q3',self.video.name,0,1,'1A','2B','3C','4D'],
                        ['Q4','missing.mp4',0,1,'1A','2B','3C','4D']])
        with self.assertRaises(ValueError):import_matches(self.project,self.csv)
        with connect(self.project) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM matches').fetchone()[0],2)

    def test_resume_after_partial_failure_and_idempotency(self):
        original=cv2.imencode
        calls=[0]
        def fail_once(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:raise ValueError('simulated interrupt')
            return original(*args,**kwargs)
        with patch('event_worker.cv2.imencode',side_effect=fail_once):
            with self.assertRaises(ValueError):run(self.project,every=.5,limit=4,width=320,budget_gb=.01)
        with connect(self.project) as db:
            q1=db.execute("SELECT * FROM jobs WHERE match_id='Q1'").fetchone()
            self.assertEqual(q1['next_index'],1)
        run(self.project,every=.5,limit=4,width=320,budget_gb=.01)
        run(self.project,every=.5,limit=4,width=320,budget_gb=.01)
        with connect(self.project) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM frames').fetchone()[0],8)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs WHERE status='done'").fetchone()[0],2)
        self.assertEqual(len(list((self.project/'frames').glob('*.jpg'))),8)
        self.assertEqual(self.source_bytes,self.video.read_bytes())

    def test_disk_budget_can_be_increased_and_resumed(self):
        with self.assertRaises(ValueError):run(self.project,every=1,limit=2,width=320,budget_gb=1e-9)
        self.assertEqual(list((self.project/'frames').glob('*.jpg')),[])
        run(self.project,every=1,limit=2,width=320,budget_gb=.01)
        self.assertEqual(len(list((self.project/'frames').glob('*.jpg'))),4)

    def test_team_changes_alliance_and_unknown_track_is_not_guessed(self):
        bind_team(self.project,'Q1','run1',7,0,9,'100A')
        bind_team(self.project,'Q2','run1',7,0,9,'100A')
        with connect(self.project) as db:
            self.assertEqual(resolve_team(db,'Q1','run1',7,1)['alliance'],'red')
            self.assertEqual(resolve_team(db,'Q2','run1',7,1)['alliance'],'blue')
            self.assertIsNone(resolve_team(db,'Q1','run1',7,10))
            self.assertIsNone(resolve_team(db,'Q1','different-run',7,1))
        with self.assertRaises(ValueError):bind_team(self.project,'Q1','run1',8,0,9,'100A')
        with self.assertRaises(ValueError):bind_team(self.project,'Q1','run1',7,5,12,'200B')
        with self.assertRaises(ValueError):bind_team(self.project,'Q1','run1',8,12,15,'999Z')
        remove_binding(self.project,'Q1','run1',7,0)
        with connect(self.project) as db:
            self.assertIsNone(resolve_team(db,'Q1','run1',7,1))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM binding_history').fetchone()[0],1)

    def test_streaming_identity_export(self):
        source=self.root/'detections.jsonl'
        source.write_text(json.dumps({'match_id':'Q1','run_id':'run1','frame':0,
            'detections':[{'class':'robot','track_id':7},{'class':'robot','track_id':8}]})+'\n')
        bind_team(self.project,'Q1','run1',7,0,9,'100A')
        target=self.root/'with-teams.jsonl'
        export_identities(self.project,source,target,'Q1','run1')
        detections=json.loads(target.read_text())['detections']
        self.assertEqual(detections[0]['team'],'100A')
        self.assertIsNone(detections[1]['team'])
        with self.assertRaises(FileExistsError):export_identities(self.project,source,target,'Q1','run1')

    def test_source_changes_block_resume(self):
        with self.video.open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):run(self.project,every=1,limit=2,width=320,budget_gb=.01)

    def test_time_formats(self):
        self.assertEqual(seconds('01:02:03.5'),3723.5)
        for value in ('nan','-1','00:60:00','inf'):
            with self.assertRaises(ValueError):seconds(value)

    def test_exclusive_worker_lock_is_released(self):
        with project_lock(self.project):
            with self.assertRaises(ValueError):
                with project_lock(self.project):pass
        with project_lock(self.project):pass

    def test_v2_migration_preserves_originals(self):
        old=self.root/'old';new=self.root/'new';old.mkdir()
        marker={'version':'override-2.1-v2','classes':['robot_red','robot_blue',*CLASSES[2:]]}
        (old/'classes.json').write_text(json.dumps(marker))
        original='0 .5 .5 .2 .2\n1 .5 .5 .3 .3\n6 .5 .5 .1 .1\n'
        (old/'sample.txt').write_text(original)
        self.assertEqual(migrate_v2(old,new),1)
        self.assertEqual((old/'sample.txt').read_text(),original)
        self.assertEqual([line.split()[0] for line in (new/'sample.txt').read_text().splitlines()],['0','0','6'])
        ensure_schema(new)
        metadata=json.loads((new/'sample.identity.json').read_text())
        self.assertEqual(metadata['objects'][1]['observed_alliance'],'blue')
        self.assertNotIn('team',metadata['objects'][1])

    def test_single_match_detection_uses_interval_and_streams_records(self):
        weights=self.root/'mock.pt';weights.touch()
        boxes=MagicMock()
        boxes.xyxy.cpu.return_value.tolist.return_value=[[1,2,30,40]]
        boxes.cls.cpu.return_value.tolist.return_value=[0]
        boxes.conf.cpu.return_value.tolist.return_value=[.9]
        boxes.id.cpu.return_value.tolist.return_value=[7]
        model=MagicMock(task='detect',names=dict(enumerate(CLASSES)))
        model.track.return_value=[SimpleNamespace(boxes=boxes)]
        with patch.dict('sys.modules',{'ultralytics':SimpleNamespace(YOLO=lambda _:model)}):
            detect_match(self.project,'Q2',str(weights))
        report=next((self.project/'detections').glob('*/detections.jsonl'))
        records=[json.loads(line) for line in report.read_text().splitlines()]
        self.assertEqual(len(records),20)
        self.assertEqual(records[0]['source_frame'],20)
        self.assertEqual(records[0]['match_id'],'Q2')
        self.assertIsNone(records[0]['detections'][0]['team'])
        self.assertTrue((report.parent/'complete.json').is_file())
        self.assertTrue(model.track.call_args.kwargs['persist'])
        self.assertFalse(model.track.call_args.kwargs['save'])
        self.assertEqual(self.video.read_bytes(),self.source_bytes)
