import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from vex_yolo import predict
from schema import CLASSES, ensure_schema


class TrackingTests(unittest.TestCase):
    def test_tracking_exports_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            weights = root/'test.pt'; weights.touch()
            boxes = MagicMock()
            boxes.xyxy.cpu.return_value.tolist.return_value = [[1, 2, 30, 40]]
            boxes.cls.cpu.return_value.tolist.return_value = [0]
            boxes.conf.cpu.return_value.tolist.return_value = [.9]
            boxes.id.cpu.return_value.tolist.return_value = [7]
            model = MagicMock(task='detect', names=dict(enumerate(CLASSES)))
            model.track.return_value = [SimpleNamespace(boxes=boxes, path='test.mp4', orig_shape=(720, 1280))]
            args = SimpleNamespace(weights=str(weights), output=str(root/'out'), source='test.mp4',
                                   track=True, show=False, conf=.35, imgsz=960, device='cpu')
            with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda _: model)}):
                predict(args)
            row = json.loads((root/'out/detections.jsonl').read_text())
            self.assertEqual(row['detections'][0]['track_id'], 7)
            self.assertEqual(model.track.call_args.kwargs['tracker'], 'bytetrack.yaml')
            model.predict.assert_not_called()

    def test_legacy_label_folder_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            (Path(folder)/'old.txt').write_text('0 .5 .5 .2 .2')
            with self.assertRaises(ValueError):
                ensure_schema(folder, create=True)
