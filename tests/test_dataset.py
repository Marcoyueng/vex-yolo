import tempfile
import unittest
from pathlib import Path
from vex_yolo import validate_dataset
from schema import ensure_schema


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        ensure_schema(self.root, create=True)
        for split, session in [('train', 'eventA'), ('val', 'eventB')]:
            for kind in ('images', 'labels'):
                (self.root / kind / split).mkdir(parents=True)
            (self.root / 'images' / split / f'{session}__00001.jpg').touch()
            (self.root / 'labels' / split / f'{session}__00001.txt').write_text(
                '0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n2 0.8 0.8 0.1 0.1\n')

    def test_valid(self):
        self.assertIn('robot', validate_dataset(self.root).read_text())

    def test_missing_annotations(self):
        (self.root / 'labels/train/eventA__00001.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing annotation'):
            validate_dataset(self.root)

    def test_nonfinite_box(self):
        (self.root / 'labels/train/eventA__00001.txt').write_text('0 nan 0.5 0.1 0.1')
        with self.assertRaisesRegex(ValueError, 'bounding box'):
            validate_dataset(self.root)

    def test_video_leakage(self):
        for kind, ext in [('images', 'jpg'), ('labels', 'txt')]:
            folder = self.root / kind / 'val'
            (folder / f'eventB__00001.{ext}').rename(folder / f'eventA__00002.{ext}')
        with self.assertRaisesRegex(ValueError, 'share video'):
            validate_dataset(self.root)


if __name__ == '__main__':
    unittest.main()
