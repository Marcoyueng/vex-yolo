"""Override dataset preparation, training, and streamed detection."""
import argparse
import json
import math
from pathlib import Path

from schema import CLASSES, ensure_schema


def validate_dataset(root):
    """Require explicit annotations, valid boxes, and isolated video sessions."""
    ensure_schema(root)
    groups = []
    counts = [0] * len(CLASSES)
    for split in ('train', 'val'):
        images = sorted((root / 'images' / split).glob('*'))
        images = [p for p in images if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if not images:
            raise ValueError(f'No images in {split}')
        stems = [p.stem for p in images]
        if len(stems) != len(set(stems)):
            raise ValueError(f'Duplicate image stems in {split}')
        groups.append({p.stem.split('__')[0] for p in images})
        for image in images:
            label = root / 'labels' / split / (image.stem + '.txt')
            if not label.is_file():
                raise ValueError(f'Missing annotation: {label}; empty files mean reviewed negatives')
            for line in label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'Invalid YOLO row in {label}')
                cls = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if not (0 <= cls < len(CLASSES) and all(math.isfinite(v) for v in (x,y,w,h))
                        and 0 < w <= 1 and 0 < h <= 1 and w/2 <= x <= 1-w/2+1e-6
                        and h/2 <= y <= 1-h/2+1e-6):
                    raise ValueError(f'Invalid class or bounding box in {label}')
                counts[cls] += 1
    if groups[0] & groups[1]:
        raise ValueError('Train/val share video session prefixes; split by video, not adjacent frames')
    if not any(counts):
        raise ValueError('At least one positive annotation is required')
    missing = [name for name, n in zip(CLASSES, counts) if n == 0]
    if missing:
        print('Warning: untrained classes without annotations: ' + ', '.join(missing))
    config = root / 'dataset.yaml'
    config.write_text('path: ' + json.dumps(root.resolve().as_posix()) + '\ntrain: images/train\nval: images/val\nnames:\n'
                      + ''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)), encoding='utf-8')
    return config


def extract(args):
    import cv2
    if not args.session.replace('-', '').replace('_', '').isalnum() or '__' in args.session:
        raise ValueError('Use a unique session name containing letters, digits, - or single _')
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Cannot open video or determine FPS')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame_id = 0
    saved = 0
    interval = max(1, round(fps * args.every))
    manifest = []
    try:
        while saved < args.limit:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % interval == 0:
                name = f'{args.session}__{frame_id:08d}.jpg'
                target = out / name
                if target.exists():
                    raise ValueError(f'Refusing to overwrite {target}')
                if not cv2.imwrite(str(target), frame):
                    raise ValueError(f'Cannot save {target}')
                manifest.append({'image': name, 'seconds': frame_id/fps, 'source': str(args.video)})
                saved += 1
            frame_id += 1
    finally:
        cap.release()
        with (out / f'{args.session}.jsonl').open('a', encoding='utf-8') as f:
            for row in manifest:
                f.write(json.dumps(row) + '\n')
    if not saved:
        raise ValueError('No readable frames')
    print(f'Saved {saved} frames. These are unlabelled; annotate before training.')


def download(args):
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import download_range_func
    if args.end <= args.start:
        raise ValueError('--end must be greater than --start')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with YoutubeDL({'format': 'bestvideo[ext=mp4][height<=720]/best[height<=720]',
                    'download_ranges': download_range_func(None, [(args.start, args.end)]),
                    'noplaylist': True, 'socket_timeout': 20, 'retries': 1,
                    'outtmpl': str(out / f'%(id)s-{args.start:g}-{args.end:g}.%(ext)s'), 'writeinfojson': True}) as ydl:
        ydl.download([args.url])


def train(args):
    from ultralytics import YOLO
    config = validate_dataset(Path(args.data).resolve())
    YOLO(args.base).train(data=str(config), epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, device=args.device, workers=0,
                          project='runs/train', name='override', seed=42)


def predict(args):
    from ultralytics import YOLO
    weights = Path(args.weights)
    if not weights.is_file():
        raise ValueError('Override weights missing. Train the custom dataset first or supply your own .pt file.')
    model = YOLO(str(weights))
    if model.task != 'detect' or list(model.names.values()) != CLASSES:
        raise ValueError(f'Expected detection weights with classes {CLASSES}; generic COCO weights are unsuitable')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = out / 'detections.jsonl'
    if report.exists():
        raise ValueError('Output already contains detections.jsonl; choose a new output directory')
    source = int(args.source) if args.source.isdecimal() else args.source
    with report.open('x', encoding='utf-8') as f:
        runner = model.track if args.track else model.predict
        tracking = {'tracker': 'bytetrack.yaml'} if args.track else {}
        for index, result in enumerate(runner(source=source, stream=True, save=True,
                 show=args.show, project=str(out), name='annotated', exist_ok=True,
                 conf=args.conf, imgsz=args.imgsz, device=args.device, **tracking)):
            boxes = result.boxes
            detections = []
            ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            for xyxy, cls, conf, track_id in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), ids):
                detections.append({'class_id': int(cls), 'class': CLASSES[int(cls)],
                                   'confidence': conf, 'xyxy': xyxy,
                                   'track_id': int(track_id) if track_id is not None else None})
            f.write(json.dumps({'frame': index, 'source': str(result.path),
                                'height': result.orig_shape[0], 'width': result.orig_shape[1],
                                'detections': detections}, ensure_ascii=False) + '\n')
            f.flush()
    print(f'Results: {out.resolve()}')


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Must be a positive finite number')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download', help='Download a public YouTube video and source metadata')
    p.add_argument('url'); p.add_argument('--output', default='data/videos'); p.set_defaults(func=download)
    p.add_argument('--start', type=float, default=3600); p.add_argument('--end', type=positive, default=3720)
    p = sub.add_parser('extract', help='Sample a local video for annotation')
    p.add_argument('video'); p.add_argument('--session', required=True)
    p.add_argument('--every', type=positive, default=3); p.add_argument('--limit', type=int, default=300)
    p.add_argument('--output', default='data/unlabelled'); p.set_defaults(func=extract)
    p = sub.add_parser('check', help='Validate annotations and write dataset.yaml')
    p.add_argument('--data', default='data/override-v3')
    p.set_defaults(func=lambda a: print(validate_dataset(Path(a.data).resolve())))
    p = sub.add_parser('train')
    p.add_argument('--data', default='data/override-v3'); p.add_argument('--base', default='yolo11n.pt')
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=960); p.add_argument('--device', default='cpu'); p.set_defaults(func=train)
    p = sub.add_parser('predict')
    p.add_argument('--weights', default='runs/train/override/weights/best.pt'); p.add_argument('--source', required=True)
    p.add_argument('--output', default='runs/predict/override'); p.add_argument('--conf', type=float, default=.35)
    p.add_argument('--imgsz', type=int, default=960); p.add_argument('--device', default='cpu')
    p.add_argument('--show', action='store_true'); p.set_defaults(func=predict)
    p.add_argument('--track', action='store_true', help='Track objects across video frames with ByteTrack IDs')
    args = parser.parse_args()
    if hasattr(args, 'limit') and args.limit <= 0:
        parser.error('--limit must be positive')
    if hasattr(args, 'conf') and not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
