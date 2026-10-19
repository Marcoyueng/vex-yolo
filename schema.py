"""Shared versioned training classes. Object counts are not class counts."""
import json
from pathlib import Path

VERSION = 'override-2.1-v2'
CLASSES = ['robot_red', 'robot_blue', 'toggle_red', 'toggle_blue', 'toggle_yellow',
           'toggle_unknown', 'goal', 'pin_half_red', 'pin_half_blue', 'pin_half_yellow', 'cup']
COLORS = ['#f87171', '#38bdf8', '#ef4444', '#3b82f6', '#facc15', '#a1a1aa',
          '#4ade80', '#fb7185', '#67e8f9', '#fde047', '#c084fc']
REGIONS = ('red1', 'red2', 'blue1', 'blue2')
SCHEMA = {'version': VERSION, 'classes': CLASSES}


def ensure_schema(folder, create=False):
    folder = Path(folder)
    marker = folder / 'classes.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != SCHEMA:
            raise ValueError('类别版本不匹配，请重新标注；旧版 robot/pin/cup 标签不可直接复用。')
    elif create:
        if any(folder.glob('*.txt')):
            raise ValueError('该标签文件夹缺少类别版本记录，请选择新的空文件夹，避免误读旧标签。')
        folder.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        raise ValueError('缺少 classes.json：请从新版标签文件夹复制到训练数据根目录。')
