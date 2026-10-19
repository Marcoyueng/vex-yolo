"""Shared versioned training classes. Object counts are not class counts."""
import json
import hashlib
from pathlib import Path

VERSION = 'override-2.1-v3-team-identity'
CLASSES = ['robot', 'team_plate', 'toggle_red', 'toggle_blue', 'toggle_yellow',
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
            raise ValueError('类别版本不匹配；v3 用 robot 和 team_plate，旧版红蓝车标签需明确迁移，不能混用。')
    elif create:
        if any(folder.glob('*.txt')):
            raise ValueError('该标签文件夹缺少类别版本记录，请选择新的空文件夹，避免误读旧标签。')
        folder.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        raise ValueError('缺少 classes.json：请从新版标签文件夹复制到训练数据根目录。')


def migrate_v2(source, target):
    """Explicit copy-only migration. Old alliance is a hint, never a team ID."""
    source,target=Path(source),Path(target)
    old=json.loads((source/'classes.json').read_text(encoding='utf-8'))
    if old != {'version':'override-2.1-v2','classes':['robot_red','robot_blue',*CLASSES[2:]]}:
        raise ValueError('仅支持本项目有 classes.json 的 v2 标签；未知旧格式不能自动转换。')
    if source.resolve()==target.resolve() or any(target.glob('*.txt')):
        raise ValueError('请选择新的空标签文件夹，原标签会保留。')
    converted=[]
    for label in source.glob('*.txt'):
        rows=[];objects=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            parts=line.split()
            if len(parts)!=5 or not 0<=int(parts[0])<len(CLASSES):raise ValueError(f'无效标签：{label.name}')
            cls=int(parts[0])
            rows.append(' '.join(['0' if cls==1 else parts[0],*parts[1:]])+'\n')
            objects.append({'observed_alliance':'red' if cls==0 else 'blue'} if cls in (0,1) else {})
        converted.append((label.name,''.join(rows),objects))
    ensure_schema(target,create=True)
    for name,text,objects in converted:
        label=target/name
        label.write_bytes(text.encode('utf-8'))
        label.with_suffix('.identity.json').write_text(json.dumps({'labels_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),
                                                                  'objects':objects},ensure_ascii=False,indent=2),encoding='utf-8')
    return len(converted)
