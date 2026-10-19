"""Persistent event catalog. Stores file references; never copies source videos."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from contextlib import contextmanager
from functools import wraps
import os

VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.mts'}
CSV_FIELDS = ['match_id', 'video', 'start', 'end', 'red1', 'red2', 'blue1', 'blue2']


@contextmanager
def project_lock(project):
    """OS-released exclusive worker lock; survives crashes without stale leases."""
    folder=Path(project);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.worker.lock').open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('该赛事项目已有运行中的抽帧或检测任务，请先停止另一任务。') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def project_task(function):
    @wraps(function)
    def wrapped(project,*args,**kwargs):
        with project_lock(project):
            return function(project,*args,**kwargs)
    return wrapped


@contextmanager
def connect(project):
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(project/'event.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript('''
      CREATE TABLE IF NOT EXISTS media(id TEXT PRIMARY KEY, path TEXT UNIQUE, size INTEGER, mtime INTEGER);
      CREATE TABLE IF NOT EXISTS matches(match_id TEXT PRIMARY KEY, media_id TEXT, start REAL, end REAL,
        red1 TEXT, red2 TEXT, blue1 TEXT, blue2 TEXT);
      CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, match_id TEXT, media_id TEXT, start REAL, end REAL,
        every REAL, max_frames INTEGER, width INTEGER, next_index INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending', error TEXT DEFAULT '');
      CREATE TABLE IF NOT EXISTS frames(job_id TEXT, sample_index INTEGER, path TEXT, seconds REAL,
        bytes INTEGER, PRIMARY KEY(job_id, sample_index));
      CREATE TABLE IF NOT EXISTS bindings(match_id TEXT, run_id TEXT, track_id INTEGER, start_frame INTEGER,
        end_frame INTEGER, team TEXT, PRIMARY KEY(match_id,run_id,track_id,start_frame));
    ''')
    try:
        with db:
            yield db
    finally:
        db.close()


def add_media(project, paths):
    with connect(project) as db:
        for path in paths:
            path = Path(path).resolve()
            if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
                raise ValueError(f'不是支持的视频文件：{path}')
            stat = path.stat()
            key = hashlib.sha256(str(path).casefold().encode()).hexdigest()[:16]
            old = db.execute('SELECT * FROM media WHERE path=?', (str(path),)).fetchone()
            if old and (old['size'], old['mtime']) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f'视频已被替换或修改，请在新赛事项目中重新导入：{path.name}')
            db.execute('INSERT OR IGNORE INTO media VALUES(?,?,?,?)', (key, str(path), stat.st_size, stat.st_mtime_ns))


def seconds(value):
    parts = str(value).strip().split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError('时间格式应为秒数或 HH:MM:SS')
    nums = [float(p) for p in parts]
    if any(not math.isfinite(n) or n < 0 for n in nums) or (len(nums) > 1 and any(n >= 60 for n in nums[1:])):
        raise ValueError('时间数值无效')
    return sum(n * 60**i for i, n in enumerate(reversed(nums)))


def team_id(value):
    value = value.strip().upper()
    if not re.fullmatch(r'[A-Z0-9-]{1,24}', value):
        raise ValueError(f'队号无效：{value}')
    return value


def import_matches(project, csv_path):
    """Import atomically. Editing an existing match needs a new ID/project."""
    csv_path = Path(csv_path)
    with connect(project) as db, csv_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(CSV_FIELDS) <= set(reader.fieldnames):
            raise ValueError('CSV 必须包含：' + ','.join(CSV_FIELDS))
        media = [dict(row) for row in db.execute('SELECT * FROM media')]
        seen = set()
        total = 0
        for row in reader:
            match = row['match_id'].strip()
            if not match or len(match) > 80 or match in seen:
                raise ValueError(f'场次为空、过长或重复：{match}')
            seen.add(match)
            raw = Path(row['video'].strip())
            resolved = raw if raw.is_absolute() else csv_path.parent/raw
            candidates = [v for v in media if Path(v['path']) == resolved.resolve()]
            if not candidates and len(raw.parts) == 1:
                candidates = [v for v in media if Path(v['path']).name.casefold() == raw.name.casefold()]
            if len(candidates) != 1:
                raise ValueError(f'{match}：视频未导入或文件名不唯一，请填完整路径。')
            start, end = seconds(row['start']), seconds(row['end'])
            if end <= start:
                raise ValueError(f'{match}：结束时间必须晚于开始时间。')
            teams = [team_id(row[k]) for k in CSV_FIELDS[4:]]
            if len(set(teams)) != 4:
                raise ValueError(f'{match}：四个队伍不能重复。')
            values = (match, candidates[0]['id'], start, end, *teams)
            old = db.execute('SELECT * FROM matches WHERE match_id=?', (match,)).fetchone()
            if old and tuple(old) != values:
                raise ValueError(f'{match} 已存在且内容不同，请使用新场次编号或新项目。')
            db.execute('INSERT OR IGNORE INTO matches VALUES(?,?,?,?,?,?,?,?)', values)
            total += 1
        return total


def enqueue(project, every, limit, width):
    if not math.isfinite(every) or every <= 0 or type(limit) is not int or limit <= 0 or not 320 <= width <= 3840:
        raise ValueError('间隔需大于 0，每场张数需为正整数，宽度需为 320–3840。')
    with connect(project) as db:
        matches = db.execute('SELECT * FROM matches ORDER BY match_id').fetchall()
        if not matches:
            raise ValueError('先导入包含视频时间段的对阵表。')
        for match in matches:
            identity = [match['match_id'], match['media_id'], match['start'], match['end'], every, limit, width]
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
            db.execute('INSERT OR IGNORE INTO jobs(id,match_id,media_id,start,end,every,max_frames,width) VALUES(?,?,?,?,?,?,?,?)',
                       (key, *identity))


def bind_team(project, match_id, run_id, track_id, first, last, team):
    if not run_id.strip() or track_id < 0 or first < 0 or last < first:
        raise ValueError('填写检测运行名称、有效 track_id 和起止帧。')
    team = team_id(team)
    with connect(project) as db:
        match = db.execute('SELECT * FROM matches WHERE match_id=?', (match_id,)).fetchone()
        if not match or team not in [match[k] for k in CSV_FIELDS[4:]]:
            raise ValueError('队号必须属于该场对阵表中的四个队伍。')
        conflicts = db.execute('SELECT * FROM bindings WHERE match_id=? AND run_id=? AND track_id=? AND NOT(end_frame<? OR start_frame>?)',
                               (match_id,run_id,track_id,first,last)).fetchall()
        if conflicts:
            raise ValueError('这个跟踪 ID 的帧区间已有绑定，请选择不重叠区间。')
        duplicates=db.execute('SELECT 1 FROM bindings WHERE match_id=? AND run_id=? AND team=? AND NOT(end_frame<? OR start_frame>?)',
                              (match_id,run_id,team,first,last)).fetchone()
        if duplicates:
            raise ValueError('同一队伍在该帧区间已有机器人绑定，请先撤销错误绑定。')
        db.execute('INSERT INTO bindings VALUES(?,?,?,?,?,?)', (match_id,run_id,track_id,first,last,team))


def remove_binding(project, match_id, run_id, track_id, first):
    with connect(project) as db:
        old=db.execute('SELECT * FROM bindings WHERE match_id=? AND run_id=? AND track_id=? AND start_frame=?',
                       (match_id,run_id,track_id,first)).fetchone()
        if not old:
            raise ValueError('未找到此起始帧的绑定。')
        db.execute('CREATE TABLE IF NOT EXISTS binding_history (removed_at TEXT DEFAULT CURRENT_TIMESTAMP, record TEXT)')
        db.execute('INSERT INTO binding_history(record) VALUES(?)',(json.dumps(dict(old)),))
        db.execute('DELETE FROM bindings WHERE match_id=? AND run_id=? AND track_id=? AND start_frame=?',
                   (match_id,run_id,track_id,first))


def resolve_team(db, match_id, run_id, track_id, frame):
    row = db.execute('SELECT team FROM bindings WHERE match_id=? AND run_id=? AND track_id=? AND start_frame<=? AND end_frame>=?',
                     (match_id,run_id,track_id,frame,frame)).fetchone()
    if not row:
        return None
    match = db.execute('SELECT * FROM matches WHERE match_id=?', (match_id,)).fetchone()
    return {'team': row['team'], 'alliance': 'red' if row['team'] in (match['red1'],match['red2']) else 'blue',
            'identity_source': 'manual_binding'}


def export_identities(project, source, output, match_id, run_id):
    """Stream JSONL; never infer identity from an unbound track ID."""
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError('导出路径不能覆盖原检测记录。')
    with connect(project) as db, Path(source).open(encoding='utf-8') as src, Path(output).open('x', encoding='utf-8') as dst:
        if not db.execute('SELECT 1 FROM matches WHERE match_id=?', (match_id,)).fetchone():
            raise ValueError('场次不存在')
        for line in src:
            record = json.loads(line)
            if record.get('match_id') != match_id or record.get('run_id') != run_id:
                raise ValueError('检测记录与所选场次/运行不匹配，请使用赛事页生成的单场记录。')
            record.update(match_id=match_id, run_id=run_id)
            for detection in record.get('detections', []):
                if detection.get('class') not in ('robot', 'robot_red', 'robot_blue'):
                    continue
                identity = resolve_team(db, match_id, run_id, detection.get('track_id'), record['frame'])
                detection.update(identity or {'team': None, 'alliance': None, 'identity_source': 'unconfirmed'})
            dst.write(json.dumps(record, ensure_ascii=False)+'\n')
