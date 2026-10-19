"""Bounded-memory, checkpointed event sampling; one source video at a time."""
import argparse
import math
from pathlib import Path
import shutil
import cv2
from event_project import connect, enqueue, project_task


@project_task
def run(project, every=5.0, limit=100, width=1280, budget_gb=10.0):
    if not math.isfinite(budget_gb) or budget_gb <= 0:
        raise ValueError('磁盘预算必须大于 0 GB')
    project = Path(project).resolve()
    enqueue(project, every, limit, width)
    output = project/'frames'
    output.mkdir(parents=True, exist_ok=True)
    with connect(project) as db:
        jobs = db.execute('SELECT * FROM jobs WHERE status != ? ORDER BY match_id', ('done',)).fetchall()
        used = db.execute('SELECT COALESCE(SUM(bytes),0) FROM frames').fetchone()[0]
        for job in jobs:
            cap = None
            try:
                media = db.execute('SELECT * FROM media WHERE id=?', (job['media_id'],)).fetchone()
                stat = Path(media['path']).stat()
                if (stat.st_size, stat.st_mtime_ns) != (media['size'],media['mtime']):
                    raise ValueError('源视频已变更，不能继续旧任务。')
                cap = cv2.VideoCapture(media['path'])
                fps, frame_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
                    raise ValueError('无法读取视频时长或帧率。')
                if job['end'] > frame_count/fps + 1/fps:
                    raise ValueError('CSV 结束时间超过视频时长，请在新项目中修正对阵表。')
                samples = min(job['max_frames'], math.ceil((job['end']-job['start'])/job['every']))
                db.execute("UPDATE jobs SET status='running',error='' WHERE id=?", (job['id'],)); db.commit()
                for index in range(job['next_index'], samples):
                    timestamp = job['start']+index*job['every']
                    if not cap.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000):
                        raise ValueError('视频不支持按时间定位，请先转换视频封装。')
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'无法解码 {timestamp:.2f} 秒的画面。')
                    h, w = frame.shape[:2]
                    if w > job['width']:
                        frame = cv2.resize(frame, (job['width'], max(1,round(h*job['width']/w))))
                    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise ValueError('图片编码失败')
                    size = encoded.nbytes
                    if used+size > budget_gb*1024**3 or shutil.disk_usage(output).free < size+16*1024**2:
                        raise ValueError('达到项目抽帧磁盘预算或磁盘空间不足。增大预算/更换项目盘后继续。')
                    # Stable job id and sample index make retries safe after a killed process.
                    target = output/f'{job["media_id"]}__{job["id"]}_{index:08d}.jpg'
                    temp = target.with_suffix('.tmp')
                    temp.write_bytes(encoded.tobytes()); temp.replace(target)
                    db.execute('INSERT OR REPLACE INTO frames VALUES(?,?,?,?,?)', (job['id'],index,str(target),timestamp,size))
                    db.execute('UPDATE jobs SET next_index=? WHERE id=?', (index+1,job['id'])); db.commit()
                    used += size
                    print(f'{job["match_id"]}: {index+1}/{samples} · {timestamp:.2f}s · {used/1024**2:.1f} MB', flush=True)
                db.execute("UPDATE jobs SET status='done' WHERE id=?", (job['id'],)); db.commit()
            except Exception as error:
                db.execute("UPDATE jobs SET status='error',error=? WHERE id=?", (str(error),job['id'])); db.commit()
                print(f'{job["match_id"]}: {error}', flush=True)
                # Fail fast on disk exhaustion; other source errors do not discard later matches.
                if isinstance(error, OSError) or '预算' in str(error):
                    raise
            finally:
                if cap is not None:
                    cap.release()
        failed = db.execute("SELECT COUNT(*) FROM jobs WHERE status!='done'").fetchone()[0]
        if failed:
            raise ValueError(f'{failed} 个任务未完成；重新开始会从已保存的抽帧位置继续。')
    print('全部抽帧任务完成。', flush=True)


@project_task
def detect_match(project, match_id, weights, device='cpu', confidence=.35):
    """Read only the selected match interval; JSONL only, no video duplication."""
    import json
    from datetime import datetime
    from ultralytics import YOLO
    from schema import CLASSES
    if not Path(weights).is_file() or not 0 <= confidence <= 1:
        raise ValueError('请选择新版模型，并填写 0–1 的置信度')
    model = YOLO(weights)
    if model.task != 'detect' or list(model.names.values()) != CLASSES:
        raise ValueError('模型类别不匹配：赛事身份识别需要 v3 robot/team_plate 模型。')
    with connect(project) as db:
        match = db.execute('SELECT m.*,v.path,v.size,v.mtime FROM matches m JOIN media v ON m.media_id=v.id WHERE match_id=?', (match_id,)).fetchone()
        if not match:
            raise ValueError('先在赛事列表选择场次。')
        stat = Path(match['path']).stat()
        if (stat.st_size,stat.st_mtime_ns) != (match['size'],match['mtime']):
            raise ValueError('源视频已变更')
        cap = cv2.VideoCapture(match['path'])
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError('无法读取视频')
            if match['end'] > cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps + 1/fps:
                raise ValueError('场次结束时间超过视频时长')
            first = round(match['start']*fps)
            last = math.ceil(match['end']*fps)
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
                raise ValueError('视频定位失败')
            run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            out = Path(project)/'detections'/run_id
            out.mkdir(parents=True)
            with (out/'detections.jsonl').open('x',encoding='utf-8') as f:
                for source_frame in range(first,last):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('视频解码提前结束；当前结果不完整，请重新运行本场。')
                    result = model.track(frame,persist=True,tracker='bytetrack.yaml',device=device,
                                         conf=confidence,verbose=False,save=False,imgsz=960)[0]
                    boxes = result.boxes
                    ids = boxes.id.cpu().tolist() if boxes.id is not None else [None]*len(boxes)
                    detections = []
                    for box, cls, conf, track in zip(boxes.xyxy.cpu().tolist(),boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist(),ids):
                        detections.append({'xyxy':box,'class':CLASSES[int(cls)],'class_id':int(cls),'confidence':conf,
                                           'track_id':int(track) if track is not None else None,
                                           'team':None,'alliance':None,'identity_source':'unconfirmed'})
                    row = {'match_id':match_id,'run_id':run_id,'source':match['path'],'frame':source_frame-first,
                           'source_frame':source_frame,'source_seconds':source_frame/fps,'width':frame.shape[1],
                           'height':frame.shape[0],'detections':detections}
                    f.write(json.dumps(row)+'\n'); f.flush()
                    if (source_frame-first)%100 == 0:
                        print(f'{match_id}: {source_frame-first}/{last-first} 帧',flush=True)
            (out/'complete.json').write_text(json.dumps({'match_id':match_id,'run_id':run_id,'frames':last-first}),encoding='utf-8')
            print(f'单场检测完成：{out}',flush=True)
        finally:
            cap.release()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('project'); p.add_argument('--every', type=float, default=5)
    p.add_argument('--limit', type=int, default=100); p.add_argument('--width', type=int, default=1280)
    p.add_argument('--budget-gb', type=float, default=10)
    p.add_argument('--mode', choices=('extract','detect'), default='extract')
    p.add_argument('--match'); p.add_argument('--weights'); p.add_argument('--device',default='cpu')
    p.add_argument('--conf',type=float,default=.35)
    a = p.parse_args()
    try:
        if a.mode == 'detect':
            detect_match(a.project,a.match,a.weights,a.device,a.conf)
        else:
            run(a.project, a.every, a.limit, a.width, a.budget_gb)
    except Exception as error:
        p.exit(1, str(error)+'\n')
