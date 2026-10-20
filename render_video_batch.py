"""Full-frame-count local scoreboard exports for a reviewed multi-video job."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import cv2
from PIL import Image
from live_dashboard import RecordIndex
from goal_memory import GoalMemoryTimeline
from goal_observer import observe_goal_stacks
from goal_overlay import goal_view
from scoreboard_view import compose,paste_video,paint_time,SIZE
from match_state import TeamMemory

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);args=parser.parse_args()
    cv2.setNumThreads(2)
    videos=json.loads(Path(args.manifest).read_text(encoding='utf-8'))['videos']
    pointer=ROOT/'runs/latest_elims_results.json';state=dict(status='running',videos=[])
    def save():
        temporary=pointer.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(pointer)
    save()
    try:
        for video in videos:
            while True:
                try:association=json.loads((ROOT/'runs/latest_elims_association.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
                if association['status']=='failed':raise ValueError('Identity association failed')
                item=next((x for x in association['videos'] if x['id']==video['id'] and x['status']=='complete'),None)
                if item:break
                time.sleep(1)
            records=RecordIndex(item['records']);timeline=GoalMemoryTimeline(records,video['fps'],start_frame=video.get('match_start_frame'))
            target=ROOT/'runs/elims-batch'/video['id'];output=target/f'miko-yolo-{video["id"]}.mp4'
            entry=dict(video,id=video['id'],source=video['source'],records=item['records'],output=str(output),status='rendering',
                       scoring_validity='unconfirmed',placed_halves_persistence=True)
            state['videos'].append(entry);save();capture=cv2.VideoCapture(video['source']);memory=TeamMemory(records.roster)
            command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{SIZE[0]}x{SIZE[1]}',
                     '-r',str(video['fps']),'-i','pipe:0','-i',video['source'],'-map','0:v:0','-map','1:a?',
                     '-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
                     '-c:a','copy','-movflags','+faststart',str(output)]
            panel=None;panel_key=None;last_panel=-1e9
            with (target/'ffmpeg.log').open('w') as errors:
                process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=errors,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                try:
                    for index in range(video['frames']):
                        ok,frame=capture.read()
                        if not ok:raise ValueError('Incomplete source decode')
                        row=records.get(index);observation=timeline.view(index,observe_goal_stacks(row))
                        overlay,observation=goal_view(frame,row,video['toggle_rois'],observation=observation)
                        states=memory.update(row['seconds'],row['robots'])
                        key=json.dumps([observation['score'],[(g,v['counts'],v['status']) for g,v in observation['goals'].items()],
                                        [(site,v.get('color')) for site,v in row['toggles'].items()],[(s['team'],s['status']) for s in states]],sort_keys=True)
                        if panel is None or key!=panel_key or row['seconds']-last_panel>=.2:
                            panel=compose(None,observation,records.roster,states,row['toggles'],timeline.history)
                            panel_key=key;last_panel=row['seconds']
                        picture=panel.copy();paste_video(picture,Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)));paint_time(picture,row['seconds'])
                        process.stdin.write(picture.tobytes())
                        if index in (0,round(video['fps']*30),round(video['fps']*100),video['frames']-1):picture.save(target/f'preview-{index:06d}.png')
                        if index%600==0:print(f'BATCH_EXPORT {video["id"]} {index}/{video["frames"]}',flush=True)
                    process.stdin.close()
                    if process.wait()!=0:raise RuntimeError('FFmpeg export failed')
                except BaseException:
                    if process.poll() is None:process.kill();process.wait()
                    raise
                finally:capture.release();records.close()
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(output)],text=True))
            stream=next(s for s in probe['streams'] if s['codec_type']=='video')
            if int(stream['nb_read_frames'])!=video['frames']:raise ValueError('Encoded frame count mismatch')
            entry.update(status='complete',encoded_frames=int(stream['nb_read_frames']),audio_streams=sum(s['codec_type']=='audio' for s in probe['streams']),
                         known_team_boxes=item['known_team_boxes'],total_robot_boxes=item['total_robot_boxes'])
            (target/'summary.json').write_text(json.dumps(entry,indent=2));save();print('BATCH_EXPORT_COMPLETE '+video['id'],flush=True)
        state.update(status='complete');save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
