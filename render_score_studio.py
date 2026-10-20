"""Export the Goal-only scoreboard view with synchronized provisional scores.

No inference is rerun and no broadcast score OCR is used. Existing draft object
records are filtered through Goal connectivity; unknown rule states stay unknown.
"""
import json,subprocess
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image
from live_dashboard import RecordIndex
from match_state import TeamMemory
from goal_observer import observe_goal_stacks
from goal_overlay import goal_view
from scoreboard_view import compose,SIZE

ROOT=Path(__file__).resolve().parent


def main():
    cv2.setNumThreads(2)
    state=json.loads((ROOT/'runs/active_goal_pilot.json').read_text(encoding='utf-8'))
    records=RecordIndex(state['preview_jsonl'])
    target=ROOT/'runs'/('score-studio-'+datetime.now().strftime('%Y%m%d-%H%M%S'));target.mkdir(parents=True)
    accepted=Path(json.loads((ROOT/'runs/active_toggle_pilot.json').read_text())['directory'])
    toggle_rois=json.loads((accepted/'toggle-rois.json').read_text())
    capture=cv2.VideoCapture(state['source']);fps=capture.get(5);total=round(capture.get(7))
    if not capture.isOpened() or total!=len(records.offsets):raise ValueError('Source / records mismatch')
    history=[]
    for i in range(0,total,max(1,round(fps))):
        row=records.get(i);score=observe_goal_stacks(row)['score']
        history.append(dict(seconds=row['seconds'],red=score['red'],blue=score['blue']))
    output=target/'miko-yolo-score-studio.mp4';result_path=target/'goal-only-detections.jsonl'
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{SIZE[0]}x{SIZE[1]}',
             '-r',str(fps),'-i','pipe:0','-i',state['source'],'-map','0:v:0','-map','1:a?',
             '-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
             '-c:a','copy','-movflags','+faststart',str(output)]
    memory=TeamMemory(records.roster)
    try:
        with (target/'ffmpeg.log').open('w') as errors,result_path.open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=errors,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                for index in range(total):
                    ok,frame=capture.read()
                    if not ok:raise ValueError('Video ended before detection records')
                    row=records.get(index);overlay,observation=goal_view(frame,row,toggle_rois)
                    states=memory.update(row['seconds'],row['robots'])
                    picture=compose(Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)),observation,records.roster,states,row['toggles'],history)
                    process.stdin.write(picture.tobytes())
                    filtered=dict(row,field_objects=[o for o in row['field_objects'] if o.get('class_id')==6]+observation['objects'],
                                  goal_estimate=observation,scoring_validity='unconfirmed')
                    log.write(json.dumps(filtered)+'\n')
                    if index%900==0 or index==total-1:picture.save(target/f'preview-{index:06d}.png')
                    if index%300==0:print(f'SCORE_VIDEO {index}/{total}',flush=True)
                process.stdin.close()
                if process.wait()!=0:raise RuntimeError('FFmpeg failed')
            except BaseException:
                if process.poll() is None:process.kill();process.wait()
                raise
    finally:capture.release();records.close()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(output)],text=True))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    if int(video['nb_read_frames'])!=total:raise ValueError('Encoded frame count mismatch')
    report=dict(source=state['source'],input_records=state['preview_jsonl'],video=str(output),records=str(result_path),frames=total,
                audio_streams=sum(s['codec_type']=='audio' for s in probe['streams']),
                scoring_validity='unconfirmed',score_method='visible Goal-connected candidates; no referee validity or broadcast score OCR',
                limitations='2-D connections do not prove Placed/visibility. Occluded Goals and midfield/bonus remain unknown.')
    (target/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pointer=ROOT/'runs/active_score_studio.json';temp=pointer.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2),encoding='utf-8');temp.replace(pointer)
    print('SCORE_VIDEO_COMPLETE '+str(output),flush=True)


if __name__=='__main__':main()
