"""Compare a specialist on reviewed, withheld full-video frames.

Review media and per-Goal counts are local inputs, never embedded test fixtures.
"""
import argparse
import json
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent


def evaluate(weights,review_path,records_path,output):
    import torch
    torch.set_num_threads(8);cv2.setNumThreads(2)
    from field_specialist import Specialist
    from live_dashboard import RecordIndex
    from goal_observer import observe_goal_stacks
    from goal_overlay import goal_view
    from scoreboard_view import compose
    from match_state import TeamMemory
    from PIL import Image
    review=json.loads(Path(review_path).read_text(encoding='utf-8'))
    samples=[s for s in review['samples'] if s['split']=='val']
    specialist=Specialist(weights)
    records=RecordIndex(records_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=[]
    try:
        for source in dict.fromkeys(s['image'] for s in samples):
            selected=[s for s in samples if s['image']==source]
            frame=selected[0]['frame'];row=records.get(frame)
            image=cv2.imread(source)
            detected=specialist.detect(image,row['seconds'],row['robots'])
            goal_rois={g:[b[0]/2048,b[1]/1152,b[2]/2048,b[3]/1152] for g,b in specialist.goals.items()}
            candidate=dict(row,field_objects=detected,goal_rois=goal_rois)
            observation=observe_goal_stacks(candidate)
            comparisons=[]
            for sample in selected:
                goal=sample['goal_id']
                expected={color:sum(o['class_id']==cls for o in sample['objects']) for color,cls in [('red',7),('blue',8),('yellow',9)]}
                found=observation['goals'][goal]['counts']
                comparisons.append(dict(goal=goal,expected=expected,detected=found,exact=found==expected,
                                        estimated_points=observation['score']['goals'][goal]))
            overlay,observation=goal_view(image,candidate,{})
            picture=compose(Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)),observation,records.roster,
                            TeamMemory(records.roster).update(row['seconds'],row['robots']),row['toggles'])
            picture.save(output/f'preview-{frame:06d}.png')
            (output/f'detections-{frame:06d}.json').write_text(json.dumps(candidate,indent=2))
            report.append(dict(frame=frame,goals=comparisons))
    finally:records.close()
    result=dict(weights=str(weights),scope='same-video withheld full frames; assistant-reviewed stack counts',frames=report)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('weights','review','records','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    evaluate(args.weights,args.review,args.records,args.output)
