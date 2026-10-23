"""Run the experimental target-Goal Pin model on one local image."""
import argparse
import json
from pathlib import Path
import cv2
from goal_target_detector import GoalTargetDetector

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--goals',required=True,help='JSON mapping Goal IDs to normalized [x1,y1,x2,y2]')
    p.add_argument('--weights',default=str(Path(__file__).resolve().parent/'models/goal-target-r4.pt'))
    p.add_argument('--output',required=True)
    args=p.parse_args()
    image=cv2.imread(args.image)
    if image is None:p.error('Cannot read image')
    goals=json.loads(Path(args.goals).read_text(encoding='utf-8'))
    objects=GoalTargetDetector(args.weights).detect(image,goals,confidence=.25)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(goals=objects,scoring_validity='unconfirmed',
        note='Zero detections do not prove an empty Goal under occlusion.'),indent=2)+'\n',encoding='utf-8')
    print(str(target))

if __name__=='__main__':main()
