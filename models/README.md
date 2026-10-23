# Trained model snapshot

See manifest.json for SHA-256, scopes and limitations. The app and four model checkpoints are included.
goal-target-r4.pt is the newest three-class conditional Goal-Pin candidate; use GoalTargetDetector or predict_goal_target.py.
It is not compatible with the general 11-class detector. Zero detections under occlusion do not establish an empty Goal.
Known-example fit is not independent-match accuracy. No automatic application promotion occurred.
robot-r10.pt remains experimental. Team IDs are not learned classes. Toggle color alone does not prove valid ownership.
Checkpoints retain upstream Ultralytics/PyTorch notices and applicable license requirements.
No new license grant over upstream models or recordings is implied by this snapshot.
