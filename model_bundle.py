"""Locate the model shipped with the portable source release."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def default_robot_weights():
    packaged = ROOT / 'models' / 'robot-r10.pt'
    return packaged if packaged.is_file() else ROOT / 'runs/train/override/weights/best.pt'
