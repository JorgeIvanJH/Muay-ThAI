import os
from pathlib import Path

# ------- YOLO CONFIGURATION -------
YOLO_DEFAULT_MODEL_PATH = Path("models") / "yolo" / "yolo26l-pose.pt"
YOLO_DEFAULT_SOURCE = Path("media") / "videos" / "Rodtang-taetat-2.mp4"
YOLO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]