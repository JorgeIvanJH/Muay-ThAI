import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
# Dataset generation keeps the high-quality large model so existing and future
# training CSVs use the same pose distribution.
YOLO_WEIGHTS = ROOT_DIR / "models" / "yolo" / "weights" / "yolo26l-pose.pt"

# YOLO26s was the largest tested model that sustained the laptop webcam's
# measured 29.6 FPS while running both classifiers, analytics and both video
# writers without dropping a frame. Inference can override it with
# ``--pose-model`` when another camera resolution or GPU is used.
YOLO_REALTIME_WEIGHTS = (
    ROOT_DIR / "models" / "yolo" / "weights" / "yolo26s-pose.pt"
)
YOLO_INPUT = ROOT_DIR / "media" / "videos" / "Rodtang-taetat-2.mp4"
YOLO_OUTPUT = ROOT_DIR / "output"
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
YOLO_SMOOTHING_ALPHA = 1
