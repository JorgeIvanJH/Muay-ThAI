import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

MDE_HF_REPO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
MDE_HF_WEIGHTS = ROOT_DIR / "models" / "mde" / "weights" / MDE_HF_REPO_ID.split("/")[-1]
MDE_DEVICE = "cuda"

MDE_INPUT_SIZE = 224
MDE_MAX_DEPTH = 5
MDE_INPUT = ROOT_DIR / "media" / "videos" / "Rodtang-taetat-2.mp4"
MDE_OUTPUT = ROOT_DIR / "output"

MDE_HALF = True
MDE_FRAME_STRIDE = 1

MDE_DEPTH_FORMAT = "npy" # Options: "npz" (compressed) or "npy" (faster, larger)