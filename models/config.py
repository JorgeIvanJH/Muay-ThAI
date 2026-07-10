import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
YOLOMDE_INPUT = ROOT_DIR / "media" / "videos" / "Rodtang-taetat-2.mp4"
YOLOMDE_OUTPUT = ROOT_DIR / "output"
YOLOMDE_INPUT_SIZE = 196
YOLOMDE_FRAME_STRIDE = 1
YOLOMDE_SAVE_JOINTS = True
YOLOMDE_SAVE_DEPTHS = True
