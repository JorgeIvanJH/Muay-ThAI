import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


MDE_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


MDE_INPUT = ROOT_DIR / "media" / "videos" / "Rodtang-taetat-2.mp4"
MDE_OUTPUT = ROOT_DIR / "output"
MDE_INPUT_SIZE = 200
MDE_MAX_DEPTH = 1000.0

MDE_WEIGHTS = ROOT_DIR / "models" / "mde" / "weights" / "depth_anything_v2_metric_vkitti_vitl.pth"
MDE_ENCODER_NAME = MDE_WEIGHTS.stem.split("_")[-1]
MDE_ENCODER_CONFIG = MDE_MODEL_CONFIGS.get(MDE_ENCODER_NAME, MDE_MODEL_CONFIGS["vitl"])

# -------- WITH HUGGINGFACE PIPELINE --------
MDE_HF_REPO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
MDE_HF_WEIGHTS = ROOT_DIR / "models" / "mde" / "weights" / MDE_HF_REPO_ID.split("/")[-1]
MDE_DEVICE = "cuda"