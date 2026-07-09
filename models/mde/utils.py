import torch
from transformers import pipeline
import config as mdecfg
from huggingface_hub import snapshot_download
import numpy as np
import matplotlib.pyplot as plt
import cv2

def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def safe_model_load(model_path):
    try:
        model = pipeline("depth-estimation", model=model_path, device=select_device())
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Attempting to download the model snapshot...")
        snapshot_download(mdecfg.MDE_HF_REPO_ID, local_dir=mdecfg.MDE_HF_WEIGHTS)
        model = pipeline("depth-estimation", model=mdecfg.MDE_HF_WEIGHTS, device=select_device())
    return model

def colorize_depth(depth):
    inverse_depth = 1.0 / np.clip(depth, 1e-4, None)
    max_invdepth = min(float(np.nanmax(inverse_depth)), 1 / 0.1)
    min_invdepth = max(float(np.nanmin(inverse_depth)), 1 / 250)
    scale = max(max_invdepth - min_invdepth, 1e-8)
    normalized = np.clip((inverse_depth - min_invdepth) / scale, 0.0, 1.0)

    color_depth = plt.get_cmap("turbo")(normalized)[..., :3]
    return (color_depth * 255).astype(np.uint8)


def colorize_depth_bgr(depth):
    return cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR)

def depth_summary(depth):

    return {
        "depth_shape": list(depth.shape),
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "depth_mean_m": float(depth.mean()),
    }