import torch
from transformers import pipeline
from . import config as mdecfg
from huggingface_hub import snapshot_download
import numpy as np
import cv2


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_model_load(model_path, use_half=False):
    device = select_device()
    pipeline_kwargs = {"model": model_path, "device": device}
    if use_half and device == "cuda":
        pipeline_kwargs["dtype"] = torch.float16

    try:
        model = pipeline("depth-estimation", **pipeline_kwargs)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Attempting to download the model snapshot...")
        snapshot_download(mdecfg.MDE_HF_REPO_ID, local_dir=mdecfg.MDE_HF_WEIGHTS)
        pipeline_kwargs["model"] = mdecfg.MDE_HF_WEIGHTS
        model = pipeline("depth-estimation", **pipeline_kwargs)
    return model


def configure_pipeline_for_inference(model, input_size=None):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(model, "model"):
        model.model.eval()

    if input_size is not None and hasattr(model, "image_processor"):
        model.image_processor.size = {
            "height": int(input_size),
            "width": int(input_size),
        }


def save_depth_array(path, depth, save_format):
    if save_format == "npy":
        np.save(path, depth)
    elif save_format == "npz":
        np.savez_compressed(path, depth=depth)
    else:
        raise ValueError(f"Unsupported depth save format: {save_format}")


def normalized_inverse_depth_uint8(depth):
    inverse_depth = 1.0 / np.clip(depth, 1e-4, None)
    max_invdepth = min(float(np.nanmax(inverse_depth)), 1 / 0.1)
    min_invdepth = max(float(np.nanmin(inverse_depth)), 1 / 250)
    scale = max(max_invdepth - min_invdepth, 1e-8)
    normalized = np.clip((inverse_depth - min_invdepth) / scale, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def colorize_depth(depth):
    return cv2.cvtColor(colorize_depth_bgr(depth), cv2.COLOR_BGR2RGB)


def colorize_depth_bgr(depth):
    return cv2.applyColorMap(
        normalized_inverse_depth_uint8(depth),
        cv2.COLORMAP_TURBO,
    )


def depth_summary(depth):

    return {
        "depth_shape": list(depth.shape),
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "depth_mean_m": float(depth.mean()),
    }
