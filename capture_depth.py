import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

repo_root = Path(__file__).resolve().parent
depth_pro_root = repo_root / "models" / "depth" / "ml-depth-pro"
sys.path.insert(0, str(depth_pro_root / "src"))

import depth_pro  # noqa: E402
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig  # noqa: E402


DEFAULT_IMAGE_PATH = depth_pro_root / "data" / "example.jpg"
DEFAULT_CHECKPOINT_PATH = depth_pro_root / "checkpoints" / "depth_pro.pt"
MODEL_LABEL = "depth-pro"


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def build_output_paths(output_dir, image_path):
    source_label = image_path.stem
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = f"{slugify(source_label)}__{slugify(MODEL_LABEL)}__{run_id}"
    run_dir = resolve_path(output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{slugify(source_label)}__{slugify(MODEL_LABEL)}"
    return {
        "run_dir": run_dir,
        "predictions": run_dir / f"{file_prefix}_predictions.npz",
        "depth_image": run_dir / f"{file_prefix}_depth.jpg",
        "metadata": run_dir / f"{file_prefix}_metadata.json",
    }


def build_depth_pro_config(checkpoint_path):
    return DepthProConfig(
        patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
        image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
        decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
        checkpoint_uri=str(checkpoint_path),
        fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
        use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
    )


def tensor_to_python(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    return value


def colorize_depth(depth):
    inverse_depth = 1.0 / np.clip(depth, 1e-4, None)
    max_invdepth = min(float(np.nanmax(inverse_depth)), 1 / 0.1)
    min_invdepth = max(float(np.nanmin(inverse_depth)), 1 / 250)
    scale = max(max_invdepth - min_invdepth, 1e-8)
    normalized = np.clip((inverse_depth - min_invdepth) / scale, 0.0, 1.0)

    color_depth = plt.get_cmap("turbo")(normalized)[..., :3]
    return (color_depth * 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        "--source",
        dest="image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Image path to run Depth Pro on.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the Depth Pro checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder where depth predictions are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = resolve_path(args.image)
    checkpoint_path = resolve_path(args.checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Depth Pro checkpoint not found at {checkpoint_path}. "
            "Run get_pretrained_models.sh from models/depth/ml-depth-pro first."
        )

    output_paths = build_output_paths(args.output, image_path)

    model, transform = depth_pro.create_model_and_transforms(
        config=build_depth_pro_config(checkpoint_path)
    )
    model.eval()

    image, _, f_px = depth_pro.load_rgb(image_path)
    prediction = model.infer(transform(image), f_px=f_px)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    focallength_px = tensor_to_python(prediction["focallength_px"])

    prediction_arrays = {"depth": depth}
    if focallength_px is not None:
        prediction_arrays["focallength_px"] = np.array(
            focallength_px,
            dtype=np.float32,
        )
    np.savez_compressed(output_paths["predictions"], **prediction_arrays)
    Image.fromarray(colorize_depth(depth)).save(
        output_paths["depth_image"],
        format="JPEG",
        quality=90,
    )

    metadata = {
        "source": str(image_path),
        "model": MODEL_LABEL,
        "checkpoint": str(checkpoint_path),
        "depth_shape": list(depth.shape),
        "focallength_px": focallength_px,
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved depth predictions to {output_paths['predictions']}")
    print(f"Saved colorized depth to {output_paths['depth_image']}")
    print(f"Saved metadata to {output_paths['metadata']}")


if __name__ == "__main__":
    main()
