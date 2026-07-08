import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

repo_root = Path(__file__).resolve().parent
depth_anything_metric_root = (
    repo_root / "models" / "depth" / "Depth-Anything-V2" / "metric_depth"
)
sys.path.insert(0, str(depth_anything_metric_root))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

DEFAULT_IMAGE_PATH = (
    repo_root / "media" / "videos" / "Rodtang-taetat-2.mp4"
)
DEFAULT_CHECKPOINT_PATH = (
    depth_anything_metric_root
    / "checkpoints"
    / "depth_anything_v2_metric_hypersim_vitl.pth"
)
DEFAULT_ENCODER = "vitl"
DEFAULT_MAX_DEPTH = 20.0
DEFAULT_INPUT_SIZE = 200
MODEL_LABEL_PREFIX = "depth-anything-v2-metric-hypersim"

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

MODEL_CONFIGS = {
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


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def model_label(encoder):
    return f"{MODEL_LABEL_PREFIX}-{encoder}"


def parse_source(source):
    source_text = str(source).strip()
    if source_text.isdigit():
        return int(source_text), f"webcam-{source_text}", "video", f"webcam-{source_text}"

    source_path = resolve_path(source_text)
    if not source_path.is_file():
        raise FileNotFoundError(f"Input source not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return source_path, source_path.stem, "image", str(source_path)
    if suffix in VIDEO_EXTENSIONS:
        return str(source_path), source_path.stem, "video", str(source_path)

    supported_extensions = sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    raise ValueError(
        f"Unsupported source type for {source_path}. "
        f"Supported extensions: {', '.join(supported_extensions)}"
    )


def build_output_paths(output_dir, source_label, media_type, label):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = f"{slugify(source_label)}__{slugify(label)}__{run_id}"
    run_dir = resolve_path(output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{slugify(source_label)}__{slugify(label)}"
    paths = {
        "run_dir": run_dir,
        "metadata": run_dir / f"{file_prefix}_metadata.json",
    }
    if media_type == "image":
        paths.update(
            {
                "predictions": run_dir / f"{file_prefix}_predictions.npz",
                "depth_image": run_dir / f"{file_prefix}_depth.jpg",
            }
        )
    elif media_type == "video":
        paths.update(
            {
                "depth_video": run_dir / f"{file_prefix}_depth.mp4",
                "frame_predictions": run_dir / f"{file_prefix}_frame_predictions.jsonl",
                "depth_frames": run_dir / f"{file_prefix}_depth_frames",
            }
        )
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    return paths


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_state_dict(checkpoint_path):
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def load_depth_model(checkpoint_path, encoder, max_depth):
    device = select_device()
    model = DepthAnythingV2(
        **{
            **MODEL_CONFIGS[encoder],
            "max_depth": max_depth,
        }
    )
    model.load_state_dict(load_state_dict(checkpoint_path))
    return model.to(device).eval(), device


def infer_depth_from_bgr(model, image_bgr, input_size):
    if image_bgr is None:
        raise ValueError("OpenCV returned an empty frame.")

    with torch.inference_mode():
        depth = model.infer_image(image_bgr, input_size)

    depth = np.asarray(depth, dtype=np.float32)
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size == 0:
        raise ValueError("The model returned no finite depth values.")

    return depth, finite_depth


def infer_depth(model, image_path, input_size):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    return infer_depth_from_bgr(model, image_bgr, input_size)


def depth_summary(depth, finite_depth=None):
    if finite_depth is None:
        finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size == 0:
        raise ValueError("The model returned no finite depth values.")

    return {
        "depth_shape": list(depth.shape),
        "depth_min_m": float(finite_depth.min()),
        "depth_max_m": float(finite_depth.max()),
        "depth_mean_m": float(finite_depth.mean()),
    }


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "--image",
        dest="source",
        default=str(DEFAULT_IMAGE_PATH),
        help="Image path, video path, or webcam index.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the Depth Anything V2 metric checkpoint.",
    )
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        choices=sorted(MODEL_CONFIGS),
        help="Depth Anything V2 encoder size.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum metric depth used by the model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Inference input size.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder where depth predictions are saved.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First source frame to process for video sources.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame for video sources.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional processed-frame limit for video sources.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open a live depth preview window for video sources.",
    )
    save_depth_group = parser.add_mutually_exclusive_group()
    save_depth_group.add_argument(
        "--save-frame-depths",
        dest="save_frame_depths",
        action="store_true",
        help="Save raw per-frame depth arrays for video sources.",
    )
    save_depth_group.add_argument(
        "--no-save-frame-depths",
        dest="save_frame_depths",
        action="store_false",
        help="Do not save raw per-frame depth arrays for video sources.",
    )
    parser.set_defaults(save_frame_depths=False)
    return parser.parse_args()


def base_metadata(args, source, checkpoint_path, label, device):
    return {
        "source": str(source),
        "model": label,
        "checkpoint": str(checkpoint_path),
        "encoder": args.encoder,
        "max_depth": args.max_depth,
        "input_size": args.input_size,
        "device": device,
        "focallength_px": None,
    }


def process_image_source(args, image_path, model, device, output_paths, label, checkpoint_path):
    depth, finite_depth = infer_depth(
        model=model,
        image_path=image_path,
        input_size=args.input_size,
    )

    np.savez_compressed(output_paths["predictions"], depth=depth)
    Image.fromarray(colorize_depth(depth)).save(
        output_paths["depth_image"],
        format="JPEG",
        quality=90,
    )

    metadata = {
        **base_metadata(args, image_path, checkpoint_path, label, device),
        **depth_summary(depth, finite_depth),
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved depth predictions to {output_paths['predictions']}")
    print(f"Saved colorized depth to {output_paths['depth_image']}")
    print(f"Saved metadata to {output_paths['metadata']}")


def process_video_source(
    args,
    capture_source,
    source_label,
    source_display,
    model,
    device,
    output_paths,
    label,
    checkpoint_path,
):
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be at least 0.")

    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {source_display}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output_fps = max(source_fps / args.frame_stride, 1.0) if source_fps > 0 else 30.0

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    if args.save_frame_depths:
        output_paths["depth_frames"].mkdir(parents=True, exist_ok=True)

    writer = None
    processed_count = 0
    read_count = args.start_frame
    first_frame_index = None
    last_frame_index = None
    depth_min = np.inf
    depth_max = -np.inf
    depth_sum = 0.0
    depth_count = 0

    try:
        with output_paths["frame_predictions"].open(
            "w",
            encoding="utf-8",
        ) as predictions_file:
            while cap.isOpened():
                success, frame_bgr = cap.read()
                if not success:
                    break

                source_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
                if source_frame_index < 0:
                    source_frame_index = read_count
                read_count = source_frame_index + 1

                if source_frame_index < args.start_frame:
                    continue
                if (source_frame_index - args.start_frame) % args.frame_stride != 0:
                    continue

                depth, finite_depth = infer_depth_from_bgr(
                    model=model,
                    image_bgr=frame_bgr,
                    input_size=args.input_size,
                )
                depth_bgr = colorize_depth_bgr(depth)

                if writer is None:
                    height, width = depth_bgr.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_paths["depth_video"]),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open video writer: {output_paths['depth_video']}"
                        )

                frame_depth_path = None
                if args.save_frame_depths:
                    frame_depth_path = (
                        output_paths["depth_frames"]
                        / f"{slugify(source_label)}_frame_{source_frame_index:06d}_depth.npz"
                    )
                    np.savez_compressed(frame_depth_path, depth=depth)

                frame_summary = depth_summary(depth, finite_depth)
                frame_predictions = {
                    "frame_index": processed_count,
                    "source_frame_index": source_frame_index,
                    "time_s": (
                        source_frame_index / source_fps if source_fps > 0 else None
                    ),
                    "depth_file": str(frame_depth_path) if frame_depth_path else None,
                    **frame_summary,
                }
                predictions_file.write(json.dumps(frame_predictions) + "\n")
                writer.write(depth_bgr)

                if first_frame_index is None:
                    first_frame_index = source_frame_index
                last_frame_index = source_frame_index
                depth_min = min(depth_min, frame_summary["depth_min_m"])
                depth_max = max(depth_max, frame_summary["depth_max_m"])
                depth_sum += float(finite_depth.sum())
                depth_count += int(finite_depth.size)

                processed_count += 1
                if args.display:
                    cv2.imshow("AI Combat Coach Depth", depth_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.max_frames is not None and processed_count >= args.max_frames:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if processed_count == 0:
        raise ValueError(
            "No video frames were processed. Check --start-frame, --frame-stride, "
            "and --max-frames."
        )

    metadata = {
        **base_metadata(args, source_display, checkpoint_path, label, device),
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "source_width": source_width,
        "source_height": source_height,
        "output_fps": output_fps,
        "start_frame": args.start_frame,
        "frame_stride": args.frame_stride,
        "processed_frames": processed_count,
        "first_source_frame_index": first_frame_index,
        "last_source_frame_index": last_frame_index,
        "depth_min_m": float(depth_min),
        "depth_max_m": float(depth_max),
        "depth_mean_m": float(depth_sum / depth_count),
        "save_frame_depths": args.save_frame_depths,
        "frame_predictions": str(output_paths["frame_predictions"]),
        "depth_video": str(output_paths["depth_video"]),
        "depth_frames": (
            str(output_paths["depth_frames"]) if args.save_frame_depths else None
        ),
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved frame predictions to {output_paths['frame_predictions']}")
    if args.save_frame_depths:
        print(f"Saved raw frame depths to {output_paths['depth_frames']}")
    print(f"Saved colorized depth video to {output_paths['depth_video']}")
    print(f"Saved metadata to {output_paths['metadata']}")


def main():
    args = parse_args()
    capture_source, source_label, media_type, source_display = parse_source(args.source)
    checkpoint_path = resolve_path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Depth Anything V2 metric checkpoint not found at {checkpoint_path}. "
            "Download it into models/depth/Depth-Anything-V2/metric_depth/checkpoints first."
        )

    label = model_label(args.encoder)
    output_paths = build_output_paths(
        output_dir=args.output,
        source_label=source_label,
        media_type=media_type,
        label=label,
    )

    model, device = load_depth_model(
        checkpoint_path=checkpoint_path,
        encoder=args.encoder,
        max_depth=args.max_depth,
    )

    if media_type == "image":
        process_image_source(
            args=args,
            image_path=capture_source,
            model=model,
            device=device,
            output_paths=output_paths,
            label=label,
            checkpoint_path=checkpoint_path,
        )
    else:
        process_video_source(
            args=args,
            capture_source=capture_source,
            source_label=source_label,
            source_display=source_display,
            model=model,
            device=device,
            output_paths=output_paths,
            label=label,
            checkpoint_path=checkpoint_path,
        )


if __name__ == "__main__":
    main()
