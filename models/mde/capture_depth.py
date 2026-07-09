import os
import sys
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import models.utils as modelutils
import utils as mdeutils
import config as mdecfg

from transformers import pipeline
from PIL import Image



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


def load_depth_model(checkpoint_path, max_depth):
    device = select_device()
    model = DepthAnythingV2(
        **{
            **mdecfg.MDE_ENCODER_CONFIG,
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
        "--model",
        default=str(mdecfg.MDE_HF_WEIGHTS),
        help="Path to the Depth Anything-V2-metric model weights.",
    )
    parser.add_argument(
        "--source",
        dest="source",
        default=str(mdecfg.MDE_INPUT),
        help="Video path or webcam index.",
    )
    parser.add_argument(
        "--output",
        default=str(mdecfg.MDE_OUTPUT),
        help="Folder where depth predictions and video are saved.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save output without opening the preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for quick test runs.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=mdecfg.MDE_MAX_DEPTH,
        help="Maximum metric depth used by the model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=mdecfg.MDE_INPUT_SIZE,
        help="Inference input size.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame for video sources.",
    )
    return parser.parse_args()


def base_metadata(args, source, checkpoint_path, label, device):
    return {
        "source": str(source),
        "model": label,
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
                        / f"{modelutils.slugify(source_label)}_frame_{source_frame_index:06d}_depth.npz"
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

    capture_source, source_label = modelutils.parse_source(args.source)
    output_paths = modelutils.build_output_paths(args.output, source_label, args.model)
    

    model = mdeutils.safe_model_load(args.model)

    breakpoint()
    model, device = load_depth_model(
        checkpoint_path=args.model,
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
