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
    print(f"Model loaded on device: {model.device}")

    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    writer = None
    frame_index = 0


    with output_paths["predictions"].open("w",        encoding="utf-8") as predictions_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            with torch.inference_mode():
                depth = model.infer_image(frame, args.input_size)



            depth = np.asarray(depth, dtype=np.float32)
            depth_bgr = mdeutils.colorize_depth_bgr(depth)
            if writer is None:
                height, width = depth_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                            str(output_paths["video"]),
                            fourcc,
                            fps,
                            (width, height),
                        ) # TODO: standarize writer for both YOLO and MDE
            frame_depth_path = (output_paths["depth_frames"] / f"{modelutils.slugify(source_label)}_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06d}_depth.npz")
            np.savez_compressed(frame_depth_path, depth=depth)

            frame_summary = mdeutils.depth_summary(depth)
            frame_predictions = {
                "depth_file": str(frame_depth_path) if frame_depth_path else None,
                **frame_summary,
            }
            predictions_file.write(json.dumps(frame_predictions) + "\n")
            writer.write(depth_bgr)

            if not args.no_display:
                cv2.imshow("Depth Estimation", depth_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Saved frame predictions to {output_paths['predictions']}")
    print(f"Saved colorized depth video to {output_paths['video']}")

if __name__ == "__main__":
    main()
