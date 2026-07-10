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

def main():
    args = parse_args()

    capture_source, source_label = modelutils.parse_source(args.source)
    output_paths = modelutils.build_output_paths(args.output, source_label, args.model)
    output_paths["depth_frames"] = output_paths["run_dir"] / "depth_frames"
    output_paths["depth_frames"].mkdir(parents=True, exist_ok=True)

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

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                result = model(image)

            depth = result["predicted_depth"].detach().cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)
            depth = np.squeeze(depth)
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
            frame_depth_path = (
                output_paths["depth_frames"]
                / f"{modelutils.slugify(source_label)}_frame_{frame_index:06d}_depth.npz"
            )
            np.savez_compressed(frame_depth_path, depth=depth)

            frame_summary = mdeutils.depth_summary(depth)
            frame_predictions = {
                "frame_index": frame_index,
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
