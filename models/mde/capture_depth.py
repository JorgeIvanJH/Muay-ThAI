import os
import argparse
import json
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import models.utils as modelutils
import utils as mdeutils
import config as mdecfg


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
        help="Inference input size. Good sizes are multiples of 14, like 196, 224, 280, 336",
    )
    parser.add_argument(
        "--half",
        type=bool,
        default=mdecfg.MDE_HALF,
        help="Use float16 model weights on CUDA for faster inference.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=mdecfg.MDE_FRAME_STRIDE,
        help="Process every Nth frame for video sources.",
    )
    parser.add_argument(
        "--depth-format",
        choices=("npz", "npy"),
        default=mdecfg.MDE_DEPTH_FORMAT,
        help="Raw depth frame format. npy is faster; npz is smaller.",
    )
    save_depth_group = parser.add_mutually_exclusive_group()## TODO: change to single argument with default True
    save_depth_group.add_argument(
        "--save-frame-depths",
        dest="save_frame_depths",
        action="store_true",
        help="Save raw metric depth arrays for every processed frame.",
    )
    save_depth_group.add_argument(
        "--no-save-frame-depths",
        dest="save_frame_depths",
        action="store_false",
        help="Skip raw per-frame depth arrays for faster video processing.",
    )
    parser.set_defaults(save_frame_depths=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1.")
    if args.input_size < 1:
        raise ValueError("--input-size must be at least 1.")

    capture_source, source_label = modelutils.parse_source(args.source)
    output_paths = modelutils.build_output_paths(args.output, source_label, args.model)
    output_paths["depth_frames"] = output_paths["run_dir"] / "depth_frames"
    if args.save_frame_depths:
        output_paths["depth_frames"].mkdir(parents=True, exist_ok=True)

    model = mdeutils.safe_model_load(args.model, use_half=args.half)
    mdeutils.configure_pipeline_for_inference(model, input_size=args.input_size)
    print(f"Model loaded on device: {model.device}")

    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    output_fps = max(fps / args.frame_stride, 1.0)

    writer = None
    processed_index = 0
    source_frame_index = -1

    with output_paths["predictions"].open("w", encoding="utf-8") as predictions_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            source_frame_index += 1
            if source_frame_index % args.frame_stride != 0:
                continue

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
                            output_fps,
                            (width, height),
                        ) # TODO: standarize writer for both YOLO and MDE
            frame_depth_path = None
            if args.save_frame_depths:
                frame_depth_path = (
                    output_paths["depth_frames"]
                    / (
                        f"{modelutils.slugify(source_label)}_frame_"
                        f"{source_frame_index:06d}_depth.{args.depth_format}"
                    )
                )
                mdeutils.save_depth_array(frame_depth_path, depth, args.depth_format)

            frame_summary = mdeutils.depth_summary(depth)
            frame_predictions = {
                "frame_index": processed_index,
                "source_frame_index": source_frame_index,
                "depth_file": str(frame_depth_path) if frame_depth_path else None,
                **frame_summary,
            }
            predictions_file.write(json.dumps(frame_predictions) + "\n")
            writer.write(depth_bgr)

            if not args.no_display:
                cv2.imshow("Depth Estimation", depth_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed_index += 1
            if args.max_frames is not None and processed_index >= args.max_frames:
                break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Saved frame predictions to {output_paths['predictions']}")
    print(f"Saved colorized depth video to {output_paths['video']}")


if __name__ == "__main__":
    main()
