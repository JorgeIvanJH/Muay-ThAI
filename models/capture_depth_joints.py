import argparse
import json
import sys
import time
from datetime import datetime
from multiprocessing import Barrier, Process, freeze_support
from pathlib import Path
from threading import BrokenBarrierError

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import models.config as modelcfg
import models.utils as modelutils
from models.mde import config as mdecfg
from models.mde import utils as mdeutils
from models.yolo import config as yolocfg
from models.yolo import utils as yoloutils


FRAME_BARRIER_TIMEOUT_SECONDS = 120


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run metric MDE and YOLO pose tracking in parallel processes."
    )
    parser.add_argument(
        "--source",
        default=str(modelcfg.YOLOMDE_INPUT),
        help="Video path or webcam index.",
    )
    parser.add_argument(
        "--depth-model",
        default=str(mdecfg.MDE_HF_WEIGHTS),
        help="Path to the Depth Anything-V2-metric model weights.",
    )
    parser.add_argument(
        "--yolo-model",
        default=str(yolocfg.YOLO_WEIGHTS),
        help="Path to the YOLO pose model weights.",
    )
    parser.add_argument(
        "--output",
        default=str(modelcfg.YOLOMDE_OUTPUT),
        help="Folder where predictions and videos are saved.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save output without opening preview windows.",
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
        help="Maximum metric depth used by the depth model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=modelcfg.YOLOMDE_INPUT_SIZE,
        help="Inference input size. Good sizes are multiples of 14, like 196, 224, 280, 336",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=mdecfg.MDE_HALF,
        help="Use float16 depth weights on CUDA for faster inference.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=modelcfg.YOLOMDE_FRAME_STRIDE,
        help="Process every Nth frame.",
    )
    parser.add_argument(
        "--depth-format",
        choices=("npz", "npy"),
        default=mdecfg.MDE_DEPTH_FORMAT,
        help="Raw depth frame format. npy is faster; npz is smaller.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=yolocfg.YOLO_SMOOTHING_ALPHA,
        help="Temporal smoothing factor for YOLO joint xy positions.",
    )
    parser.add_argument(
        "--yolo-device",
        default="cpu",
        help="Device for YOLO pose inference. Defaults to CPU.",
    )
    parser.add_argument(
        "--depth-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for depth inference. Defaults to CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--save-frame-joints",
        action=argparse.BooleanOptionalAction,
        default=modelcfg.YOLOMDE_SAVE_JOINTS,
        help="Save raw joint xy positions for every processed frame.",
    )
    parser.add_argument(
        "--save-frame-depths",
        action=argparse.BooleanOptionalAction,
        default=modelcfg.YOLOMDE_SAVE_DEPTHS,
        help="Save raw metric depth arrays for every processed frame.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1.")
    if args.input_size < 1:
        raise ValueError("--input-size must be at least 1.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1.")
    if not 0.0 <= args.smooth_alpha <= 1.0:
        raise ValueError("--smooth-alpha must be between 0.0 and 1.0.")
    if (
        str(args.depth_device).split(":", 1)[0].lower() == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError("--depth-device cuda was requested, but CUDA is not available.")


def build_combined_output_paths(output_dir, source_label, yolo_model_path, depth_model_path):
    source_slug = modelutils.slugify(source_label)
    yolo_slug = modelutils.slugify(Path(yolo_model_path).stem)
    depth_slug = modelutils.slugify(Path(depth_model_path).stem)
    model_slug = f"{yolo_slug}-{depth_slug}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{source_slug}__{model_slug}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{source_slug}__{model_slug}"
    return {
        "run_dir": run_dir,
        "predictions": run_dir / f"{file_prefix}_predictions.jsonl",
        "metadata": run_dir / f"{file_prefix}_metadata.json",
        "yolo_predictions": run_dir / f"{file_prefix}_yolo_predictions.jsonl",
        "depth_predictions": run_dir / f"{file_prefix}_depth_predictions.jsonl",
        "yolo_video": run_dir / f"{file_prefix}_yolo_annotated.mp4",
        "depth_video": run_dir / f"{file_prefix}_depth_annotated.mp4",
        "depth_frames": run_dir / "depth_frames",
    }


def wait_for_matching_frame(frame_barrier, processed_index):
    try:
        frame_barrier.wait(timeout=FRAME_BARRIER_TIMEOUT_SECONDS)
    except BrokenBarrierError as exc:
        raise RuntimeError(
            f"Timed out waiting for both models at processed frame {processed_index}."
        ) from exc


def read_jsonl_by_source_frame(path):
    records = {}
    with path.open("r", encoding="utf-8") as records_file:
        for line in records_file:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["source_frame_index"]] = record
    return records


def depth_at_keypoint(depth, x, y, image_width, image_height):
    if x is None or y is None or image_width <= 0 or image_height <= 0:
        return None

    depth_height, depth_width = depth.shape[:2]
    depth_x = int(round(float(x) * (depth_width - 1) / max(image_width - 1, 1)))
    depth_y = int(round(float(y) * (depth_height - 1) / max(image_height - 1, 1)))
    if depth_x < 0 or depth_y < 0 or depth_x >= depth_width or depth_y >= depth_height:
        return None

    value = float(depth[depth_y, depth_x])
    if not np.isfinite(value):
        return None
    return value


def add_depths_to_people(people, depth, image_width, image_height):
    for person in people:
        for keypoint in person.get("keypoints", []):
            keypoint["depth_m"] = depth_at_keypoint(
                depth,
                keypoint.get("x"),
                keypoint.get("y"),
                image_width,
                image_height,
            )


def load_depth_file(path):
    depth_data = np.load(path)
    if isinstance(depth_data, np.lib.npyio.NpzFile):
        try:
            return np.asarray(depth_data["depth"], dtype=np.float32)
        finally:
            depth_data.close()
    return np.asarray(depth_data, dtype=np.float32)


def merge_predictions(output_paths):
    yolo_records = read_jsonl_by_source_frame(output_paths["yolo_predictions"])
    depth_records = read_jsonl_by_source_frame(output_paths["depth_predictions"])
    source_frame_indices = sorted(set(yolo_records) & set(depth_records))

    with output_paths["predictions"].open("w", encoding="utf-8") as predictions_file:
        for frame_index, source_frame_index in enumerate(source_frame_indices):
            yolo_record = yolo_records[source_frame_index]
            depth_record = depth_records[source_frame_index]

            depth_file = depth_record["depth"].get("file")
            if depth_file:
                depth = load_depth_file(output_paths["run_dir"] / depth_file)
                add_depths_to_people(
                    yolo_record["joints"]["people"],
                    depth,
                    yolo_record["image_width"],
                    yolo_record["image_height"],
                )

            frame_predictions = {
                "frame_index": frame_index,
                "source_frame_index": source_frame_index,
                "source": yolo_record["source"],
                "image_width": yolo_record["image_width"],
                "image_height": yolo_record["image_height"],
                "joints": yolo_record["joints"],
                "depth": depth_record["depth"],
                "inference": {
                    "yolo": yolo_record["inference"],
                    "depth": depth_record["inference"],
                },
            }
            predictions_file.write(json.dumps(frame_predictions) + "\n")

    unmatched_frames = sorted(set(yolo_records) ^ set(depth_records))
    return len(source_frame_indices), unmatched_frames


def write_metadata(args, output_paths, source_label, frame_count, unmatched_frames):
    metadata = {
        "source": args.source,
        "source_label": source_label,
        "frame_count": frame_count,
        "unmatched_source_frame_indices": unmatched_frames,
        "models": {
            "yolo": {
                "path": str(args.yolo_model),
                "device": args.yolo_device,
            },
            "depth": {
                "path": str(args.depth_model),
                "device": args.depth_device,
                "input_size": args.input_size,
                "half": args.half,
            },
        },
        "frame_stride": args.frame_stride,
        "outputs": {
            "predictions": output_paths["predictions"].name,
            "yolo_predictions": output_paths["yolo_predictions"].name,
            "depth_predictions": output_paths["depth_predictions"].name,
            "yolo_video": output_paths["yolo_video"].name,
            "depth_video": output_paths["depth_video"].name,
            "depth_frames": output_paths["depth_frames"].name,
        },
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_depth_capture(args, output_paths, frame_barrier):
    capture_source, source_label = modelutils.parse_source(args.source)
    if args.save_frame_depths:
        output_paths["depth_frames"].mkdir(parents=True, exist_ok=True)

    model = mdeutils.safe_model_load(
        args.depth_model,
        use_half=args.half
    )
    mdeutils.configure_pipeline_for_inference(model, input_size=args.input_size)
    print(f"Depth model loaded on device: {args.depth_device}")

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

    with output_paths["depth_predictions"].open("w", encoding="utf-8") as predictions_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            source_frame_index += 1
            if source_frame_index % args.frame_stride != 0:
                continue

            wait_for_matching_frame(frame_barrier, processed_index)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inference_started_at = time.time()
            with torch.inference_mode():
                result = model(image)
            inference_finished_at = time.time()

            depth = result["predicted_depth"].detach().cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)
            depth = np.squeeze(depth)
            depth_bgr = mdeutils.colorize_depth_bgr(depth)
            if writer is None:
                height, width = depth_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_paths["depth_video"]),
                    fourcc,
                    output_fps,
                    (width, height),
                )

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

            frame_predictions = {
                "frame_index": processed_index,
                "source_frame_index": source_frame_index,
                "source": args.source,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "depth": {
                    "file": str(frame_depth_path.relative_to(output_paths["run_dir"]))
                    if frame_depth_path
                    else None,
                    **mdeutils.depth_summary(depth),
                },
                "inference": {
                    "model": str(args.depth_model),
                    "device": args.depth_device,
                    "started_at": inference_started_at,
                    "finished_at": inference_finished_at,
                },
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


def run_yolo_capture(args, output_paths, frame_barrier):
    capture_source, _ = modelutils.parse_source(args.source)
    model = YOLO(args.yolo_model)
    print(f"YOLO model loaded for device: {args.yolo_device}")

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
    previous_smoothed_keypoints = None

    with output_paths["yolo_predictions"].open("w", encoding="utf-8") as predictions_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            source_frame_index += 1
            if source_frame_index % args.frame_stride != 0:
                continue

            wait_for_matching_frame(frame_barrier, processed_index)
            inference_started_at = time.time()
            results = model(frame, verbose=False, device=args.yolo_device)
            inference_finished_at = time.time()
            # temporal smoothing
            previous_smoothed_keypoints = yoloutils.smooth_result_keypoints(
                results[0],
                previous_smoothed_keypoints,
                args.smooth_alpha,
            )
            annotated_frame = results[0].plot()

            if writer is None:
                height, width = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_paths["yolo_video"]),
                    fourcc,
                    output_fps,
                    (width, height),
                )

            people = yoloutils.keypoints_to_people(results[0]) if args.save_frame_joints else []
            frame_predictions = {
                "frame_index": processed_index,
                "source_frame_index": source_frame_index,
                "source": args.source,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "joints": {
                    "model": str(args.yolo_model),
                    "people": people,
                    "boxes": yoloutils.boxes_to_detections(results[0]),
                },
                "inference": {
                    "model": str(args.yolo_model),
                    "device": args.yolo_device,
                    "started_at": inference_started_at,
                    "finished_at": inference_finished_at,
                },
            }
            predictions_file.write(json.dumps(frame_predictions) + "\n")
            writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow("AI Combat Coach", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed_index += 1
            if args.max_frames is not None and processed_index >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    validate_args(args)
    capture_source, source_label = modelutils.parse_source(args.source)
    output_paths = build_combined_output_paths(
        args.output,
        source_label,
        args.yolo_model,
        args.depth_model,
    )

    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {args.source}")
    cap.release()

    # Run YOLO and MDE in parallel processes.
    frame_barrier = Barrier(2)
    processes = [
        Process(
            target=run_depth_capture,
            args=(args, output_paths, frame_barrier),
            name="mde-depth",
        ),
        Process(
            target=run_yolo_capture,
            args=(args, output_paths, frame_barrier),
            name="yolo-pose",
        ),
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    failed = [process.name for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"Parallel capture failed in: {', '.join(failed)}")

    frame_count, unmatched_frames = merge_predictions(output_paths)
    write_metadata(args, output_paths, source_label, frame_count, unmatched_frames)

    print(f"Saved combined predictions to {output_paths['predictions']}")
    print(f"Saved metadata to {output_paths['metadata']}")
    print(f"Saved YOLO video to {output_paths['yolo_video']}")
    print(f"Saved depth video to {output_paths['depth_video']}")


if __name__ == "__main__":
    freeze_support()
    main()
