import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from capture_depth import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_ENCODER,
    DEFAULT_IMAGE_PATH,
    DEFAULT_INPUT_SIZE,
    DEFAULT_MAX_DEPTH,
    MODEL_CONFIGS,
    colorize_depth,
    colorize_depth_bgr,
    depth_summary,
    infer_depth_from_bgr,
    load_depth_model,
    model_label as depth_model_label,
    parse_source,
    resolve_path,
    slugify,
)
from capture_joints import DEFAULT_MODEL_PATH, boxes_to_detections, keypoints_to_people


def build_output_paths(output_dir, source_label, media_type, pose_model_path, depth_label):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    pose_label = Path(pose_model_path).stem
    label = f"{pose_label}__{depth_label}"
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
                "predictions": run_dir / f"{file_prefix}_predictions.json",
                "depth_array": run_dir / f"{file_prefix}_depth.npz",
                "depth_image": run_dir / f"{file_prefix}_depth.jpg",
                "joints_image": run_dir / f"{file_prefix}_joints.jpg",
            }
        )
    elif media_type == "video":
        paths.update(
            {
                "predictions": run_dir / f"{file_prefix}_predictions.jsonl",
                "depth_video": run_dir / f"{file_prefix}_depth.mp4",
                "joints_video": run_dir / f"{file_prefix}_joints.mp4",
                "depth_frames": run_dir / f"{file_prefix}_depth_frames",
            }
        )
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO pose tracking and Depth Anything metric depth on matching "
            "frames from the same image, video, or webcam source."
        )
    )
    parser.add_argument(
        "--source",
        "--image",
        dest="source",
        default=str(DEFAULT_IMAGE_PATH),
        help="Image path, video path, or webcam index.",
    )
    parser.add_argument(
        "--pose-model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the YOLO pose model file.",
    )
    parser.add_argument(
        "--pose-device",
        default="cpu",
        help="Device for YOLO pose inference. Defaults to CPU.",
    )
    parser.add_argument(
        "--depth-checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the Depth Anything V2 metric checkpoint.",
    )
    parser.add_argument(
        "--depth-encoder",
        default=DEFAULT_ENCODER,
        choices=sorted(MODEL_CONFIGS),
        help="Depth Anything V2 encoder size.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum metric depth used by the depth model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Depth model inference input size.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder where aligned predictions and previews are saved.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First source frame to process for video or webcam sources.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame for video or webcam sources.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional processed-frame limit for video or webcam sources.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open a live side-by-side preview for video or webcam sources.",
    )
    parser.add_argument(
        "--keypoint-depth-radius",
        type=int,
        default=2,
        help="Pixel radius used when sampling depth around each detected keypoint.",
    )
    save_depth_group = parser.add_mutually_exclusive_group()
    save_depth_group.add_argument(
        "--save-frame-depths",
        dest="save_frame_depths",
        action="store_true",
        help="Save raw metric depth arrays for every processed video frame.",
    )
    save_depth_group.add_argument(
        "--no-save-frame-depths",
        dest="save_frame_depths",
        action="store_false",
        help="Do not save raw metric depth arrays for every processed video frame.",
    )
    parser.set_defaults(save_frame_depths=True)
    return parser.parse_args()


def resolve_existing_file(path, description):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def infer_pose_from_bgr(model, image_bgr, device):
    results = model(image_bgr, verbose=False, device=device)
    result = results[0]
    return {
        "people": keypoints_to_people(result),
        "detections": boxes_to_detections(result),
        "annotated_frame": result.plot(),
    }


def sample_depth(depth, x, y, image_shape, radius):
    image_height, image_width = image_shape[:2]
    depth_height, depth_width = depth.shape[:2]

    if image_width <= 0 or image_height <= 0:
        return None, 0

    depth_x = int(round(float(x) * (depth_width - 1) / max(image_width - 1, 1)))
    depth_y = int(round(float(y) * (depth_height - 1) / max(image_height - 1, 1)))

    if depth_x < 0 or depth_x >= depth_width or depth_y < 0 or depth_y >= depth_height:
        return None, 0

    radius = max(int(radius), 0)
    x0 = max(depth_x - radius, 0)
    x1 = min(depth_x + radius + 1, depth_width)
    y0 = max(depth_y - radius, 0)
    y1 = min(depth_y + radius + 1, depth_height)

    patch = depth[y0:y1, x0:x1]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return None, 0

    return float(np.median(finite)), int(finite.size)


def attach_depth_to_people(people, depth, image_shape, radius):
    enriched_people = []
    for person in people:
        enriched_keypoints = []
        for keypoint in person["keypoints"]:
            depth_m, sample_count = sample_depth(
                depth=depth,
                x=keypoint["x"],
                y=keypoint["y"],
                image_shape=image_shape,
                radius=radius,
            )
            enriched_keypoints.append(
                {
                    **keypoint,
                    "depth_m": depth_m,
                    "depth_sample_count": sample_count,
                }
            )

        enriched_people.append({**person, "keypoints": enriched_keypoints})

    return enriched_people


def run_models_for_frame(
    executor,
    depth_model,
    pose_model,
    frame_bgr,
    depth_input_size,
    pose_device,
):
    depth_future = executor.submit(
        infer_depth_from_bgr,
        model=depth_model,
        image_bgr=frame_bgr.copy(),
        input_size=depth_input_size,
    )
    pose_future = executor.submit(
        infer_pose_from_bgr,
        model=pose_model,
        image_bgr=frame_bgr.copy(),
        device=pose_device,
    )

    depth, finite_depth = depth_future.result()
    pose = pose_future.result()
    return depth, finite_depth, pose


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def image_record(
    args,
    source_display,
    source_frame_index,
    frame_bgr,
    depth,
    finite_depth,
    pose,
    depth_array_path,
):
    depth_stats = depth_summary(depth, finite_depth)
    return {
        "frame_index": 0,
        "source_frame_index": source_frame_index,
        "time_s": 0.0,
        "source": source_display,
        "image_width": int(frame_bgr.shape[1]),
        "image_height": int(frame_bgr.shape[0]),
        "depth": {
            "file": str(depth_array_path),
            **depth_stats,
        },
        "joints": {
            "people": attach_depth_to_people(
                people=pose["people"],
                depth=depth,
                image_shape=frame_bgr.shape,
                radius=args.keypoint_depth_radius,
            ),
            "detections": pose["detections"],
        },
    }


def video_record(
    args,
    processed_count,
    source_frame_index,
    source_fps,
    frame_bgr,
    depth,
    finite_depth,
    pose,
    depth_frame_path,
):
    depth_stats = depth_summary(depth, finite_depth)
    return {
        "frame_index": processed_count,
        "source_frame_index": source_frame_index,
        "time_s": source_frame_index / source_fps if source_fps > 0 else None,
        "image_width": int(frame_bgr.shape[1]),
        "image_height": int(frame_bgr.shape[0]),
        "depth": {
            "file": str(depth_frame_path) if depth_frame_path else None,
            **depth_stats,
        },
        "joints": {
            "people": attach_depth_to_people(
                people=pose["people"],
                depth=depth,
                image_shape=frame_bgr.shape,
                radius=args.keypoint_depth_radius,
            ),
            "detections": pose["detections"],
        },
    }


def open_video_writer(path, fps, frame_bgr):
    height, width = frame_bgr.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def preview_frame(joints_bgr, depth_bgr):
    if joints_bgr.shape[:2] != depth_bgr.shape[:2]:
        depth_bgr = cv2.resize(
            depth_bgr,
            (joints_bgr.shape[1], joints_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.hstack([joints_bgr, depth_bgr])


def base_metadata(
    args,
    source_display,
    source_label,
    media_type,
    pose_model_path,
    depth_checkpoint_path,
    depth_label,
    depth_device,
    output_paths,
):
    return {
        "source": source_display,
        "source_label": source_label,
        "media_type": media_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "models": {
            "pose": {
                "path": str(pose_model_path),
                "device": args.pose_device,
            },
            "depth": {
                "label": depth_label,
                "checkpoint": str(depth_checkpoint_path),
                "encoder": args.depth_encoder,
                "max_depth": args.max_depth,
                "input_size": args.input_size,
                "device": depth_device,
            },
        },
        "alignment": {
            "key": "source_frame_index",
            "description": (
                "Each prediction record was created from a single source frame "
                "passed to both models in parallel."
            ),
            "keypoint_depth_radius_px": args.keypoint_depth_radius,
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


def process_image_source(
    args,
    image_path,
    source_label,
    source_display,
    depth_model,
    depth_device,
    pose_model,
    pose_model_path,
    depth_checkpoint_path,
    depth_label,
    output_paths,
):
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        depth, finite_depth, pose = run_models_for_frame(
            executor=executor,
            depth_model=depth_model,
            pose_model=pose_model,
            frame_bgr=frame_bgr,
            depth_input_size=args.input_size,
            pose_device=args.pose_device,
        )

    np.savez_compressed(output_paths["depth_array"], depth=depth)
    Image.fromarray(colorize_depth(depth)).save(
        output_paths["depth_image"],
        format="JPEG",
        quality=90,
    )
    cv2.imwrite(str(output_paths["joints_image"]), pose["annotated_frame"])

    record = image_record(
        args=args,
        source_display=source_display,
        source_frame_index=0,
        frame_bgr=frame_bgr,
        depth=depth,
        finite_depth=finite_depth,
        pose=pose,
        depth_array_path=output_paths["depth_array"],
    )
    write_json(output_paths["predictions"], record)

    metadata = {
        **base_metadata(
            args=args,
            source_display=source_display,
            source_label=source_label,
            media_type="image",
            pose_model_path=pose_model_path,
            depth_checkpoint_path=depth_checkpoint_path,
            depth_label=depth_label,
            depth_device=depth_device,
            output_paths=output_paths,
        ),
        "processed_frames": 1,
        **depth_summary(depth, finite_depth),
    }
    write_json(output_paths["metadata"], metadata)

    print(f"Saved aligned prediction to {output_paths['predictions']}")
    print(f"Saved raw depth array to {output_paths['depth_array']}")
    print(f"Saved colorized depth image to {output_paths['depth_image']}")
    print(f"Saved annotated joints image to {output_paths['joints_image']}")
    print(f"Saved metadata to {output_paths['metadata']}")


def process_video_source(
    args,
    capture_source,
    source_label,
    source_display,
    depth_model,
    depth_device,
    pose_model,
    pose_model_path,
    depth_checkpoint_path,
    depth_label,
    output_paths,
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

    joints_writer = None
    depth_writer = None
    processed_count = 0
    read_count = args.start_frame
    first_frame_index = None
    last_frame_index = None
    depth_min = np.inf
    depth_max = -np.inf
    depth_sum = 0.0
    depth_count = 0
    stop_reason = "end_of_stream"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            with output_paths["predictions"].open(
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

                    depth, finite_depth, pose = run_models_for_frame(
                        executor=executor,
                        depth_model=depth_model,
                        pose_model=pose_model,
                        frame_bgr=frame_bgr,
                        depth_input_size=args.input_size,
                        pose_device=args.pose_device,
                    )

                    depth_bgr = colorize_depth_bgr(depth)
                    if depth_writer is None:
                        depth_writer = open_video_writer(
                            output_paths["depth_video"],
                            output_fps,
                            depth_bgr,
                        )
                    if joints_writer is None:
                        joints_writer = open_video_writer(
                            output_paths["joints_video"],
                            output_fps,
                            pose["annotated_frame"],
                        )

                    depth_frame_path = None
                    if args.save_frame_depths:
                        depth_frame_path = (
                            output_paths["depth_frames"]
                            / f"{slugify(source_label)}_frame_{source_frame_index:06d}_depth.npz"
                        )
                        np.savez_compressed(depth_frame_path, depth=depth)

                    record = video_record(
                        args=args,
                        processed_count=processed_count,
                        source_frame_index=source_frame_index,
                        source_fps=source_fps,
                        frame_bgr=frame_bgr,
                        depth=depth,
                        finite_depth=finite_depth,
                        pose=pose,
                        depth_frame_path=depth_frame_path,
                    )
                    predictions_file.write(json.dumps(record) + "\n")

                    depth_writer.write(depth_bgr)
                    joints_writer.write(pose["annotated_frame"])

                    if first_frame_index is None:
                        first_frame_index = source_frame_index
                    last_frame_index = source_frame_index
                    depth_min = min(depth_min, record["depth"]["depth_min_m"])
                    depth_max = max(depth_max, record["depth"]["depth_max_m"])
                    depth_sum += float(finite_depth.sum())
                    depth_count += int(finite_depth.size)

                    processed_count += 1
                    if args.display:
                        cv2.imshow(
                            "AI Combat Coach Joints + Depth",
                            preview_frame(pose["annotated_frame"], depth_bgr),
                        )
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_reason = "display_quit"
                            break

                    if args.max_frames is not None and processed_count >= args.max_frames:
                        stop_reason = "max_frames"
                        break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        cap.release()
        if depth_writer is not None:
            depth_writer.release()
        if joints_writer is not None:
            joints_writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if processed_count == 0:
        raise ValueError(
            "No video frames were processed. Check --start-frame, --frame-stride, "
            "and --max-frames."
        )

    metadata = {
        **base_metadata(
            args=args,
            source_display=source_display,
            source_label=source_label,
            media_type="video",
            pose_model_path=pose_model_path,
            depth_checkpoint_path=depth_checkpoint_path,
            depth_label=depth_label,
            depth_device=depth_device,
            output_paths=output_paths,
        ),
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
        "stop_reason": stop_reason,
        "save_frame_depths": args.save_frame_depths,
        "depth_min_m": float(depth_min),
        "depth_max_m": float(depth_max),
        "depth_mean_m": float(depth_sum / depth_count),
    }
    write_json(output_paths["metadata"], metadata)

    print(f"Saved aligned frame predictions to {output_paths['predictions']}")
    if args.save_frame_depths:
        print(f"Saved raw frame depths to {output_paths['depth_frames']}")
    print(f"Saved colorized depth video to {output_paths['depth_video']}")
    print(f"Saved annotated joints video to {output_paths['joints_video']}")
    print(f"Saved metadata to {output_paths['metadata']}")


def main():
    args = parse_args()
    capture_source, source_label, media_type, source_display = parse_source(args.source)
    pose_model_path = resolve_existing_file(args.pose_model, "YOLO pose model")
    depth_checkpoint_path = resolve_existing_file(
        args.depth_checkpoint,
        "Depth Anything V2 metric checkpoint",
    )
    depth_label = depth_model_label(args.depth_encoder)

    output_paths = build_output_paths(
        output_dir=args.output,
        source_label=source_label,
        media_type=media_type,
        pose_model_path=pose_model_path,
        depth_label=depth_label,
    )

    depth_model, depth_device = load_depth_model(
        checkpoint_path=depth_checkpoint_path,
        encoder=args.depth_encoder,
        max_depth=args.max_depth,
    )
    pose_model = YOLO(str(pose_model_path))

    if media_type == "image":
        process_image_source(
            args=args,
            image_path=capture_source,
            source_label=source_label,
            source_display=source_display,
            depth_model=depth_model,
            depth_device=depth_device,
            pose_model=pose_model,
            pose_model_path=pose_model_path,
            depth_checkpoint_path=depth_checkpoint_path,
            depth_label=depth_label,
            output_paths=output_paths,
        )
    else:
        process_video_source(
            args=args,
            capture_source=capture_source,
            source_label=source_label,
            source_display=source_display,
            depth_model=depth_model,
            depth_device=depth_device,
            pose_model=pose_model,
            pose_model_path=pose_model_path,
            depth_checkpoint_path=depth_checkpoint_path,
            depth_label=depth_label,
            output_paths=output_paths,
        )


if __name__ == "__main__":
    main()
