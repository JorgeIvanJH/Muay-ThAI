"""Build frame-level raw pose and action-label CSV files.

Each output CSV contains one row per detected person/box per decoded video
frame. Frames without detections retain one row with empty detection fields so
the temporal sequence stays aligned. Subject selection is intentionally
deferred to the shared model preprocessing pipeline.

Joint and bounding-box coordinates are stored in pixels alongside the frame
dimensions. Normalization is intentionally deferred to the shared model
preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.yolo import config as yolocfg
from models.yolo import utils as yoloutils


DEFAULT_ANNOTATIONS = (
    ROOT_DIR
    / "dataset"
    / "classification"
    / "project-1-at-2026-07-29-11-24-494107dd.json"
)
DEFAULT_VIDEO_DIR = ROOT_DIR / "media" / "videos" / "30fps"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dataset" / "jointswithactionlabels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO pose inference and join each frame to its action label."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Simplified Label Studio JSON export.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing the CFR videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for one output CSV per video.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=yolocfg.YOLO_WEIGHTS,
        help="YOLO pose-model weights.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional per-video frame limit for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace CSV files that already exist.",
    )
    return parser.parse_args()


def load_annotation_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as annotations_file:
        tasks = json.load(annotations_file)

    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Expected a non-empty list of tasks in {path}")
    return tasks


def build_frame_labels(task: dict) -> tuple[dict[int, str], int]:
    """Return the 1-based frame-to-label mapping and final labeled frame."""
    labels: dict[int, str] = {}

    for result in task.get("videoLabels", []):
        action_labels = result.get("timelinelabels", [])
        if len(action_labels) != 1:
            raise ValueError(
                f"Task {task.get('id')} has a range without exactly one label"
            )
        action_label = action_labels[0]

        for frame_range in result.get("ranges", []):
            start = int(frame_range["start"])
            end = int(frame_range["end"])
            if start < 1 or end < start:
                raise ValueError(
                    f"Task {task.get('id')} has invalid frame range {start}-{end}"
                )

            for label_frame in range(start, end + 1):
                existing = labels.get(label_frame)
                if existing is not None and existing != action_label:
                    raise ValueError(
                        f"Task {task.get('id')} has overlapping labels at "
                        f"frame {label_frame}: {existing!r} and {action_label!r}"
                    )
                labels[label_frame] = action_label

    if not labels:
        raise ValueError(f"Task {task.get('id')} contains no video labels")

    final_frame = max(labels)
    missing = [frame for frame in range(1, final_frame + 1) if frame not in labels]
    if missing:
        preview = ", ".join(str(frame) for frame in missing[:10])
        raise ValueError(
            f"Task {task.get('id')} has unlabeled frames; first missing: {preview}"
        )

    return labels, final_frame


def resolve_video_path(label_studio_video: str, video_dir: Path) -> Path:
    """Resolve either the exported upload name or its prefix-free CFR name."""
    exported_name = Path(label_studio_video).name
    exact_path = video_dir / exported_name
    if exact_path.is_file():
        return exact_path

    # Label Studio prepends an upload token, e.g. 9687240c-video.mp4.
    if "-" in exported_name:
        prefix_free_path = video_dir / exported_name.split("-", 1)[1]
        if prefix_free_path.is_file():
            return prefix_free_path

    suffix_matches = [
        candidate
        for candidate in video_dir.iterdir()
        if candidate.is_file() and exported_name.endswith(candidate.name)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise FileNotFoundError(
        f"Could not uniquely match {exported_name!r} in {video_dir}"
    )


def output_columns() -> list[str]:
    columns = [
        "task_id",
        "video_id",
        "frame_index",
        "label_frame",
        "time_seconds",
        "action_label",
        "pose_detected",
        "bbox_detected",
        "people_detected",
        "boxes_detected",
        "detection_index",
        "person_index",
        "box_index",
        "frame_width_px",
        "frame_height_px",
        "bbox_confidence",
        "bbox_class_id",
        "bbox_x1_px",
        "bbox_y1_px",
        "bbox_x2_px",
        "bbox_y2_px",
    ]
    for joint_name in yolocfg.YOLO_KEYPOINT_NAMES:
        columns.extend(
            [
                f"{joint_name}_x_px",
                f"{joint_name}_y_px",
                f"{joint_name}_confidence",
            ]
        )
    return columns


def build_rows(
    *,
    task_id: int,
    video_id: str,
    frame_index: int,
    fps: float,
    action_label: str,
    frame_width: int,
    frame_height: int,
    people: list[dict],
    boxes: list[dict],
) -> list[dict]:
    base_row = {
        "task_id": task_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "label_frame": frame_index + 1,
        "time_seconds": frame_index / fps,
        "action_label": action_label,
        "pose_detected": 0,
        "bbox_detected": 0,
        "people_detected": len(people),
        "boxes_detected": len(boxes),
        "detection_index": "",
        "person_index": "",
        "box_index": "",
        "frame_width_px": frame_width,
        "frame_height_px": frame_height,
        "bbox_confidence": "",
        "bbox_class_id": "",
        "bbox_x1_px": "",
        "bbox_y1_px": "",
        "bbox_x2_px": "",
        "bbox_y2_px": "",
    }

    detection_count = max(len(people), len(boxes))
    if detection_count == 0:
        return [base_row]

    rows = []
    for detection_index in range(detection_count):
        row = dict(base_row)
        row["detection_index"] = detection_index

        if detection_index < len(boxes):
            box = boxes[detection_index]
            x1, y1, x2, y2 = box["xyxy"]
            row.update(
                {
                    "bbox_detected": 1,
                    "box_index": box["box_index"],
                    "bbox_confidence": box["confidence"],
                    "bbox_class_id": box["class_id"],
                    "bbox_x1_px": x1,
                    "bbox_y1_px": y1,
                    "bbox_x2_px": x2,
                    "bbox_y2_px": y2,
                }
            )

        if detection_index < len(people):
            person = people[detection_index]
            row["pose_detected"] = 1
            row["person_index"] = person["person_index"]
            for keypoint in person["keypoints"]:
                joint_name = keypoint["name"]
                row[f"{joint_name}_x_px"] = keypoint["x"]
                row[f"{joint_name}_y_px"] = keypoint["y"]
                row[f"{joint_name}_confidence"] = keypoint["confidence"]

        rows.append(row)

    return rows


def process_video(
    *,
    model,
    task: dict,
    video_path: Path,
    output_path: Path,
    max_frames: int | None,
) -> tuple[int, int]:
    import cv2

    frame_labels, final_labeled_frame = build_frame_labels(task)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = video_path.stem
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame_index = 0
    rows_written = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=output_columns())
            writer.writeheader()

            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break

                label_frame = frame_index + 1
                action_label = frame_labels.get(label_frame)
                if action_label is None:
                    raise ValueError(
                        f"{video_path.name} frame {label_frame} has no action label"
                    )

                results = model(frame, verbose=False)
                result = results[0]
                people = yoloutils.keypoints_to_people(result)
                boxes = yoloutils.boxes_to_detections(result)
                frame_height, frame_width = frame.shape[:2]

                frame_rows = build_rows(
                    task_id=int(task["id"]),
                    video_id=video_id,
                    frame_index=frame_index,
                    fps=fps,
                    action_label=action_label,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    people=people,
                    boxes=boxes,
                )
                writer.writerows(frame_rows)
                rows_written += len(frame_rows)

                frame_index += 1
                if frame_index % 100 == 0:
                    print(f"  {video_path.name}: {frame_index} frames")
                if max_frames is not None and frame_index >= max_frames:
                    break

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    if max_frames is None and frame_index != final_labeled_frame:
        difference = final_labeled_frame - frame_index
        print(
            f"  Warning: decoded {frame_index} frames but annotations end at "
            f"{final_labeled_frame} ({difference:+d} trailing labeled frames)."
        )
    if reported_frames > 0 and frame_index != reported_frames and max_frames is None:
        print(
            f"  Warning: OpenCV reported {reported_frames} frames but decoded "
            f"{frame_index}."
        )

    return frame_index, rows_written


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")

    from ultralytics import YOLO

    tasks = load_annotation_tasks(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    resolved_paths: set[Path] = set()
    for task in tasks:
        video_path = resolve_video_path(task["video"], args.video_dir)
        if video_path in resolved_paths:
            raise ValueError(f"Multiple tasks resolve to the same video: {video_path}")
        resolved_paths.add(video_path)

        output_path = args.output_dir / f"{video_path.stem}_joints_labels.csv"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_path}")
            continue

        print(f"Processing task {task['id']}: {video_path.name}")
        frames_written, rows_written = process_video(
            model=model,
            task=task,
            video_path=video_path,
            output_path=output_path,
            max_frames=args.max_frames,
        )
        print(
            f"  Saved {rows_written} rows from {frames_written} frames "
            f"to {output_path}"
        )


if __name__ == "__main__":
    main()
