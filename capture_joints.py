import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("models") / "yolo" / "yolov8n-pose.pt"
DEFAULT_SOURCE = Path("videos") / "Rodtang-taetat-2.mp4"
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def parse_source(source):
    if str(source).isdigit():
        return int(source), f"webcam-{source}"

    path = Path(source)
    return str(path), path.stem


def build_output_paths(output_dir, source_label, model_path):
    model_label = Path(model_path).stem
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = f"{slugify(source_label)}__{slugify(model_label)}__{run_id}"
    run_dir = Path(output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{slugify(source_label)}__{slugify(model_label)}"
    return {
        "run_dir": run_dir,
        "video": run_dir / f"{file_prefix}_annotated.mp4",
        "predictions": run_dir / f"{file_prefix}_predictions.jsonl",
    }


def keypoints_to_people(result):
    if result.keypoints is None:
        return []

    xy = result.keypoints.xy.cpu().numpy()
    conf = None
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf.cpu().numpy()

    people = []
    for person_index, person in enumerate(xy):
        keypoints = []
        for keypoint_index, coordinates in enumerate(person):
            keypoint_name = (
                KEYPOINT_NAMES[keypoint_index]
                if keypoint_index < len(KEYPOINT_NAMES)
                else f"keypoint_{keypoint_index}"
            )
            score = None if conf is None else float(conf[person_index][keypoint_index])
            keypoints.append(
                {
                    "name": keypoint_name,
                    "x": float(coordinates[0]),
                    "y": float(coordinates[1]),
                    "confidence": score,
                }
            )

        people.append(
            {
                "person_index": person_index,
                "keypoints": keypoints,
            }
        )

    return people


def boxes_to_detections(result):
    if result.boxes is None:
        return []

    boxes = []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
    cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

    for box_index, coordinates in enumerate(xyxy):
        boxes.append(
            {
                "box_index": box_index,
                "xyxy": [float(value) for value in coordinates],
                "confidence": float(conf[box_index]) if len(conf) > box_index else None,
                "class_id": int(cls[box_index]) if len(cls) > box_index else None,
            }
        )

    return boxes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Video path or webcam index.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder where predictions and video are saved.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    capture_source, source_label = parse_source(args.source)
    output_paths = build_output_paths(args.output, source_label, args.model)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    writer = None
    frame_index = 0

    with output_paths["predictions"].open("w", encoding="utf-8") as predictions_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            if writer is None:
                height, width = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_paths["video"]),
                    fourcc,
                    fps,
                    (width, height),
                )

            frame_predictions = {
                "frame_index": frame_index,
                "source": args.source,
                "model": args.model,
                "people": keypoints_to_people(results[0]),
                "detections": boxes_to_detections(results[0]),
            }
            predictions_file.write(json.dumps(frame_predictions) + "\n")
            writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow("AI Combat Coach", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Saved predictions to {output_paths['predictions']}")
    print(f"Saved annotated video to {output_paths['video']}")


if __name__ == "__main__":
    main()
