"""Shared real-time video inference for pose-based action classifiers."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from models import utils as modelutils
from models.action_detection.analytics import AnalyticsConfig, StrikeAnalytics
from models.action_detection.analytics.types import JointPoint
from models.action_detection.config import (
    CLASS_COLORS,
    SKELETON_EDGES,
    validate_task_labels,
)
from models.action_detection.preprocessing import (
    causal_windows,
    normalize_selected_frames,
    select_largest_person,
)
from models.yolo import config as yolocfg
from models.yolo import utils as yoloutils


ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_FPS = 30.0
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_RAW_VIDEO_DIR = ROOT_DIR / "media" / "videos" / "raw"


@dataclass(frozen=True)
class ActionModelRuntime:
    """Loaded classifier plus the preprocessing settings saved at training."""

    model_name: str
    classification_task: str
    class_names: tuple[str, ...]
    window_size: int
    confidence_threshold: float
    coordinate_clip: float
    predict_probabilities: Callable[[np.ndarray], np.ndarray]
    is_stub: bool = False


@dataclass(frozen=True)
class ActionPrediction:
    """
    One task model's prediction for the current frame.
    """

    classification_task: str
    model_name: str
    class_names: tuple[str, ...]
    class_name: str
    probability: float
    probabilities: np.ndarray
    is_stub: bool


def validate_model_bundle(
    bundle: dict,
    *,
    classification_task: str,
    source: Path,
) -> tuple[str, ...]:
    """
    Validate task metadata and classes in a saved model bundle.

    Usage: Inference only.

    Raises ValueError if the bundle's task or classes are invalid.
    """

    saved_task = bundle.get("classification_task")
    if saved_task is not None and str(saved_task) != classification_task:
        raise ValueError(
            f"{source} was trained for task {saved_task!r}, not "
            f"{classification_task!r}"
        )
    class_names = tuple(str(name) for name in bundle["class_names"])
    validate_task_labels(
        classification_task,
        class_names,
        source=str(source),
    )
    return class_names


def build_inference_parser(
    *,
    description: str,
    default_guard_weights: Path,
    default_striking_weights: Path,
) -> argparse.ArgumentParser:
    """
    Create the arguments shared by the TCN and LightGBM entry points.

    Usage: Inference only.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--guard-weights",
        type=Path,
        default=default_guard_weights,
        help="Trained guard-classifier bundle.",
    )
    parser.add_argument(
        "--striking-weights",
        type=Path,
        default=default_striking_weights,
        help="Trained striking-classifier bundle (required).",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=yolocfg.YOLO_WEIGHTS,
        help="Ultralytics YOLO pose weights.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video path or webcam index, for example 0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root folder for annotated video and JSONL predictions.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_VIDEO_DIR,
        help="Folder for unannotated webcam recordings.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the annotated inference stream; press q to stop.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional 30-FPS output-frame limit.",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.25,
        help="YOLO person-detection confidence threshold.",
    )
    parser.add_argument(
        "--yolo-device",
        help="Optional Ultralytics device, for example cpu, 0, or 0,1.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        help="Requested webcam capture height.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("count", "speed"),
        default=("count", "speed"),
        help=(
            "Strike metrics to compute. Defaults to both; pass only count or "
            "only speed when desired."
        ),
    )
    parser.add_argument(
        "--person-height-cm",
        type=float,
        default=175.0,
        help="Person height used for approximate physical speed (default: 175).",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate shared video-inference command-line arguments.

    Usage: Inference only.
    """
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    if not 0.0 <= args.yolo_confidence <= 1.0:
        raise ValueError("--yolo-confidence must be between 0 and 1")
    if args.camera_width is not None and args.camera_width < 1:
        raise ValueError("--camera-width must be positive")
    if args.camera_height is not None and args.camera_height < 1:
        raise ValueError("--camera-height must be positive")
    if not 50.0 <= args.person_height_cm <= 250.0:
        raise ValueError("--person-height-cm must be between 50 and 250")


def _open_writer(path: Path, frame: np.ndarray) -> cv2.VideoWriter:
    """
    Open a 30 FPS video writer using the dimensions of the first frame.

    Usage: Inference only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _webcam_raw_path(
    raw_output_dir: Path,
    source_label: str,
    model_name: str,
    run_dir: Path,
) -> Path:
    """
    Build the distinctive path for an unannotated webcam recording.

    Usage: Inference only.
    """
    run_timestamp = run_dir.name.rsplit("__", 1)[-1]
    filename = (
        f"{modelutils.slugify(source_label)}__"
        f"{modelutils.slugify(model_name)}__{run_timestamp}_raw.mp4"
    )
    return raw_output_dir / filename


def _iter_cfr_frames(
    capture: cv2.VideoCapture,
    *,
    webcam: bool,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Yield a 30-FPS constant-rate stream.

    Usage: Inference only.

    Files are time-resampled using their declared frame rate, including frame
    dropping or duplication. Webcam capture is requested and paced at 30 FPS.
    """

    if webcam:
        frame_index = 0
        frame_period = 1.0 / TARGET_FPS
        next_deadline = time.perf_counter()
        while capture.isOpened():
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

            success, frame = capture.read()
            if not success:
                break
            yield frame_index, frame_index / TARGET_FPS, frame
            frame_index += 1

            next_deadline += frame_period
            now = time.perf_counter()
            if next_deadline < now:
                next_deadline = now
        return

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise RuntimeError("The input video does not report a valid frame rate")

    source_frame_index = 0
    output_frame_index = 0
    last_frame: np.ndarray | None = None
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        last_frame = frame
        source_time = source_frame_index / source_fps
        while output_frame_index / TARGET_FPS <= source_time + 1e-9:
            yield (
                output_frame_index,
                output_frame_index / TARGET_FPS,
                frame.copy(),
            )
            output_frame_index += 1
        source_frame_index += 1

    # A decoded frame represents the interval until the following frame. For
    # low-FPS inputs, fill any remaining 30-FPS samples in the final interval
    # with the last decoded frame.
    source_duration = source_frame_index / source_fps
    while (
        last_frame is not None
        and output_frame_index / TARGET_FPS < source_duration - 1e-9
    ):
        yield (
            output_frame_index,
            output_frame_index / TARGET_FPS,
            last_frame.copy(),
        )
        output_frame_index += 1


def _raw_detection_rows(
    result,
    *,
    frame_index: int,
    frame: np.ndarray,
) -> pd.DataFrame:
    """
    Convert one YOLO result to the raw row shape used by preprocessing.

    Usage: Inference only.
    """

    people = yoloutils.keypoints_to_people(result)
    boxes = yoloutils.boxes_to_detections(result)
    height, width = frame.shape[:2]
    base_row: dict[str, object] = {
        "video_id": "inference",
        "frame_index": frame_index,
        "pose_detected": 0,
        "bbox_detected": 0,
        "people_detected": len(people),
        "boxes_detected": len(boxes),
        "detection_index": np.nan,
        "person_index": np.nan,
        "box_index": np.nan,
        "frame_width_px": width,
        "frame_height_px": height,
        "bbox_confidence": np.nan,
        "bbox_class_id": np.nan,
        "bbox_x1_px": np.nan,
        "bbox_y1_px": np.nan,
        "bbox_x2_px": np.nan,
        "bbox_y2_px": np.nan,
    }
    for joint_name in yolocfg.YOLO_KEYPOINT_NAMES:
        base_row[f"{joint_name}_x_px"] = np.nan
        base_row[f"{joint_name}_y_px"] = np.nan
        base_row[f"{joint_name}_confidence"] = np.nan

    detection_count = max(len(people), len(boxes))
    if detection_count == 0:
        return pd.DataFrame([base_row])

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
                if joint_name not in yolocfg.YOLO_KEYPOINT_NAMES:
                    continue
                row[f"{joint_name}_x_px"] = keypoint["x"]
                row[f"{joint_name}_y_px"] = keypoint["y"]
                row[f"{joint_name}_confidence"] = keypoint["confidence"]
        rows.append(row)

    return pd.DataFrame(rows)


def _select_pose(
    result,
    *,
    frame_index: int,
    frame: np.ndarray,
) -> pd.DataFrame:
    """
    Select the largest person from one YOLO result and return its raw row(s).

    Usage: Inference only.

    If no person is detected, return a single row with NaN keypoints.
    """
    raw_rows = _raw_detection_rows(
        result,
        frame_index=frame_index,
        frame=frame,
    )
    return select_largest_person(raw_rows).reset_index(drop=True)


def _pose_points(pose: pd.Series) -> dict[str, JointPoint]:
    """
    Extract raw keypoints from the selected-person row.

    Usage: Inference only.
    """

    if not int(pose.get("pose_detected", 0) or 0):
        return {}
    return {
        joint_name: JointPoint(
            x_px=float(pose.get(f"{joint_name}_x_px", np.nan)),
            y_px=float(pose.get(f"{joint_name}_y_px", np.nan)),
            confidence=float(
                pose.get(f"{joint_name}_confidence", np.nan)
            ),
        )
        for joint_name in yolocfg.YOLO_KEYPOINT_NAMES
    }


def draw_action_overlay(
    frame: np.ndarray,
    pose: pd.Series,
    *,
    predictions: Sequence[ActionPrediction],
    confidence_threshold: float,
    processing_fps: float,
    analytics_lines: Sequence[str] = (),
) -> np.ndarray:
    """
    Draw the selected pose and both task predictions on a BGR frame.

    Usage: Inference only.

    Returns a copy of the frame with the overlay.
    """

    output = frame.copy()
    height, width = output.shape[:2]
    active_prediction = next(
        (
            prediction
            for prediction in reversed(predictions)
            if prediction.class_name != "background"
        ),
        predictions[0],
    )
    color_rgb = CLASS_COLORS.get(
        active_prediction.class_name,
        CLASS_COLORS["background"],
    )
    color = tuple(int(channel) for channel in reversed(color_rgb))
    points: dict[str, tuple[int, int]] = {}

    if int(pose.get("pose_detected", 0) or 0):
        for joint_name in yolocfg.YOLO_KEYPOINT_NAMES:
            confidence = pose.get(f"{joint_name}_confidence", np.nan)
            x = pose.get(f"{joint_name}_x_px", np.nan)
            y = pose.get(f"{joint_name}_y_px", np.nan)
            if not (
                np.isfinite(confidence)
                and np.isfinite(x)
                and np.isfinite(y)
                and float(confidence) >= confidence_threshold
            ):
                continue
            points[joint_name] = (
                int(np.clip(round(float(x)), 0, width - 1)),
                int(np.clip(round(float(y)), 0, height - 1)),
            )

    for start_name, end_name in SKELETON_EDGES:
        if start_name in points and end_name in points:
            cv2.line(output, points[start_name], points[end_name], color, 3)
    for point in points.values():
        cv2.circle(output, point, 5, (255, 255, 255), -1)
        cv2.circle(output, point, 5, color, 2)

    header_height = 36 * len(predictions) + 34 + 26 * len(analytics_lines)
    cv2.rectangle(
        output,
        (0, 0),
        (width, min(height, header_height)),
        (20, 20, 20),
        -1,
    )
    for prediction_index, prediction in enumerate(predictions):
        prediction_color_rgb = CLASS_COLORS.get(
            prediction.class_name,
            (255, 200, 0),
        )
        prediction_color = tuple(
            int(channel) for channel in reversed(prediction_color_rgb)
        )
        stub_text = " [STUB]" if prediction.is_stub else ""
        label = (
            f"{prediction.classification_task.capitalize()}: "
            f"{prediction.class_name}  {prediction.probability:.1%}"
            f"{stub_text}"
        )
        cv2.putText(
            output,
            label,
            (16, 30 + prediction_index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            prediction_color,
            2,
            cv2.LINE_AA,
        )

    analytics_start_y = 30 + 36 * len(predictions)
    for line_index, line in enumerate(analytics_lines):
        cv2.putText(
            output,
            line,
            (16, analytics_start_y + line_index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    status = (
        f"processing {processing_fps:.1f} FPS | "
        f"people {int(pose.get('people_detected', 0) or 0)}"
    )
    cv2.putText(
        output,
        status,
        (16, header_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return output


def run_action_inference(
    args: argparse.Namespace,
    runtimes: Sequence[ActionModelRuntime],
) -> None:
    """
    Run one YOLO pose pass and both action classifiers on every frame.

    Usage: Inference only.

    Writes annotated video and JSONL predictions to disk, optionally displaying the annotated stream in real time. 

    Raises ValueError if the arguments or runtimes are invalid.
    Raises RuntimeError if the input source cannot be opened.
    """

    _validate_arguments(args)
    runtimes = tuple(runtimes)
    runtime_tasks = tuple(runtime.classification_task for runtime in runtimes)
    if set(runtime_tasks) != {"guard", "striking"} or len(runtimes) != 2:
        raise ValueError(
            "Inference requires exactly one guard and one striking runtime"
        )
    for runtime in runtimes:
        if runtime.window_size < 1:
            raise ValueError(
                f"{runtime.model_name} window_size must be at least 1"
            )
        validate_task_labels(
            runtime.classification_task,
            runtime.class_names,
            source=runtime.model_name,
        )

    capture_source, source_label = modelutils.parse_source(args.source)
    webcam = isinstance(capture_source, int)
    model_label = "+".join(
        f"{runtime.model_name}-{runtime.classification_task}"
        for runtime in runtimes
    )
    output_paths = modelutils.build_output_paths(
        args.output,
        source_label,
        model_label,
    )
    raw_video_path = (
        _webcam_raw_path(
            args.raw_output,
            source_label,
            model_label,
            output_paths["run_dir"],
        )
        if webcam
        else None
    )
    output_prefix = output_paths["predictions"].name.removesuffix(
        "_predictions.jsonl"
    )
    analytics_events_path = (
        output_paths["run_dir"] / f"{output_prefix}_events.csv"
    )
    analytics_summary_path = (
        output_paths["run_dir"] / f"{output_prefix}_summary.json"
    )

    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input source: {args.source}")
    if webcam:
        capture.set(cv2.CAP_PROP_FPS, TARGET_FPS) # Request 30 FPS capture from the webcam
        if args.camera_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if reported_fps > 0 and abs(reported_fps - TARGET_FPS) > 0.5:
            print(
                f"Warning: webcam reports {reported_fps:.2f} FPS after "
                f"requesting {TARGET_FPS:.0f}; output is still written as "
                "30-FPS CFR."
            )

    pose_model = YOLO(str(args.pose_model))
    histories = {
        runtime.classification_task: deque(maxlen=runtime.window_size)
        for runtime in runtimes
    }
    analytics = StrikeAnalytics(
        AnalyticsConfig(
            enabled_metrics=tuple(args.metrics),
            person_height_m=args.person_height_cm / 100.0,
            keypoint_confidence=min(
                runtime.confidence_threshold for runtime in runtimes
            ),
        )
    )
    annotated_writer: cv2.VideoWriter | None = None
    raw_writer: cv2.VideoWriter | None = None
    processed_frames = 0
    started_at = time.perf_counter()
    display_started_at = started_at

    try:
        with output_paths["predictions"].open(
            "w", encoding="utf-8"
        ) as predictions_file:
            for frame_index, timestamp, frame in _iter_cfr_frames(
                capture,
                webcam=webcam,
            ):
                if raw_video_path is not None:
                    if raw_writer is None:
                        raw_writer = _open_writer(raw_video_path, frame)
                    raw_writer.write(frame)

                inference_started = time.perf_counter()
                yolo_arguments: dict[str, object] = {
                    "verbose": False,
                    "conf": args.yolo_confidence,
                }
                if args.yolo_device:
                    yolo_arguments["device"] = args.yolo_device
                result = pose_model(frame, **yolo_arguments)[0]
                selected_pose = _select_pose(
                    result,
                    frame_index=frame_index,
                    frame=frame,
                )
                pose = selected_pose.iloc[0]
                frame_predictions = []
                for runtime in runtimes:
                    current_features = normalize_selected_frames(
                        selected_pose,
                        confidence_threshold=runtime.confidence_threshold,
                        coordinate_clip=runtime.coordinate_clip,
                    )[0]
                    history = histories[runtime.classification_task]
                    history.append(current_features)
                    history_features = np.stack(tuple(history)).astype(
                        np.float32,
                        copy=False,
                    )
                    window = causal_windows(
                        history_features,
                        runtime.window_size,
                    )[-1]
                    probabilities = np.asarray(
                        runtime.predict_probabilities(window),
                        dtype=np.float32,
                    ).reshape(-1)
                    if probabilities.shape != (len(runtime.class_names),):
                        raise ValueError(
                            f"{runtime.model_name} returned "
                            f"{probabilities.shape} probabilities for "
                            f"{len(runtime.class_names)} classes"
                        )
                    if not np.all(np.isfinite(probabilities)):
                        raise ValueError(
                            f"{runtime.model_name} returned non-finite "
                            "probabilities"
                        )
                    class_index = int(np.argmax(probabilities))
                    frame_predictions.append(
                        ActionPrediction(
                            classification_task=runtime.classification_task,
                            model_name=runtime.model_name,
                            class_names=runtime.class_names,
                            class_name=runtime.class_names[class_index],
                            probability=float(probabilities[class_index]),
                            probabilities=probabilities,
                            is_stub=runtime.is_stub,
                        )
                    )

                striking_prediction = next(
                    prediction
                    for prediction in frame_predictions
                    if prediction.classification_task == "striking"
                )
                analytics_snapshot = analytics.update(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    pose_points=_pose_points(pose),
                    striking_probabilities={
                        class_name: float(
                            striking_prediction.probabilities[class_index]
                        )
                        for class_index, class_name in enumerate(
                            striking_prediction.class_names
                        )
                    },
                )
                elapsed = max(time.perf_counter() - inference_started, 1e-9)
                annotated_frame = draw_action_overlay(
                    frame,
                    pose,
                    predictions=frame_predictions,
                    confidence_threshold=min(
                        runtime.confidence_threshold for runtime in runtimes
                    ),
                    processing_fps=1.0 / elapsed,
                    analytics_lines=analytics.overlay_lines(
                        analytics_snapshot
                    ),
                )

                if annotated_writer is None:
                    annotated_writer = _open_writer(
                        output_paths["video"],
                        annotated_frame,
                    )
                annotated_writer.write(annotated_frame)
                predictions_file.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "time_seconds": timestamp,
                            "predictions": {
                                prediction.classification_task: {
                                    "model_name": prediction.model_name,
                                    "is_stub": prediction.is_stub,
                                    "class_name": prediction.class_name,
                                    "confidence": prediction.probability,
                                    "class_probabilities": {
                                        name: float(
                                            prediction.probabilities[index]
                                        )
                                        for index, name in enumerate(
                                            prediction.class_names
                                        )
                                    },
                                }
                                for prediction in frame_predictions
                            },
                            "pose_detected": bool(pose["pose_detected"]),
                            "people_detected": int(pose["people_detected"]),
                            "selected_detection_index": (
                                None
                                if pd.isna(pose["detection_index"])
                                else int(pose["detection_index"])
                            ),
                            "analytics": analytics_snapshot.as_record(),
                        }
                    )
                    + "\n"
                )

                processed_frames += 1
                if args.display:
                    if not webcam:
                        display_delay = (
                            display_started_at
                            + timestamp
                            - time.perf_counter()
                        )
                        if display_delay > 0:
                            time.sleep(display_delay)
                    cv2.imshow(
                        "Muay-ThAI - guard + striking",
                        annotated_frame,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if (
                    args.max_frames is not None
                    and processed_frames >= args.max_frames
                ):
                    break
        analytics.finalize()
    finally:
        capture.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if raw_writer is not None:
            raw_writer.release()
        cv2.destroyAllWindows()

    analytics.write_outputs(
        analytics_events_path,
        analytics_summary_path,
    )

    duration = max(time.perf_counter() - started_at, 1e-9)
    print(
        f"Processed {processed_frames:,} CFR frames at "
        f"{processed_frames / duration:.1f} average FPS"
    )
    print(f"Saved predictions to {output_paths['predictions']}")
    print(f"Saved strike events to {analytics_events_path}")
    print(f"Saved analytics summary to {analytics_summary_path}")
    if annotated_writer is not None:
        print(f"Saved annotated video to {output_paths['video']}")
    if raw_video_path is not None and raw_writer is not None:
        print(f"Saved raw webcam video to {raw_video_path}")
