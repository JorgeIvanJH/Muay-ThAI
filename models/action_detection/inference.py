"""Shared concurrent video inference for pose-based action classifiers."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from models import utils as modelutils
from models.action_detection.analytics import AnalyticsConfig, StrikeAnalytics
from models.action_detection.config import (
    CLASS_COLORS,
    SKELETON_EDGES,
    validate_task_labels,
)
from models.action_detection.realtime.actions import (
    ActionModelRuntime,
    ActionPrediction,
    DualActionPredictor,
)
from models.action_detection.realtime.display import (
    InferenceWindows,
    close_inference_windows,
    overlay_metrics_panel,
    render_metrics_panel,
)
from models.action_detection.realtime.output import AsyncVideoWriter
from models.action_detection.realtime.pose import SelectedPose, pose_points
from models.action_detection.realtime.telemetry import (
    PerformanceSnapshot,
    PipelineTelemetry,
)
from models.action_detection.realtime.types import END_OF_STREAM, PosePacket
from models.action_detection.realtime.workers import (
    run_capture_worker,
    run_pose_worker,
)
from models.yolo import config as yolocfg


ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_FPS = 30.0
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_RAW_VIDEO_DIR = ROOT_DIR / "media" / "videos" / "raw"


@dataclass(frozen=True)
class _OutputResources:
    """Paths and optional asynchronous writers owned by one inference run."""

    output_paths: dict[str, Path]
    analytics_events_path: Path
    analytics_summary_path: Path
    raw_video_path: Path
    annotated_writer: AsyncVideoWriter | None
    raw_writer: AsyncVideoWriter | None


@dataclass(frozen=True)
class _WorkerResources:
    """Queues, stop state and threads forming the capture/pose pipeline."""

    pose_queue: queue.Queue
    error_queue: queue.Queue[BaseException]
    stop_event: threading.Event
    capture_thread: threading.Thread
    pose_thread: threading.Thread


@dataclass(frozen=True)
class _EvaluatedFrame:
    """Ordered action/analytics result ready for display and persistence."""

    predictions: tuple[ActionPrediction, ...]
    analytics_snapshot: object
    analytics_lines: tuple[str, ...]
    performance: PerformanceSnapshot
    annotated_frame: np.ndarray | None


def validate_model_bundle(
    bundle: dict,
    *,
    classification_task: str,
    source: Path,
) -> tuple[str, ...]:
    """Validate task metadata and classes in a saved model bundle.

    Usage: Inference only.
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
    """Create arguments shared by the TCN and LightGBM entry points.

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
        help="Trained striking-classifier bundle.",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=yolocfg.YOLO_REALTIME_WEIGHTS,
        help="Ultralytics YOLO pose weights.",
    )
    parser.add_argument(
        "--pose-imgsz",
        type=int,
        default=640,
        help="YOLO pose inference size (default: 640).",
    )
    parser.add_argument(
        "--pose-precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="YOLO arithmetic precision; validate accuracy before using fp16.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Explicit video path or webcam index, for example 0.",
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
        "--no-save-annotated",
        action="store_true",
        help="Do not encode the annotated output video.",
    )
    parser.add_argument(
        "--no-save-raw",
        action="store_true",
        help="Do not save an unannotated webcam recording.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional processed-frame limit.",
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
        "--frame-queue-size",
        type=int,
        default=4,
        help="Bounded capture/pose queue capacity (default: 4).",
    )
    parser.add_argument(
        "--output-queue-size",
        type=int,
        default=12,
        help="Bounded queue capacity for each video encoder (default: 12).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("count", "speed"),
        default=("count", "speed"),
        help="Strike metrics to compute; defaults to count and speed.",
    )
    parser.add_argument(
        "--person-height-cm",
        type=float,
        default=175.0,
        help="Person height used for approximate physical speed.",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    """Validate shared video-inference command-line arguments.

    Usage: Inference only.
    """

    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    if not 0.0 <= args.yolo_confidence <= 1.0:
        raise ValueError("--yolo-confidence must be between 0 and 1")
    if args.pose_imgsz < 32:
        raise ValueError("--pose-imgsz must be at least 32")
    if args.frame_queue_size < 1:
        raise ValueError("--frame-queue-size must be at least 1")
    if args.output_queue_size < 1:
        raise ValueError("--output-queue-size must be at least 1")
    if args.camera_width is not None and args.camera_width < 1:
        raise ValueError("--camera-width must be positive")
    if args.camera_height is not None and args.camera_height < 1:
        raise ValueError("--camera-height must be positive")
    if not 50.0 <= args.person_height_cm <= 250.0:
        raise ValueError("--person-height-cm must be between 50 and 250")


def _webcam_raw_path(
    raw_output_dir: Path,
    source_label: str,
    model_name: str,
    run_dir: Path,
) -> Path:
    """Build the distinctive path for an unannotated webcam recording.

    Usage: Inference only.
    """

    run_timestamp = run_dir.name.rsplit("__", 1)[-1]
    filename = (
        f"{modelutils.slugify(source_label)}__"
        f"{modelutils.slugify(model_name)}__{run_timestamp}_raw.mp4"
    )
    return raw_output_dir / filename


def _configure_capture(
    args: argparse.Namespace,
) -> tuple[cv2.VideoCapture, bool, str]:
    """Open the requested source and configure low-latency webcam capture.

    Usage: Inference only.
    """

    capture_source, source_label = modelutils.parse_source(args.source)
    webcam = isinstance(capture_source, int)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input source: {args.source}")
    if not webcam:
        return capture, webcam, source_label

    # Camera properties are requests to the backend. Runtime telemetry reports
    # measured capture FPS rather than trusting these nominal values.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if args.camera_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Webcam reports {reported_width}x{reported_height} at "
        f"{reported_fps:.2f} nominal FPS"
    )
    return capture, webcam, source_label


def _pose_predict_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Translate stable CLI options into Ultralytics prediction arguments.

    Usage: Inference only.
    """

    arguments: dict[str, object] = {
        "verbose": False,
        "conf": args.yolo_confidence,
        "imgsz": args.pose_imgsz,
    }
    if args.yolo_device:
        arguments["device"] = args.yolo_device
    if args.pose_precision == "fp16":
        arguments["quantize"] = 16
    return arguments


def _warm_up_pose_model(
    pose_model: YOLO,
    capture: cv2.VideoCapture,
    predict_arguments: dict[str, object],
) -> None:
    """Run two blank frames before capture timing starts.

    Usage: Inference only.

    Warm-up pays predictor construction and kernel initialization before the
    real-time queues begin collecting latency measurements.
    """

    width = max(32, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640)
    height = max(32, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    pose_model(blank, **predict_arguments)
    pose_model(blank, **predict_arguments)


def _analytics_paths(output_paths: dict[str, Path]) -> tuple[Path, Path]:
    """Build event CSV and summary JSON paths beside frame predictions.

    Usage: Inference only.
    """

    prefix = output_paths["predictions"].name.removesuffix(
        "_predictions.jsonl"
    )
    return (
        output_paths["run_dir"] / f"{prefix}_events.csv",
        output_paths["run_dir"] / f"{prefix}_summary.json",
    )


def draw_action_overlay(
    frame: np.ndarray,
    pose: SelectedPose,
    *,
    predictions: Sequence[ActionPrediction],
    confidence_threshold: float,
) -> np.ndarray:
    """Draw only the selected pose skeleton over a video frame.

    Usage: Inference only.
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
    if pose.pose_detected:
        for joint_index, joint_name in enumerate(yolocfg.YOLO_KEYPOINT_NAMES):
            confidence = pose.keypoint_confidences[joint_index]
            x, y = pose.keypoints_xy[joint_index]
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

    return output


def _striking_probabilities(
    predictions: Sequence[ActionPrediction],
) -> dict[str, float]:
    """Extract the striking probability mapping consumed by analytics.

    Usage: Inference only.
    """

    striking = next(
        prediction
        for prediction in predictions
        if prediction.classification_task == "striking"
    )
    return {
        name: float(striking.probabilities[index])
        for index, name in enumerate(striking.class_names)
    }


def _prediction_document(
    packet: PosePacket,
    predictions: Sequence[ActionPrediction],
    analytics_snapshot,
    performance: PerformanceSnapshot,
) -> dict[str, object]:
    """Build one JSON-safe persisted result for a completed pose frame.

    Usage: Inference only.
    """

    pose = packet.pose
    return {
        "frame_index": packet.frame.frame_index,
        "time_seconds": packet.frame.timestamp,
        "predictions": {
            prediction.classification_task: {
                "model_name": prediction.model_name,
                "class_name": prediction.class_name,
                "confidence": prediction.probability,
                "class_probabilities": {
                    name: float(prediction.probabilities[index])
                    for index, name in enumerate(prediction.class_names)
                },
            }
            for prediction in predictions
        },
        "pose_detected": pose.pose_detected,
        "people_detected": pose.people_detected,
        "selected_detection_index": pose.detection_index,
        "analytics": analytics_snapshot.as_record(),
        "performance": {
            "capture_fps": performance.capture_fps,
            "pose_fps": performance.pose_fps,
            "action_fps": performance.action_fps,
            "pose_p95_ms": performance.pose_p95_ms,
            "action_p95_ms": performance.action_p95_ms,
            "end_to_end_p95_ms": performance.end_to_end_p95_ms,
            "dropped_frames": performance.dropped_frames,
        },
    }


def _raise_worker_error(error_queue: queue.Queue[BaseException]) -> None:
    """Raise the first pending background-worker failure.

    Usage: Inference only.
    """

    try:
        error = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError("Real-time inference worker failed") from error


def _create_outputs(
    args: argparse.Namespace,
    *,
    webcam: bool,
    source_label: str,
    model_label: str,
    telemetry: PipelineTelemetry,
) -> _OutputResources:
    """Create timestamped paths and optional asynchronous video writers.

    Usage: Inference only.
    """

    output_paths = modelutils.build_output_paths(
        args.output,
        source_label,
        model_label,
    )
    events_path, summary_path = _analytics_paths(output_paths)
    raw_video_path = _webcam_raw_path(
        args.raw_output,
        source_label,
        model_label,
        output_paths["run_dir"],
    )
    annotated_writer = None
    if not args.no_save_annotated:
        annotated_writer = AsyncVideoWriter(
            output_paths["video"],
            fps=TARGET_FPS,
            queue_size=args.output_queue_size,
            telemetry=telemetry,
            stage_name="annotated_output",
            drop_when_full=webcam,
        )
    raw_writer = None
    if webcam and not args.no_save_raw:
        raw_writer = AsyncVideoWriter(
            raw_video_path,
            fps=TARGET_FPS,
            queue_size=args.output_queue_size,
            telemetry=telemetry,
            stage_name="raw_output",
            drop_when_full=True,
        )
    return _OutputResources(
        output_paths=output_paths,
        analytics_events_path=events_path,
        analytics_summary_path=summary_path,
        raw_video_path=raw_video_path,
        annotated_writer=annotated_writer,
        raw_writer=raw_writer,
    )


def _start_workers(
    capture: cv2.VideoCapture,
    pose_model: YOLO,
    *,
    args: argparse.Namespace,
    webcam: bool,
    predict_arguments: dict[str, object],
    telemetry: PipelineTelemetry,
    raw_writer: AsyncVideoWriter | None,
) -> _WorkerResources:
    """Create and start the capture and pose-inference threads.

    Usage: Inference only.
    """

    frame_queue: queue.Queue = queue.Queue(maxsize=args.frame_queue_size)
    pose_queue: queue.Queue = queue.Queue(maxsize=args.frame_queue_size)
    error_queue: queue.Queue[BaseException] = queue.Queue()
    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=run_capture_worker,
        kwargs={
            "capture": capture,
            "webcam": webcam,
            "target_fps": TARGET_FPS,
            "output_queue": frame_queue,
            "stop_event": stop_event,
            "error_queue": error_queue,
            "telemetry": telemetry,
            "raw_frame_callback": (
                None if raw_writer is None else raw_writer.submit
            ),
        },
        name="frame-capture",
        daemon=True,
    )
    pose_thread = threading.Thread(
        target=run_pose_worker,
        kwargs={
            "pose_model": pose_model,
            "input_queue": frame_queue,
            "output_queue": pose_queue,
            "stop_event": stop_event,
            "error_queue": error_queue,
            "telemetry": telemetry,
            "predict_arguments": predict_arguments,
            "webcam": webcam,
        },
        name="pose-inference",
        daemon=True,
    )
    resources = _WorkerResources(
        pose_queue=pose_queue,
        error_queue=error_queue,
        stop_event=stop_event,
        capture_thread=capture_thread,
        pose_thread=pose_thread,
    )
    try:
        pose_thread.start()
        capture_thread.start()
    except BaseException:
        stop_event.set()
        capture_thread.join(timeout=1.0)
        pose_thread.join(timeout=1.0)
        raise
    return resources


def _evaluate_pose_packet(
    packet: PosePacket,
    *,
    action_predictor: DualActionPredictor,
    analytics: StrikeAnalytics,
    telemetry: PipelineTelemetry,
    confidence_threshold: float,
    render_output: bool,
) -> _EvaluatedFrame:
    """Run both classifiers and ordered analytics for one pose packet.

    Usage: Inference only.
    """

    action_started = time.perf_counter()
    predictions = action_predictor.predict(packet.pose)
    analytics_snapshot = analytics.update(
        frame_index=packet.frame.frame_index,
        timestamp=packet.frame.timestamp,
        pose_points=pose_points(packet.pose),
        striking_probabilities=_striking_probabilities(predictions),
    )
    action_finished = time.perf_counter()
    telemetry.record_latency("action", action_finished - action_started)
    telemetry.record_event("action", action_finished)
    telemetry.record_latency(
        "end_to_end",
        action_finished - packet.frame.captured_at,
    )
    performance = telemetry.snapshot()
    analytics_lines = analytics.overlay_lines(analytics_snapshot)
    annotated_frame = None
    if render_output:
        annotated_frame = draw_action_overlay(
            packet.frame.frame,
            packet.pose,
            predictions=predictions,
            confidence_threshold=confidence_threshold,
        )
    return _EvaluatedFrame(
        predictions=predictions,
        analytics_snapshot=analytics_snapshot,
        analytics_lines=analytics_lines,
        performance=performance,
        annotated_frame=annotated_frame,
    )


def _pace_file_display(
    *,
    webcam: bool,
    source_timestamp: float,
    display_started_at: float,
) -> None:
    """Delay offline display until the corresponding source timestamp.

    Usage: Inference only.
    """

    if webcam:
        return
    delay = display_started_at + source_timestamp - time.perf_counter()
    if delay > 0.0:
        time.sleep(delay)


def _consume_pose_stream(
    args: argparse.Namespace,
    *,
    webcam: bool,
    workers: _WorkerResources,
    outputs: _OutputResources,
    action_predictor: DualActionPredictor,
    analytics: StrikeAnalytics,
    telemetry: PipelineTelemetry,
) -> tuple[int, float]:
    """Consume ordered pose packets until source end, limit, or user stop.

    Usage: Inference only.
    """

    processed_frames = 0
    display_started_at = time.perf_counter()
    display = InferenceWindows() if args.display else None
    confidence_threshold = min(
        runtime.confidence_threshold
        for runtime in action_predictor.runtimes
    )
    render_output = args.display or outputs.annotated_writer is not None
    with outputs.output_paths["predictions"].open(
        "w",
        encoding="utf-8",
    ) as predictions_file:
        while True:
            _raise_worker_error(workers.error_queue)
            try:
                item = workers.pose_queue.get(timeout=0.1)
            except queue.Empty:
                if (
                    not workers.capture_thread.is_alive()
                    and not workers.pose_thread.is_alive()
                ):
                    break
                continue
            if item is END_OF_STREAM:
                break
            if not isinstance(item, PosePacket):
                raise TypeError(f"Unexpected pose queue item: {type(item)}")

            evaluated = _evaluate_pose_packet(
                item,
                action_predictor=action_predictor,
                analytics=analytics,
                telemetry=telemetry,
                confidence_threshold=confidence_threshold,
                render_output=render_output,
            )
            metrics_panel = None
            if evaluated.annotated_frame is not None:
                metrics_panel = render_metrics_panel(
                    evaluated.predictions,
                    evaluated.analytics_lines,
                    evaluated.performance,
                )
            if (
                outputs.annotated_writer is not None
                and evaluated.annotated_frame is not None
                and metrics_panel is not None
            ):
                outputs.annotated_writer.submit(
                    overlay_metrics_panel(
                        evaluated.annotated_frame,
                        metrics_panel,
                    )
                )
            predictions_file.write(
                json.dumps(
                    _prediction_document(
                        item,
                        evaluated.predictions,
                        evaluated.analytics_snapshot,
                        evaluated.performance,
                    )
                )
                + "\n"
            )
            processed_frames += 1

            if (
                display is not None
                and evaluated.annotated_frame is not None
                and metrics_panel is not None
            ):
                _pace_file_display(
                    webcam=webcam,
                    source_timestamp=item.frame.timestamp,
                    display_started_at=display_started_at,
                )
                if display.show(
                    evaluated.annotated_frame,
                    metrics_panel,
                ):
                    workers.stop_event.set()
                    break
            if (
                args.max_frames is not None
                and processed_frames >= args.max_frames
            ):
                workers.stop_event.set()
                break
    _raise_worker_error(workers.error_queue)
    return processed_frames, time.perf_counter()


def _close_output_writers(outputs: _OutputResources | None) -> None:
    """Drain all active output writers.

    Usage: Inference only.
    """

    if outputs is None:
        return
    errors = []
    for writer in (outputs.annotated_writer, outputs.raw_writer):
        if writer is None:
            continue
        try:
            writer.close()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise errors[0]


def _stop_pipeline(
    capture: cv2.VideoCapture,
    workers: _WorkerResources | None,
    outputs: _OutputResources | None,
) -> None:
    """Stop workers, release capture, drain encoders and close GUI state.

    Usage: Inference only.
    """

    if workers is not None:
        workers.stop_event.set()
    capture.release()
    if workers is not None:
        workers.capture_thread.join(timeout=5.0)
        workers.pose_thread.join(timeout=5.0)
    try:
        _close_output_writers(outputs)
    finally:
        close_inference_windows()


def _report_run(
    *,
    processed_frames: int,
    active_duration: float,
    telemetry: PipelineTelemetry,
    outputs: _OutputResources,
) -> None:
    """Print final throughput, drop counts and generated artifact paths.

    Usage: Inference only.
    """

    performance = telemetry.snapshot()
    print(
        f"Processed {processed_frames:,} frames at "
        f"{processed_frames / max(active_duration, 1e-9):.1f} end-to-end "
        "FPS including source startup"
    )
    print(performance.overlay_text())
    if telemetry.drop_counts():
        print(f"Dropped items by stage: {telemetry.drop_counts()}")
    print(f"Saved predictions to {outputs.output_paths['predictions']}")
    print(f"Saved strike events to {outputs.analytics_events_path}")
    print(f"Saved analytics summary to {outputs.analytics_summary_path}")
    if outputs.annotated_writer is not None:
        print(
            f"Saved {outputs.annotated_writer.written_frames:,} annotated "
            f"frames to {outputs.output_paths['video']}"
        )
    if outputs.raw_writer is not None:
        print(
            f"Saved {outputs.raw_writer.written_frames:,} raw webcam frames "
            f"to {outputs.raw_video_path}"
        )


def run_action_inference(
    args: argparse.Namespace,
    runtimes: Sequence[ActionModelRuntime],
) -> None:
    """Run concurrent capture, pose, dual-action and analytics inference.

    Usage: Inference only.

    Setup, ordered consumption, shutdown and reporting are delegated to focused
    helpers so changes to one pipeline concern do not enlarge this coordinator.
    """

    _validate_arguments(args)
    action_predictor = DualActionPredictor(runtimes)
    capture, webcam, source_label = _configure_capture(args)
    model_label = "+".join(
        f"{runtime.model_name}-{runtime.classification_task}"
        for runtime in action_predictor.runtimes
    )
    predict_arguments = _pose_predict_arguments(args)
    outputs: _OutputResources | None = None
    workers: _WorkerResources | None = None
    try:
        pose_model = YOLO(str(args.pose_model))
        _warm_up_pose_model(pose_model, capture, predict_arguments)
        telemetry = PipelineTelemetry()
        analytics = StrikeAnalytics(
            AnalyticsConfig(
                enabled_metrics=tuple(args.metrics),
                person_height_m=args.person_height_cm / 100.0,
                keypoint_confidence=min(
                    runtime.confidence_threshold
                    for runtime in action_predictor.runtimes
                ),
            )
        )
        outputs = _create_outputs(
            args,
            webcam=webcam,
            source_label=source_label,
            model_label=model_label,
            telemetry=telemetry,
        )
        workers = _start_workers(
            capture,
            pose_model,
            args=args,
            webcam=webcam,
            predict_arguments=predict_arguments,
            telemetry=telemetry,
            raw_writer=outputs.raw_writer,
        )
        started_at = time.perf_counter()
        processed_frames, inference_finished_at = _consume_pose_stream(
            args,
            webcam=webcam,
            workers=workers,
            outputs=outputs,
            action_predictor=action_predictor,
            analytics=analytics,
            telemetry=telemetry,
        )
        analytics.finalize()
    finally:
        _stop_pipeline(capture, workers, outputs)

    if outputs is None:
        raise RuntimeError("Inference outputs were not initialized")
    analytics.write_outputs(
        outputs.analytics_events_path,
        outputs.analytics_summary_path,
    )
    _report_run(
        processed_frames=processed_frames,
        active_duration=inference_finished_at - started_at,
        telemetry=telemetry,
        outputs=outputs,
    )
