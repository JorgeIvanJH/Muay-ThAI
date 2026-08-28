"""Thread workers that isolate camera I/O from GPU pose inference."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Mapping

import cv2

from models.action_detection.realtime.capture import produce_frames
from models.action_detection.realtime.pose import extract_largest_pose
from models.action_detection.realtime.queues import put_latest, put_ordered
from models.action_detection.realtime.telemetry import PipelineTelemetry
from models.action_detection.realtime.types import (
    END_OF_STREAM,
    FramePacket,
    PosePacket,
)


def run_pose_worker(
    pose_model,
    *,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    error_queue: queue.Queue[BaseException],
    telemetry: PipelineTelemetry,
    predict_arguments: Mapping[str, object],
    webcam: bool,
) -> None:
    """Run YOLO on queued frames and emit compact selected-pose packets.

    Usage: Inference only.

    This is the sole owner of the YOLO model. Keeping one GPU worker avoids
    concurrent access to the same predictor while capture and CPU analytics
    continue independently.
    """

    try:
        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is END_OF_STREAM:
                break
            if not isinstance(item, FramePacket):
                raise TypeError(f"Unexpected frame queue item: {type(item)}")

            started_at = time.perf_counter()
            result = pose_model(item.frame, **predict_arguments)[0]
            pose = extract_largest_pose(result, item.frame.shape)
            telemetry.record_latency("pose", time.perf_counter() - started_at)
            telemetry.record_event("pose")
            packet = PosePacket(frame=item, pose=pose)
            if webcam:
                dropped = put_latest(output_queue, packet)
                telemetry.record_drop("pose_queue", dropped)
            elif not put_ordered(output_queue, packet, stop_event):
                break
    except BaseException as error:
        error_queue.put(error)
        stop_event.set()
    finally:
        if webcam or stop_event.is_set():
            put_latest(output_queue, END_OF_STREAM)
        else:
            output_queue.put(END_OF_STREAM)


def run_capture_worker(
    capture: cv2.VideoCapture,
    *,
    webcam: bool,
    target_fps: float,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    error_queue: queue.Queue[BaseException],
    telemetry: PipelineTelemetry,
    raw_frame_callback=None,
) -> None:
    """Run frame production and forward failures to the coordinator.

    Usage: Inference only.
    """

    try:
        produce_frames(
            capture,
            webcam=webcam,
            target_fps=target_fps,
            output_queue=output_queue,
            stop_event=stop_event,
            telemetry=telemetry,
            raw_frame_callback=raw_frame_callback,
        )
    except BaseException as error:
        error_queue.put(error)
        stop_event.set()
        put_latest(output_queue, END_OF_STREAM)
