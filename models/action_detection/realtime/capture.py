"""Truthfully timestamped webcam and constant-rate file frame sources."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from models.action_detection.realtime.queues import put_latest, put_ordered
from models.action_detection.realtime.telemetry import PipelineTelemetry
from models.action_detection.realtime.types import END_OF_STREAM, FramePacket


def iter_webcam_frames(capture: cv2.VideoCapture) -> Iterator[FramePacket]:
    """Yield webcam frames with measured monotonic timestamps.

    Usage: Inference only.

    The camera driver controls arrival cadence. No synthetic 30-FPS timestamp
    is assigned, which keeps speed analytics correct when capture is irregular.
    """

    first_capture_time: float | None = None
    frame_index = 0
    while capture.isOpened():
        success, frame = capture.read()
        captured_at = time.perf_counter()
        if not success:
            break
        if first_capture_time is None:
            first_capture_time = captured_at
        yield FramePacket(
            frame_index=frame_index,
            timestamp=captured_at - first_capture_time,
            captured_at=captured_at,
            frame=frame,
        )
        frame_index += 1


def iter_file_cfr_frames(
    capture: cv2.VideoCapture,
    target_fps: float,
) -> Iterator[FramePacket]:
    """Resample a decoded file onto a constant target-FPS timeline.

    Usage: Inference only.

    Frames are dropped or duplicated when the declared source FPS differs from
    the model timeline. File mode preserves every target slot and may process
    more slowly than wall-clock playback.
    """

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
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
        while output_frame_index / target_fps <= source_time + 1e-9:
            yield FramePacket(
                frame_index=output_frame_index,
                timestamp=output_frame_index / target_fps,
                captured_at=time.perf_counter(),
                frame=frame.copy(),
            )
            output_frame_index += 1
        source_frame_index += 1

    source_duration = source_frame_index / source_fps
    while (
        last_frame is not None
        and output_frame_index / target_fps < source_duration - 1e-9
    ):
        yield FramePacket(
            frame_index=output_frame_index,
            timestamp=output_frame_index / target_fps,
            captured_at=time.perf_counter(),
            frame=last_frame.copy(),
        )
        output_frame_index += 1


def produce_frames(
    capture: cv2.VideoCapture,
    *,
    webcam: bool,
    target_fps: float,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    telemetry: PipelineTelemetry,
    raw_frame_callback: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Read frames on one thread and feed the bounded pose queue.

    Usage: Inference only.

    Webcam mode drops stale queued frames to protect latency. File mode blocks
    instead because offline inference must retain every CFR frame.
    """

    iterator = (
        iter_webcam_frames(capture)
        if webcam
        else iter_file_cfr_frames(capture, target_fps)
    )
    try:
        for packet in iterator:
            if stop_event.is_set():
                break
            telemetry.record_event("capture", packet.captured_at)
            if raw_frame_callback is not None:
                raw_frame_callback(packet.frame)
            if webcam:
                dropped = put_latest(output_queue, packet)
                telemetry.record_drop("capture_queue", dropped)
            elif not put_ordered(output_queue, packet, stop_event):
                break
    finally:
        if webcam or stop_event.is_set():
            # Live shutdown must not deadlock behind a stale queued frame.
            put_latest(output_queue, END_OF_STREAM)
        else:
            # Offline mode preserves the final frame before its sentinel.
            output_queue.put(END_OF_STREAM)
