"""Asynchronous CFR video writing that cannot silently block inference."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from models.action_detection.realtime.queues import put_latest
from models.action_detection.realtime.telemetry import PipelineTelemetry
from models.action_detection.realtime.types import END_OF_STREAM


class AsyncVideoWriter:
    """Write frames on a dedicated bounded worker thread."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        queue_size: int,
        telemetry: PipelineTelemetry,
        stage_name: str,
        drop_when_full: bool,
    ) -> None:
        """Configure but do not yet open the lazily sized video writer.

        Usage: Inference only.
        """

        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        self.path = Path(path)
        self.fps = float(fps)
        self.telemetry = telemetry
        self.stage_name = stage_name
        self.drop_when_full = bool(drop_when_full)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._written_frames = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"{stage_name}-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def written_frames(self) -> int:
        """Return how many frames the encoder accepted.

        Usage: Inference only.
        """

        return self._written_frames

    def submit(self, frame: np.ndarray) -> None:
        """Queue one read-only frame without modifying its pixel data.

        Usage: Inference only.
        """

        self._raise_if_failed()
        if self.drop_when_full:
            dropped = put_latest(self._queue, frame)
            self.telemetry.record_drop(self.stage_name, dropped)
        else:
            self._queue.put(frame)

    def close(self) -> None:
        """Drain queued frames, close the codec and propagate worker errors.

        Usage: Inference only.
        """

        while self._thread.is_alive() and self._error is None:
            try:
                self._queue.put(END_OF_STREAM, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        """Raise a writer failure on the coordinating thread.

        Usage: Inference only.
        """

        if self._error is not None:
            raise RuntimeError(
                f"Asynchronous video writer failed for {self.path}"
            ) from self._error

    def _open(self, frame: np.ndarray) -> cv2.VideoWriter:
        """Open a 30-FPS MP4 writer using the first frame's dimensions.

        Usage: Inference only.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.path}")
        return writer

    def _run(self) -> None:
        """Consume queued frames until the close sentinel arrives.

        Usage: Inference only.
        """

        writer: cv2.VideoWriter | None = None
        try:
            while True:
                item = self._queue.get()
                if item is END_OF_STREAM:
                    break
                if writer is None:
                    writer = self._open(item)
                writer.write(item)
                self._written_frames += 1
                self.telemetry.record_event(self.stage_name)
        except BaseException as error:
            self._error = error
        finally:
            if writer is not None:
                writer.release()
