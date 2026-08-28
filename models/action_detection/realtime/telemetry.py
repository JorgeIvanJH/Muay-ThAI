"""Thread-safe rolling performance measurements for the live overlay."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Current rolling throughput, latency and dropped-frame values."""

    capture_fps: float
    pose_fps: float
    action_fps: float
    pose_p95_ms: float
    action_p95_ms: float
    end_to_end_p95_ms: float
    dropped_frames: int

    def overlay_text(self) -> str:
        """Format the compact performance line drawn over inference video.

        Usage: Inference only.
        """

        return (
            f"capture {self.capture_fps:.1f} | pose {self.pose_fps:.1f} | "
            f"actions {self.action_fps:.1f} FPS | latency "
            f"{self.end_to_end_p95_ms:.0f} ms p95 | dropped "
            f"{self.dropped_frames}"
        )


class PipelineTelemetry:
    """Collect bounded rolling samples from concurrent pipeline workers."""

    def __init__(self, sample_count: int = 180) -> None:
        """Initialize event clocks, latency windows and drop counters.

        Usage: Inference only.
        """

        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        self._sample_count = int(sample_count)
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._sample_count)
        )
        self._latencies: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._sample_count)
        )
        self._drops: Counter[str] = Counter()

    def record_event(self, stage: str, when: float | None = None) -> None:
        """Record completion time for rolling stage throughput.

        Usage: Inference only.
        """

        timestamp = time.perf_counter() if when is None else float(when)
        with self._lock:
            self._events[stage].append(timestamp)

    def record_latency(self, stage: str, seconds: float) -> None:
        """Record one non-negative stage latency sample.

        Usage: Inference only.
        """

        with self._lock:
            self._latencies[stage].append(max(0.0, float(seconds)))

    def record_drop(self, stage: str, count: int = 1) -> None:
        """Add frames discarded by a bounded real-time queue.

        Usage: Inference only.
        """

        if count <= 0:
            return
        with self._lock:
            self._drops[stage] += int(count)

    @staticmethod
    def _fps(events: tuple[float, ...]) -> float:
        """Calculate rolling throughput from monotonic completion times.

        Usage: Inference only.
        """

        if len(events) < 2:
            return 0.0
        duration = events[-1] - events[0]
        return 0.0 if duration <= 0.0 else (len(events) - 1) / duration

    @staticmethod
    def _p95_ms(samples: tuple[float, ...]) -> float:
        """Return the 95th percentile of seconds as milliseconds.

        Usage: Inference only.
        """

        if not samples:
            return 0.0
        return float(np.percentile(samples, 95)) * 1000.0

    def snapshot(self) -> PerformanceSnapshot:
        """Copy current measurements into an immutable display snapshot.

        Usage: Inference only.
        """

        with self._lock:
            events = {
                name: tuple(values) for name, values in self._events.items()
            }
            latencies = {
                name: tuple(values) for name, values in self._latencies.items()
            }
            dropped = sum(self._drops.values())
        return PerformanceSnapshot(
            capture_fps=self._fps(events.get("capture", ())),
            pose_fps=self._fps(events.get("pose", ())),
            action_fps=self._fps(events.get("action", ())),
            pose_p95_ms=self._p95_ms(latencies.get("pose", ())),
            action_p95_ms=self._p95_ms(latencies.get("action", ())),
            end_to_end_p95_ms=self._p95_ms(
                latencies.get("end_to_end", ())
            ),
            dropped_frames=dropped,
        )

    def drop_counts(self) -> dict[str, int]:
        """Return dropped items grouped by queue or output stage.

        Usage: Inference only.
        """

        with self._lock:
            return dict(self._drops)
