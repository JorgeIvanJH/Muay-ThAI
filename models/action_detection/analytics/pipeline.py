"""Shared orchestration, display state and persistence for strike analytics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from models.action_detection.analytics.kinematics import KinematicsTracker
from models.action_detection.analytics.strike_events import (
    StrikeEventStateMachine,
    StrikeObservation,
    StrikeStateMachineConfig,
)
from models.action_detection.analytics.strike_speed import (
    SpeedEstimate,
    estimate_strike_speed,
)
from models.action_detection.analytics.types import (
    JointPoint,
    SIDES,
    STRIKE_TYPES,
    StrikeEvent,
)


METRIC_NAMES = ("count", "speed")
EVENT_FIELDNAMES = (
    "event_id",
    "strike_type",
    "side",
    "start_frame",
    "apex_frame",
    "end_frame",
    "start_time_seconds",
    "apex_time_seconds",
    "end_time_seconds",
    "duration_seconds",
    "peak_classification_confidence",
    "average_speed_mps",
    "sampled_peak_speed_mps",
    "robust_peak_speed_mps",
    "interpolated_peak_speed_mps",
    "interpolated_peak_time_seconds",
    "valid_speed_samples",
)


@dataclass(frozen=True)
class AnalyticsConfig: # TODO: why not config file?
    """User-facing selection plus basic kinematic settings."""

    enabled_metrics: tuple[str, ...] = METRIC_NAMES
    person_height_m: float = 1.75
    keypoint_confidence: float = 0.25
    smoothing_alpha: float = 0.5

    def __post_init__(self) -> None: # TODO: is this special __init__?
        """
        Normalize and validate the runtime analytics settings.

        Usage: Inference only.
        """

        metrics = tuple(dict.fromkeys(self.enabled_metrics))
        invalid = sorted(set(metrics) - set(METRIC_NAMES))
        if invalid:
            raise ValueError("Unknown analytics metrics: " + ", ".join(invalid))
        if not metrics:
            raise ValueError("At least one analytics metric is required")
        if not 0.5 <= self.person_height_m <= 2.5:
            raise ValueError("person_height_m must be between 0.5 and 2.5")
        if not 0.0 <= self.keypoint_confidence <= 1.0:
            raise ValueError("keypoint_confidence must be between 0 and 1")
        object.__setattr__(self, "enabled_metrics", metrics)


@dataclass(frozen=True)
class CompletedStrike:
    """A completed event with its optional requested speed estimate."""

    event: StrikeEvent
    speed: SpeedEstimate | None

    def as_record(self) -> dict[str, object]:
        """
        Convert the completed strike into a serializable output row.

        Usage: Inference only.
        """

        record: dict[str, object] = {
            "event_id": self.event.event_id,
            "strike_type": self.event.strike_type,
            "side": self.event.side,
            "start_frame": self.event.start_frame,
            "apex_frame": self.event.apex_frame,
            "end_frame": self.event.end_frame,
            "start_time_seconds": self.event.start_timestamp,
            "apex_time_seconds": self.event.apex_timestamp,
            "end_time_seconds": self.event.end_timestamp,
            "duration_seconds": (
                self.event.end_timestamp - self.event.start_timestamp
            ),
            "peak_classification_confidence": (
                self.event.peak_classification_confidence
            ),
        }
        if self.speed is None:
            record.update(
                {
                    "average_speed_mps": None,
                    "sampled_peak_speed_mps": None,
                    "robust_peak_speed_mps": None,
                    "interpolated_peak_speed_mps": None,
                    "interpolated_peak_time_seconds": None,
                    "valid_speed_samples": 0,
                }
            )
        else:
            record.update(
                {
                    "average_speed_mps": self.speed.average_mps,
                    "sampled_peak_speed_mps": self.speed.sampled_peak_mps,
                    "robust_peak_speed_mps": self.speed.robust_peak_mps,
                    "interpolated_peak_speed_mps": (
                        self.speed.interpolated_peak_mps
                    ),
                    "interpolated_peak_time_seconds": (
                        self.speed.interpolated_peak_timestamp
                    ),
                    "valid_speed_samples": self.speed.valid_samples,
                }
            )
        return record


@dataclass(frozen=True)
class AnalyticsSnapshot:
    """Analytics values available for one video frame."""

    state: str
    active_strike: tuple[str, str] | None
    pixels_per_metre: float | None
    counts: dict[str, dict[str, int]] | None
    new_events: tuple[CompletedStrike, ...]
    latest_event: CompletedStrike | None

    def as_record(self) -> dict[str, object]:
        """
        Convert one frame's analytics state into a JSON-safe record.

        Usage: Inference only.
        """

        return {
            "state": self.state,
            "active_strike": (
                None
                if self.active_strike is None
                else {
                    "strike_type": self.active_strike[0],
                    "side": self.active_strike[1],
                }
            ),
            "pixels_per_metre": self.pixels_per_metre,
            "counts": self.counts,
            "completed_events": [event.as_record() for event in self.new_events],
        }


class StrikeAnalytics:
    """
    Compute selected metrics from pose points and striking probabilities.
    """

    def __init__(
        self,
        config: AnalyticsConfig,
        *,
        state_machine_config: StrikeStateMachineConfig | None = None,
    ) -> None:
        """
        Initialize kinematics, event detection, counts, and event history.

        Usage: Inference only.
        """

        self.config = config
        self._kinematics = KinematicsTracker(
            config.person_height_m,
            smoothing_alpha=config.smoothing_alpha,
        )
        self._detector = StrikeEventStateMachine(state_machine_config)
        self._counts = {
            strike_type: {side: 0 for side in SIDES}
            for strike_type in STRIKE_TYPES
        }
        self._events: list[CompletedStrike] = []
        self._latest_event: CompletedStrike | None = None
        self._pixels_per_metre: float | None = None

    @property
    def events(self) -> tuple[CompletedStrike, ...]:
        """
        Return an immutable view of the completed events in this run.

        Usage: Inference only.
        """

        return tuple(self._events)

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """
        Return a defensive copy of strike counts by type and side.

        Usage: Inference only.
        """

        return {
            strike_type: dict(side_counts)
            for strike_type, side_counts in self._counts.items()
        }

    def _complete(
        self,
        events: tuple[StrikeEvent, ...],
    ) -> tuple[CompletedStrike, ...]:
        """
        Add speeds, persist events, and update the requested counts.

        Usage: Inference only.
        """

        completed = []
        for event in events:
            speed = (
                estimate_strike_speed(event.speed_samples)
                if "speed" in self.config.enabled_metrics
                else None
            )
            strike = CompletedStrike(event=event, speed=speed)
            self._events.append(strike)
            self._latest_event = strike
            if "count" in self.config.enabled_metrics:
                self._counts[event.strike_type][event.side] += 1
            completed.append(strike)
        return tuple(completed)

    def _snapshot(
        self,
        new_events: tuple[CompletedStrike, ...] = (),
    ) -> AnalyticsSnapshot:
        """
        Capture the analytics state exposed for the current frame.

        Usage: Inference only.
        """

        return AnalyticsSnapshot(
            state=self._detector.state.value,
            active_strike=self._detector.active_strike,
            pixels_per_metre=self._pixels_per_metre,
            counts=(self.counts if "count" in self.config.enabled_metrics else None),
            new_events=new_events,
            latest_event=self._latest_event,
        )

    def update(
        self,
        *,
        frame_index: int,
        timestamp: float,
        pose_points: Mapping[str, JointPoint],
        striking_probabilities: Mapping[str, float],
    ) -> AnalyticsSnapshot:
        """
        Process one pose and its striking-class probability inference.

        Usage: Inference only.
        """

        filtered_points = {
            name: point
            for name, point in pose_points.items()
            if np.isfinite(point.x_px)
            and np.isfinite(point.y_px)
            and np.isfinite(point.confidence)
            and point.confidence >= self.config.keypoint_confidence
        }
        kinematics = self._kinematics.update(timestamp, filtered_points)
        self._pixels_per_metre = kinematics.pixels_per_metre
        events = self._detector.update(
            StrikeObservation(
                frame_index=frame_index,
                timestamp=timestamp,
                probabilities=striking_probabilities,
                limb_speeds_mps=kinematics.limb_speeds_mps,
            )
        )
        return self._snapshot(self._complete(events))

    def finalize(self) -> AnalyticsSnapshot:
        """
        Flush an active event when the input stream ends.

        Usage: Inference only.
        """

        return self._snapshot(self._complete(self._detector.flush()))

    def overlay_lines(self, snapshot: AnalyticsSnapshot) -> tuple[str, ...]:
        """
        Format compact count, speed, and event-state overlay lines.

        Usage: Inference only.
        """

        lines: list[str] = []
        if snapshot.counts is not None:
            counts = snapshot.counts
            lines.extend(
                (
                    "Counts  Punch L/R {}/{} | Elbow L/R {}/{}".format(
                        counts["punch"]["left"],
                        counts["punch"]["right"],
                        counts["elbow"]["left"],
                        counts["elbow"]["right"],
                    ),
                    "Counts  Kick L/R {}/{} | Knee L/R {}/{}".format(
                        counts["kick"]["left"],
                        counts["kick"]["right"],
                        counts["knee"]["left"],
                        counts["knee"]["right"],
                    ),
                )
            )
        if "speed" in self.config.enabled_metrics:
            latest = snapshot.latest_event
            if latest is None or latest.speed is None:
                lines.append("Speed  waiting for a completed strike")
            else:
                lines.append(
                    "Speed  last {} {}: robust {:.2f} m/s | interpolated {:.2f} m/s".format(
                        latest.event.side,
                        latest.event.strike_type,
                        latest.speed.robust_peak_mps,
                        latest.speed.interpolated_peak_mps,
                    )
                )
        if snapshot.active_strike is not None:
            lines.append(
                f"Event  {snapshot.state}: "
                f"{snapshot.active_strike[0]} {snapshot.active_strike[1]}"
            )
        return tuple(lines)

    def write_outputs(
        self,
        events_path: Path,
        summary_path: Path,
    ) -> None:
        """
        Write one-row-per-event CSV and an aggregate JSON summary.

        Usage: Inference only.
        """

        events_path.parent.mkdir(parents=True, exist_ok=True)
        event_rows = [event.as_record() for event in self._events]
        with events_path.open("w", encoding="utf-8", newline="") as events_file:
            writer = csv.DictWriter(events_file, fieldnames=EVENT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(event_rows)

        by_type: dict[str, list[float]] = defaultdict(list)
        for completed in self._events:
            if completed.speed is not None:
                by_type[completed.event.strike_type].append(
                    completed.speed.robust_peak_mps
                )
        speed_summary = {
            strike_type: {
                "events_with_speed": len(values),
                "average_robust_peak_mps": (
                    float(np.mean(values)) if values else None
                ),
                "maximum_robust_peak_mps": max(values) if values else None,
            }
            for strike_type in STRIKE_TYPES
            for values in (by_type.get(strike_type, []),)
        }
        summary = {
            "configuration": {
                "enabled_metrics": list(self.config.enabled_metrics),
                "person_height_m": self.config.person_height_m,
                "keypoint_confidence": self.config.keypoint_confidence,
                "smoothing_alpha": self.config.smoothing_alpha,
                "state_machine": asdict(self._detector.config),
            },
            "completed_events": len(self._events),
            "counts": (
                self.counts if "count" in self.config.enabled_metrics else None
            ),
            "speed": (
                speed_summary if "speed" in self.config.enabled_metrics else None
            ),
            "final_pixels_per_metre": self._pixels_per_metre,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
            summary_file.write("\n")
