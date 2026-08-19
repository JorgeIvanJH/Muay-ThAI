"""Causal pose smoothing and limb endpoint velocities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.action_detection.analytics.anthropometry import (
    AnthropometricScaleEstimator,
)
from models.action_detection.analytics.types import JointPoint, SIDES, STRIKE_TYPES


STRIKE_ENDPOINTS = {
    "punch": "wrist",
    "elbow": "elbow",
    "kick": "ankle",
    "knee": "knee",
}


@dataclass(frozen=True)
class KinematicFrame:
    """Smoothed pose-derived quantities for one frame."""

    timestamp: float
    points: dict[str, JointPoint]
    joint_speeds_mps: dict[str, float]
    limb_speeds_mps: dict[str, dict[str, float | None]]
    pixels_per_metre: float | None


class KinematicsTracker:
    """Maintain causal EMA coordinates, session scale and physical velocity."""

    def __init__(
        self,
        person_height_m: float,
        *,
        smoothing_alpha: float = 0.5,
        maximum_velocity_gap_seconds: float = 0.12,
    ) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be between 0 and 1")
        if maximum_velocity_gap_seconds <= 0.0:
            raise ValueError("maximum_velocity_gap_seconds must be positive")
        self._alpha = smoothing_alpha
        self._maximum_gap = maximum_velocity_gap_seconds
        self._scale = AnthropometricScaleEstimator(person_height_m)
        self._smoothed: dict[str, JointPoint] = {}
        self._previous: dict[str, tuple[float, JointPoint]] = {}
        self._last_timestamp: float | None = None

    def update(
        self,
        timestamp: float,
        points: dict[str, JointPoint],
    ) -> KinematicFrame:
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("analytics timestamps must be strictly increasing")
        self._last_timestamp = timestamp

        current: dict[str, JointPoint] = {}
        for name, point in points.items():
            previous = self._smoothed.get(name)
            if previous is None:
                smoothed = point
            else:
                smoothed = JointPoint(
                    x_px=self._alpha * point.x_px + (1.0 - self._alpha) * previous.x_px,
                    y_px=self._alpha * point.y_px + (1.0 - self._alpha) * previous.y_px,
                    confidence=point.confidence,
                )
            current[name] = smoothed
        self._smoothed.update(current)

        pixels_per_metre = self._scale.update(current)
        joint_speeds: dict[str, float] = {}
        if pixels_per_metre is not None and pixels_per_metre > 0.0:
            for name, point in current.items():
                previous_sample = self._previous.get(name)
                if previous_sample is None:
                    continue
                previous_timestamp, previous_point = previous_sample
                elapsed = timestamp - previous_timestamp
                if not 0.0 < elapsed <= self._maximum_gap:
                    continue
                distance_px = float(
                    np.hypot(
                        point.x_px - previous_point.x_px,
                        point.y_px - previous_point.y_px,
                    )
                )
                joint_speeds[name] = distance_px / elapsed / pixels_per_metre

        for name, point in current.items():
            self._previous[name] = (timestamp, point)

        limb_speeds = {
            strike_type: {
                side: joint_speeds.get(f"{side}_{endpoint}")
                for side in SIDES
            }
            for strike_type, endpoint in STRIKE_ENDPOINTS.items()
            if strike_type in STRIKE_TYPES
        }
        return KinematicFrame(
            timestamp=timestamp,
            points=current,
            joint_speeds_mps=joint_speeds,
            limb_speeds_mps=limb_speeds,
            pixels_per_metre=pixels_per_metre,
        )
