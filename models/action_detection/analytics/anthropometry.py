"""Height-based canonical proportions and image-to-metre calibration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from models.action_detection.analytics.types import JointPoint


@dataclass(frozen=True)
class SegmentSpec:
    """A visible long-bone segment and its approximate height ratio."""

    name: str
    start_joint: str
    end_joint: str
    height_ratio: float


# Approximate adult proportions. They provide a practical monocular scale,
# not a clinical body measurement. Each left/right segment is calibrated
# independently so temporary occlusion of one side does not stop estimation.
CANONICAL_SEGMENTS = (
    SegmentSpec("left_upper_arm", "left_shoulder", "left_elbow", 0.186),
    SegmentSpec("right_upper_arm", "right_shoulder", "right_elbow", 0.186),
    SegmentSpec("left_forearm", "left_elbow", "left_wrist", 0.146),
    SegmentSpec("right_forearm", "right_elbow", "right_wrist", 0.146),
    SegmentSpec("left_thigh", "left_hip", "left_knee", 0.245),
    SegmentSpec("right_thigh", "right_hip", "right_knee", 0.245),
    SegmentSpec("left_lower_leg", "left_knee", "left_ankle", 0.246),
    SegmentSpec("right_lower_leg", "right_knee", "right_ankle", 0.246),
)


def canonical_segment_lengths(person_height_m: float) -> dict[str, float]:
    """Return estimated segment lengths in metres for one supplied height."""

    if not 0.5 <= person_height_m <= 2.5:
        raise ValueError("person_height_m must be between 0.5 and 2.5")
    return {
        segment.name: person_height_m * segment.height_ratio
        for segment in CANONICAL_SEGMENTS
    }


class AnthropometricScaleEstimator:
    """
    Estimate pixels per metre from projected long-bone lengths.

    A high temporal quantile is retained for every segment because a projected
    2-D bone is shortened when it points partly towards the camera. The median
    across visible segments and an EMA keep the session scale stable.
    """

    def __init__(
        self,
        person_height_m: float,
        *,
        history_size: int = 150,
        projection_quantile: float = 0.9,
        smoothing_alpha: float = 0.15,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if not 0.5 <= projection_quantile <= 1.0:
            raise ValueError("projection_quantile must be between 0.5 and 1")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be between 0 and 1")
        self._lengths = canonical_segment_lengths(person_height_m)
        self._history = {
            segment.name: deque(maxlen=history_size)
            for segment in CANONICAL_SEGMENTS
        }
        self._quantile = projection_quantile
        self._alpha = smoothing_alpha
        self._pixels_per_metre: float | None = None

    @property
    def pixels_per_metre(self) -> float | None:
        return self._pixels_per_metre

    def update(self, points: dict[str, JointPoint]) -> float | None:
        for segment in CANONICAL_SEGMENTS:
            start = points.get(segment.start_joint)
            end = points.get(segment.end_joint)
            if start is None or end is None:
                continue
            projected_length = float(
                np.hypot(end.x_px - start.x_px, end.y_px - start.y_px)
            )
            if projected_length < 2.0:
                continue
            self._history[segment.name].append(
                projected_length / self._lengths[segment.name]
            )

        segment_scales = [
            float(np.quantile(values, self._quantile))
            for values in self._history.values()
            if values
        ]
        if not segment_scales:
            return self._pixels_per_metre
        observed_scale = float(np.median(segment_scales))
        if self._pixels_per_metre is None:
            self._pixels_per_metre = observed_scale
        else:
            self._pixels_per_metre = (
                self._alpha * observed_scale
                + (1.0 - self._alpha) * self._pixels_per_metre
            )
        return self._pixels_per_metre
