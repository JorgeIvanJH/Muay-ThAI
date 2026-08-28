"""Robust and sub-frame peak estimates for a completed strike trajectory."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from models.action_detection.analytics.types import SpeedSample


@dataclass(frozen=True)
class SpeedEstimate:
    """Summary of the valid physical-speed samples in one strike."""

    average_mps: float
    sampled_peak_mps: float
    robust_peak_mps: float
    interpolated_peak_mps: float
    interpolated_peak_timestamp: float
    valid_samples: int


def _quadratic_peak(
    samples: tuple[SpeedSample, ...],
) -> tuple[float, float]:
    """
    Refine a sampled peak with a guarded local quadratic fit.

    Usage: Inference only.
    """

    speeds = np.asarray([sample.speed_mps for sample in samples], dtype=float)
    timestamps = np.asarray([sample.timestamp for sample in samples], dtype=float)
    peak_index = int(np.argmax(speeds))
    start = max(0, peak_index - 2)
    stop = min(len(samples), peak_index + 3)
    local_speeds = speeds[start:stop]
    local_timestamps = timestamps[start:stop]
    sampled_peak = float(speeds[peak_index])
    sampled_time = float(timestamps[peak_index])
    if len(local_speeds) < 3 or len(np.unique(local_timestamps)) < 3:
        return sampled_peak, sampled_time

    centre = sampled_time
    local_time = local_timestamps - centre
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quadratic, linear, constant = np.polyfit(
                local_time,
                local_speeds,
                2,
            )
    except (ValueError, np.linalg.LinAlgError, Warning):
        return sampled_peak, sampled_time

    if not np.all(np.isfinite((quadratic, linear, constant))) or quadratic >= 0:
        return sampled_peak, sampled_time
    vertex_time = float(-linear / (2.0 * quadratic))
    if not float(local_time.min()) <= vertex_time <= float(local_time.max()):
        return sampled_peak, sampled_time
    vertex_speed = float(
        quadratic * vertex_time**2 + linear * vertex_time + constant
    )
    # Reject polynomial overshoot. Interpolation refines the observed motion;
    # it must not invent an arbitrarily faster strike.
    if not sampled_peak <= vertex_speed <= sampled_peak * 1.5:
        return sampled_peak, sampled_time
    return vertex_speed, centre + vertex_time


def estimate_strike_speed(
    speed_samples: tuple[SpeedSample, ...],
) -> SpeedEstimate | None:
    """
    Return robust sampled and quadratic-refined speed summaries.

    Usage: Inference only.
    """

    valid = tuple(
        sample
        for sample in speed_samples
        if np.isfinite(sample.timestamp)
        and np.isfinite(sample.speed_mps)
        and sample.speed_mps >= 0.0
    )
    if not valid:
        return None
    speeds = np.asarray([sample.speed_mps for sample in valid], dtype=float)
    fastest_count = min(3, len(speeds))
    fastest = np.partition(speeds, len(speeds) - fastest_count)[-fastest_count:]
    interpolated_peak, interpolated_time = _quadratic_peak(valid)
    return SpeedEstimate(
        average_mps=float(np.mean(speeds)),
        sampled_peak_mps=float(np.max(speeds)),
        robust_peak_mps=float(np.mean(fastest)),
        interpolated_peak_mps=interpolated_peak,
        interpolated_peak_timestamp=interpolated_time,
        valid_samples=len(valid),
    )
