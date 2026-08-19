"""Small shared data contracts for action analytics."""

from __future__ import annotations

from dataclasses import dataclass


STRIKE_TYPES = ("punch", "elbow", "kick", "knee")
SIDES = ("left", "right")


@dataclass(frozen=True)
class JointPoint:
    """One image-plane keypoint measurement."""

    x_px: float
    y_px: float
    confidence: float


@dataclass(frozen=True)
class SpeedSample:
    """One timestamped physical-speed estimate."""

    timestamp: float
    speed_mps: float


@dataclass(frozen=True)
class StrikeEvent:
    """One completed and limb-resolved strike event."""

    event_id: int
    strike_type: str
    side: str
    start_frame: int
    apex_frame: int
    end_frame: int
    start_timestamp: float
    apex_timestamp: float
    end_timestamp: float
    peak_classification_confidence: float
    speed_samples: tuple[SpeedSample, ...]
