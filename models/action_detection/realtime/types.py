"""Small immutable messages passed between real-time pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.action_detection.realtime.pose import SelectedPose


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame together with its truthful source timing.

    Usage: Inference only.

    ``timestamp`` belongs to the source timeline. For a webcam it is measured
    with a monotonic clock; for a file it belongs to the resampled CFR
    timeline. ``captured_at`` is always a wall-clock performance timestamp and
    is used to measure end-to-end latency.
    """

    frame_index: int
    timestamp: float
    captured_at: float
    frame: np.ndarray


@dataclass(frozen=True)
class PosePacket:
    """A frame paired with the selected pose produced by YOLO.

    Usage: Inference only.
    """

    frame: FramePacket
    pose: SelectedPose


# A shared identity sentinel is safer than using ``None`` because ``None`` can
# be a legitimate optional value in other pipeline messages.
END_OF_STREAM = object()
