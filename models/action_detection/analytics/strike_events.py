"""A small configurable state machine that turns frame scores into strikes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from models.action_detection.analytics.types import (
    SIDES,
    STRIKE_TYPES,
    SpeedSample,
    StrikeEvent,
)


class StrikeState(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"


@dataclass(frozen=True)
class StrikeStateMachineConfig:
    """Initial conservative rules; ground truth can tune these later."""

    activation_confidence: float = 0.60
    continuation_confidence: float = 0.35
    candidate_frames: int = 2
    release_grace_seconds: float = 0.10
    minimum_event_seconds: float = 0.08
    maximum_event_seconds: float = 2.0
    cooldown_seconds: float = 0.15
    continuation_motion_ratio: float = 0.45
    motion_thresholds_mps: Mapping[str, float] = field(
        default_factory=lambda: {
            "punch": 0.80,
            "elbow": 0.60,
            "kick": 0.90,
            "knee": 0.60,
        }
    )

    def __post_init__(self) -> None:
        """
        Validate state-machine thresholds after configuration creation.

        Usage: Inference only.
        """
        if not 0.0 <= self.continuation_confidence <= self.activation_confidence <= 1.0:
            raise ValueError("invalid state-machine confidence thresholds")
        if self.candidate_frames < 1:
            raise ValueError("candidate_frames must be positive")
        if self.release_grace_seconds < 0.0:
            raise ValueError("release_grace_seconds cannot be negative")
        if self.minimum_event_seconds < 0.0:
            raise ValueError("minimum_event_seconds cannot be negative")
        if self.maximum_event_seconds <= self.minimum_event_seconds:
            raise ValueError("maximum_event_seconds must exceed minimum_event_seconds")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds cannot be negative")
        if not 0.0 <= self.continuation_motion_ratio <= 1.0:
            raise ValueError("continuation_motion_ratio must be between 0 and 1")
        missing = set(STRIKE_TYPES) - set(self.motion_thresholds_mps)
        if missing:
            raise ValueError("missing motion thresholds: " + ", ".join(sorted(missing)))


@dataclass(frozen=True)
class StrikeObservation:
    frame_index: int
    timestamp: float
    probabilities: Mapping[str, float]
    limb_speeds_mps: Mapping[str, Mapping[str, float | None]]


@dataclass
class _Candidate:
    strike_type: str
    side: str
    observations: list[tuple[int, float, float, float]]


@dataclass
class _ActiveEvent:
    strike_type: str
    side: str
    start_frame: int
    start_timestamp: float
    last_supported_frame: int
    last_supported_timestamp: float
    apex_frame: int
    apex_timestamp: float
    apex_speed: float
    peak_confidence: float
    speed_samples: list[SpeedSample]


class StrikeEventStateMachine:
    """
    Confirm, delimit and de-duplicate limb-specific strikes.

    Side is selected from endpoint velocity during the candidate phase and is
    locked for the event. Classification and continued limb motion keep an
    event active; a short release grace tolerates classifier/keypoint dropouts.
    """

    def __init__(self, config: StrikeStateMachineConfig | None = None) -> None:
        """
        Initialize an idle strike detector and its cooldown state.

        Usage: Inference only.
        """
        self.config = config or StrikeStateMachineConfig()
        self._candidate: _Candidate | None = None
        self._active: _ActiveEvent | None = None
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._next_event_id = 1

    @property
    def state(self) -> StrikeState:
        """
        Return whether the detector is idle, confirming, or active.

        Usage: Inference only.
        """
        if self._active is not None:
            return StrikeState.ACTIVE
        if self._candidate is not None:
            return StrikeState.CANDIDATE
        return StrikeState.IDLE

    @property
    def active_strike(self) -> tuple[str, str] | None:
        """
        Return the current candidate or active strike type and side.

        Usage: Inference only.
        """
        if self._active is not None:
            return self._active.strike_type, self._active.side
        if self._candidate is not None:
            return self._candidate.strike_type, self._candidate.side
        return None

    def _eligible_candidate(
        self,
        observation: StrikeObservation,
    ) -> tuple[str, str, float, float] | None:
        """
        Select a threshold-passing strike and faster eligible limb.

        Usage: Inference only.
        """
        strike_type = max(
            STRIKE_TYPES,
            key=lambda name: float(observation.probabilities.get(name, 0.0)),
        )
        probability = float(observation.probabilities.get(strike_type, 0.0))
        if probability < self.config.activation_confidence:
            return None
        side_speeds = observation.limb_speeds_mps.get(strike_type, {})
        finite_speeds = {
            side: float(side_speeds[side])
            for side in SIDES
            if side_speeds.get(side) is not None
            and np.isfinite(side_speeds[side])
        }
        if not finite_speeds:
            return None
        side = max(finite_speeds, key=finite_speeds.get)
        speed = finite_speeds[side]
        if speed < self.config.motion_thresholds_mps[strike_type]:
            return None
        if observation.timestamp < self._cooldowns.get((strike_type, side), -np.inf):
            return None
        return strike_type, side, probability, speed

    def _start_or_advance_candidate(
        self,
        observation: StrikeObservation,
    ) -> None:
        """
        Start, continue, replace, or reject the current candidate.

        Usage: Inference only.
        """
        eligible = self._eligible_candidate(observation)
        if eligible is None:
            self._candidate = None
            return
        strike_type, side, probability, speed = eligible
        if (
            self._candidate is None
            or self._candidate.strike_type != strike_type
            or self._candidate.side != side
        ):
            self._candidate = _Candidate(
                strike_type=strike_type,
                side=side,
                observations=[
                    (observation.frame_index, observation.timestamp, probability, speed)
                ],
            )
        else:
            self._candidate.observations.append(
                (observation.frame_index, observation.timestamp, probability, speed)
            )
        if len(self._candidate.observations) >= self.config.candidate_frames:
            self._activate_candidate()

    def _activate_candidate(self) -> None:
        """
        Promote a sufficiently sustained candidate to an active event.

        Usage: Inference only.
        """
        if self._candidate is None:
            return
        observations = self._candidate.observations
        apex = max(observations, key=lambda item: item[3])
        first = observations[0]
        last = observations[-1]
        self._active = _ActiveEvent(
            strike_type=self._candidate.strike_type,
            side=self._candidate.side,
            start_frame=first[0],
            start_timestamp=first[1],
            last_supported_frame=last[0],
            last_supported_timestamp=last[1],
            apex_frame=apex[0],
            apex_timestamp=apex[1],
            apex_speed=apex[3],
            peak_confidence=max(item[2] for item in observations),
            speed_samples=[
                SpeedSample(item[1], item[3]) for item in observations
            ],
        )
        self._candidate = None

    def _finish_active(self) -> StrikeEvent | None:
        """
        Validate and emit the active event, then start its cooldown.

        Usage: Inference only.
        """
        active = self._active
        self._active = None
        if active is None:
            return None
        duration = active.last_supported_timestamp - active.start_timestamp
        if duration < self.config.minimum_event_seconds:
            return None
        samples = tuple(
            sample
            for sample in active.speed_samples
            if sample.timestamp <= active.last_supported_timestamp + 1e-9
        )
        event = StrikeEvent(
            event_id=self._next_event_id,
            strike_type=active.strike_type,
            side=active.side,
            start_frame=active.start_frame,
            apex_frame=active.apex_frame,
            end_frame=active.last_supported_frame,
            start_timestamp=active.start_timestamp,
            apex_timestamp=active.apex_timestamp,
            end_timestamp=active.last_supported_timestamp,
            peak_classification_confidence=active.peak_confidence,
            speed_samples=samples,
        )
        self._next_event_id += 1
        self._cooldowns[(event.strike_type, event.side)] = (
            event.end_timestamp + self.config.cooldown_seconds
        )
        return event

    def update(self, observation: StrikeObservation) -> tuple[StrikeEvent, ...]:
        """
        Advance the state machine with one timestamped inference frame.

        Usage: Inference only.
        """
        if self._active is None:
            self._start_or_advance_candidate(observation)
            return ()

        active = self._active
        probability = float(observation.probabilities.get(active.strike_type, 0.0))
        side_speed = observation.limb_speeds_mps.get(active.strike_type, {}).get(
            active.side
        )
        valid_speed = side_speed is not None and np.isfinite(side_speed)
        if valid_speed:
            speed = float(side_speed)
            active.speed_samples.append(SpeedSample(observation.timestamp, speed))
        active.peak_confidence = max(active.peak_confidence, probability)

        motion_supported = bool(
            valid_speed
            and float(side_speed)
            >= self.config.motion_thresholds_mps[active.strike_type]
            * self.config.continuation_motion_ratio
        )
        # Classification identifies the strike family; endpoint motion
        # separates consecutive strikes even when the classifier remains
        # confident across a short background gap. The release grace below
        # still tolerates brief keypoint or classification dropouts.
        supported = (
            probability >= self.config.continuation_confidence
            and motion_supported
        )
        if supported:
            active.last_supported_frame = observation.frame_index
            active.last_supported_timestamp = observation.timestamp
            if valid_speed and float(side_speed) > active.apex_speed:
                active.apex_speed = float(side_speed)
                active.apex_frame = observation.frame_index
                active.apex_timestamp = observation.timestamp

        expired = (
            observation.timestamp - active.start_timestamp
            >= self.config.maximum_event_seconds
        )
        released = (
            not supported
            and observation.timestamp - active.last_supported_timestamp
            >= self.config.release_grace_seconds
        )
        if not expired and not released:
            return ()

        event = self._finish_active()
        self._start_or_advance_candidate(observation)
        return () if event is None else (event,)

    def flush(self) -> tuple[StrikeEvent, ...]:
        """
        Complete a valid active event at end-of-stream; discard candidates.

        Usage: Inference only.
        """

        self._candidate = None
        event = self._finish_active()
        return () if event is None else (event,)
