import unittest

from models.action_detection.analytics.anthropometry import (
    AnthropometricScaleEstimator,
)
from models.action_detection.analytics.strike_events import (
    StrikeEventStateMachine,
    StrikeObservation,
)
from models.action_detection.analytics.strike_speed import estimate_strike_speed
from models.action_detection.analytics.types import JointPoint, SpeedSample


def _limb_speeds(
    strike_type: str = "punch",
    left: float | None = 2.0,
    right: float | None = 0.1,
) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            "left": left if name == strike_type else 0.0,
            "right": right if name == strike_type else 0.0,
        }
        for name in ("punch", "elbow", "kick", "knee")
    }


def _observation(
    frame: int,
    timestamp: float,
    *,
    punch_probability: float,
    left_speed: float,
) -> StrikeObservation:
    return StrikeObservation(
        frame_index=frame,
        timestamp=timestamp,
        probabilities={
            "background": 1.0 - punch_probability,
            "punch": punch_probability,
            "elbow": 0.0,
            "kick": 0.0,
            "knee": 0.0,
        },
        limb_speeds_mps=_limb_speeds(left=left_speed),
    )


class AnthropometryTests(unittest.TestCase):
    def test_known_projected_segment_produces_expected_scale(self) -> None:
        estimator = AnthropometricScaleEstimator(
            2.0,
            smoothing_alpha=1.0,
        )
        # A 2 m person's canonical upper arm is 0.372 m. At 500 px/m it
        # projects to 186 px.
        scale = estimator.update(
            {
                "left_shoulder": JointPoint(0.0, 0.0, 1.0),
                "left_elbow": JointPoint(186.0, 0.0, 1.0),
            }
        )
        self.assertAlmostEqual(scale, 500.0)


class StrikeStateMachineTests(unittest.TestCase):
    def test_emits_one_limb_locked_event_after_release(self) -> None:
        machine = StrikeEventStateMachine()
        completed = []
        observations = (
            _observation(0, 0.000, punch_probability=0.90, left_speed=2.0),
            _observation(1, 0.033, punch_probability=0.90, left_speed=2.5),
            _observation(2, 0.067, punch_probability=0.80, left_speed=3.0),
            _observation(3, 0.100, punch_probability=0.70, left_speed=1.0),
            _observation(4, 0.133, punch_probability=0.10, left_speed=0.1),
            _observation(7, 0.233, punch_probability=0.10, left_speed=0.1),
        )
        for observation in observations:
            completed.extend(machine.update(observation))

        self.assertEqual(len(completed), 1)
        event = completed[0]
        self.assertEqual((event.strike_type, event.side), ("punch", "left"))
        self.assertEqual(event.start_frame, 0)
        self.assertEqual(event.apex_frame, 2)
        self.assertEqual(event.end_frame, 3)
        self.assertLessEqual(event.apex_frame, event.end_frame)
        self.assertAlmostEqual(event.peak_classification_confidence, 0.90)

    def test_one_frame_probability_spike_is_rejected(self) -> None:
        machine = StrikeEventStateMachine()
        machine.update(
            _observation(0, 0.000, punch_probability=0.90, left_speed=2.0)
        )
        machine.update(
            _observation(1, 0.033, punch_probability=0.10, left_speed=0.1)
        )
        self.assertEqual(machine.flush(), ())

    def test_motion_valley_separates_events_while_class_stays_confident(self) -> None:
        machine = StrikeEventStateMachine()
        completed = []
        observations = (
            _observation(0, 0.000, punch_probability=0.90, left_speed=2.0),
            _observation(1, 0.033, punch_probability=0.90, left_speed=2.0),
            _observation(2, 0.100, punch_probability=0.90, left_speed=1.0),
            _observation(3, 0.133, punch_probability=0.90, left_speed=0.1),
            _observation(7, 0.233, punch_probability=0.90, left_speed=0.1),
        )
        for observation in observations:
            completed.extend(machine.update(observation))

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].end_frame, 2)


class StrikeSpeedTests(unittest.TestCase):
    def test_quadratic_fit_refines_a_peak_between_frames(self) -> None:
        true_peak_time = 0.010
        samples = tuple(
            SpeedSample(
                timestamp=timestamp,
                speed_mps=6.0 - 100.0 * (timestamp - true_peak_time) ** 2,
            )
            for timestamp in (-0.066, -0.033, 0.000, 0.033, 0.066)
        )
        estimate = estimate_strike_speed(samples)

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertGreaterEqual(
            estimate.interpolated_peak_mps,
            estimate.sampled_peak_mps,
        )
        self.assertAlmostEqual(estimate.interpolated_peak_mps, 6.0, places=6)
        self.assertAlmostEqual(
            estimate.interpolated_peak_timestamp,
            true_peak_time,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
