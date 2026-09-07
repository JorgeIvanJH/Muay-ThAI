import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from models.action_detection.offline import (
    LabelRange,
    open_cfr_video,
    predict_video_probabilities,
    probabilities_to_ranges,
)
from models.action_detection.preprocessing import FEATURE_CHANNELS, JOINT_NAMES
from models.action_detection.realtime.actions import ActionModelRuntime


FEATURE_COUNT = len(JOINT_NAMES) * len(FEATURE_CHANNELS)


class _EmptyResult:
    """Mimic an Ultralytics result with no detections."""

    boxes = None
    keypoints = None


class _FakePoseModel:
    """Count calls and return no person for every frame."""

    def __init__(self) -> None:
        self.calls = 0
        self.arguments = None

    def __call__(self, frame, **arguments):
        self.calls += 1
        self.arguments = arguments
        return [_EmptyResult()]


def _constant_runtime(task: str, class_names: tuple[str, ...], window_size: int):
    def predict_probabilities(window: np.ndarray) -> np.ndarray:
        assert window.shape == (window_size, FEATURE_COUNT)
        values = np.zeros(len(class_names), dtype=np.float32)
        values[0] = 1.0
        return values

    return ActionModelRuntime(
        model_name=f"fake-{task}",
        classification_task=task,
        class_names=class_names,
        window_size=window_size,
        confidence_threshold=0.25,
        coordinate_clip=5.0,
        predict_probabilities=predict_probabilities,
    )


def _write_clip(path: Path, *, fps: float, frames: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    try:
        for index in range(frames):
            frame = np.full((48, 64, 3), index * 3 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class ProbabilitiesToRangesTests(unittest.TestCase):
    def test_merges_runs_into_one_based_inclusive_ranges(self) -> None:
        names = ("background", "guard_down", "guard_up")
        probabilities = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.10, 0.10],
                [0.1, 0.80, 0.10],
                [0.2, 0.60, 0.20],
                [0.2, 0.60, 0.20],
                [0.1, 0.10, 0.80],
                [0.7, 0.20, 0.10],
            ],
            dtype=np.float32,
        )

        ranges = probabilities_to_ranges(probabilities, names)

        self.assertEqual(
            [(r.start, r.end, r.label) for r in ranges],
            [
                (1, 2, "background"),
                (3, 5, "guard_down"),
                (6, 6, "guard_up"),
                (7, 7, "background"),
            ],
        )
        self.assertAlmostEqual(ranges[0].score, 0.85, places=6)
        self.assertAlmostEqual(ranges[1].score, (0.8 + 0.6 + 0.6) / 3, places=6)

    def test_ranges_cover_every_frame_without_gaps_or_overlaps(self) -> None:
        rng = np.random.default_rng(7)
        names = ("background", "punch", "elbow", "kick", "knee")
        probabilities = rng.random((500, len(names)), dtype=np.float32)

        ranges = probabilities_to_ranges(probabilities, names)

        self.assertEqual(ranges[0].start, 1)
        self.assertEqual(ranges[-1].end, 500)
        for previous, current in zip(ranges, ranges[1:]):
            self.assertEqual(current.start, previous.end + 1)
            self.assertNotEqual(current.label, previous.label)
        covered = sum(r.end - r.start + 1 for r in ranges)
        self.assertEqual(covered, 500)
        self.assertTrue(all(isinstance(r, LabelRange) for r in ranges))

    def test_rejects_mismatched_class_count_and_handles_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            probabilities_to_ranges(np.zeros((3, 2), dtype=np.float32), ("a", "b", "c"))
        self.assertEqual(
            probabilities_to_ranges(np.zeros((0, 3), dtype=np.float32), ("a", "b", "c")),
            [],
        )


class PredictVideoProbabilitiesTests(unittest.TestCase):
    def test_runs_both_models_on_every_decoded_frame(self) -> None:
        runtimes = (
            _constant_runtime("guard", ("background", "guard_down", "guard_up"), 4),
            _constant_runtime("striking", ("background", "elbow", "kick", "knee", "punch"), 8),
        )
        pose_model = _FakePoseModel()
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip_30fps.mp4"
            _write_clip(clip, fps=30.0, frames=12)

            result = predict_video_probabilities(
                clip,
                pose_model=pose_model,
                runtimes=runtimes,
                predict_arguments={"verbose": False, "device": "cpu"},
                max_frames=10,
            )

        self.assertEqual(result.frame_count, 10)
        self.assertEqual(pose_model.calls, 10)
        self.assertEqual(pose_model.arguments, {"verbose": False, "device": "cpu"})
        self.assertAlmostEqual(result.fps, 30.0, places=3)
        self.assertEqual(result.probabilities["guard"].shape, (10, 3))
        self.assertEqual(result.probabilities["striking"].shape, (10, 5))
        self.assertEqual(result.argmax_labels("guard"), ["background"] * 10)
        self.assertEqual(
            result.class_names["striking"],
            ("background", "elbow", "kick", "knee", "punch"),
        )

    def test_rejects_videos_that_are_not_30_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip_25fps.mp4"
            _write_clip(clip, fps=25.0, frames=5)
            with self.assertRaises(ValueError):
                open_cfr_video(clip)


if __name__ == "__main__":
    unittest.main()
