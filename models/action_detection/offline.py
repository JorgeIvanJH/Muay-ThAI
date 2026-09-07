"""Sequential whole-video inference for offline use (evaluation, pre-annotation).

Usage: Inference only.

The real-time pipeline in ``inference.py`` is built for a live stream: bounded
queues, several threads, analytics and display. Pre-annotation and evaluation
need something simpler: run YOLO and the two action classifiers over every frame
of one file, in order, and return per-frame class probabilities. This module
does exactly that with the same building blocks the live pipeline uses:

- ``extract_largest_pose`` selects the person (``realtime/pose.py``).
- ``DualActionPredictor`` normalises the pose, keeps the causal window and runs
  both classifiers (``realtime/actions.py``).

Because the preprocessing objects are shared, features are identical to
training and to live inference; ``tests/test_action_realtime.py`` guards that
equivalence.

Frame numbering: arrays are indexed by 0-based OpenCV frame index. Label Studio
timelines are 1-based, so ``probabilities_to_ranges`` emits
``start = first_index + 1`` and ``end = last_index + 1`` (inclusive), matching
``dataset/build_action_joint_dataset.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from models.action_detection.realtime.actions import (
    ActionModelRuntime,
    DualActionPredictor,
)
from models.action_detection.realtime.pose import extract_largest_pose


TARGET_FPS = 30.0

# Same Ultralytics arguments as ``inference.py`` and the dataset builder, so the
# pose distribution seen here matches the one the classifiers were trained on.
DEFAULT_POSE_PREDICT_ARGUMENTS: dict[str, object] = {
    "verbose": False,
    "conf": 0.25,
    "imgsz": 640,
}


@dataclass(frozen=True)
class VideoProbabilities:
    """Per-frame class probabilities for every task model on one video."""

    video_path: Path
    frame_count: int
    fps: float
    class_names: dict[str, tuple[str, ...]]
    probabilities: dict[str, np.ndarray]  # task -> [frame_count, n_classes]

    def argmax_labels(self, task: str) -> list[str]:
        """Return the most probable class name for every frame of one task.

        Usage: Inference only.
        """

        names = self.class_names[task]
        indices = np.argmax(self.probabilities[task], axis=1)
        return [names[int(index)] for index in indices]


@dataclass(frozen=True)
class LabelRange:
    """One run of consecutive frames sharing a label, 1-based inclusive."""

    start: int
    end: int
    label: str
    score: float


def pose_predict_arguments(device: str | int | None = None) -> dict[str, object]:
    """Build Ultralytics keyword arguments, optionally pinning a device.

    Usage: Inference only.
    """

    arguments = dict(DEFAULT_POSE_PREDICT_ARGUMENTS)
    if device is not None and str(device) != "":
        arguments["device"] = device
    return arguments


def open_cfr_video(video_path: Path | str) -> cv2.VideoCapture:
    """Open a video and require the 30 FPS constant frame rate the project assumes.

    Usage: Inference only.

    Label Studio derives timeline frames from ``frameRate="30.0"``; a video at
    another rate would misalign every predicted range, so this fails loudly
    like ``build_action_joint_dataset.py`` does.
    """

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = TARGET_FPS
    if abs(fps - TARGET_FPS) > 1e-3:
        capture.release()
        raise ValueError(
            f"Expected a {TARGET_FPS:g} FPS constant-frame-rate video, got "
            f"{fps:.3f} FPS for {path.name}. Normalise it with "
            "media/videos/preprocess_fps.sh first."
        )
    return capture


def predict_video_probabilities(
    video_path: Path | str,
    *,
    pose_model: Callable,
    runtimes: Sequence[ActionModelRuntime],
    predict_arguments: Mapping[str, object] | None = None,
    max_frames: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> VideoProbabilities:
    """Run pose + both action classifiers over every frame of one video.

    Usage: Inference only.

    A fresh ``DualActionPredictor`` is created per call so the causal window
    starts empty for every video, exactly as in training (left zero padding).
    ``pose_model`` is called once per frame and must be the only user of the
    YOLO predictor while this runs.
    """

    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    arguments = dict(
        DEFAULT_POSE_PREDICT_ARGUMENTS
        if predict_arguments is None
        else predict_arguments
    )
    predictor = DualActionPredictor(runtimes)
    capture = open_cfr_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or TARGET_FPS
    collected: dict[str, list[np.ndarray]] = {
        runtime.classification_task: [] for runtime in predictor.runtimes
    }
    class_names = {
        runtime.classification_task: runtime.class_names
        for runtime in predictor.runtimes
    }
    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            result = pose_model(frame, **arguments)[0]
            pose = extract_largest_pose(result, frame.shape)
            for prediction in predictor.predict(pose):
                collected[prediction.classification_task].append(
                    np.array(prediction.probabilities, dtype=np.float32)
                )
            frame_index += 1
            if progress is not None:
                progress(frame_index)
    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames could be decoded from {video_path}")
    probabilities = {
        task: np.stack(rows, axis=0) for task, rows in collected.items()
    }
    return VideoProbabilities(
        video_path=Path(video_path),
        frame_count=frame_index,
        fps=fps,
        class_names=class_names,
        probabilities=probabilities,
    )


def probabilities_to_ranges(
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> list[LabelRange]:
    """Merge per-frame argmax labels into contiguous 1-based inclusive ranges.

    Usage: Inference only.

    Every frame receives exactly one label (``background`` included), so the
    ranges are non-overlapping and cover ``1..frame_count`` without gaps, which
    is what the dataset builder requires of an export. ``score`` is the mean
    probability of the run's label over its frames.
    """

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(class_names):
        raise ValueError(
            f"probabilities must have shape [frames, {len(class_names)}], got "
            f"{values.shape}"
        )
    if values.shape[0] == 0:
        return []

    labels = np.argmax(values, axis=1)
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(labels)]))  # exclusive
    ranges = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        class_index = int(labels[start])
        ranges.append(
            LabelRange(
                start=start + 1,
                end=end,
                label=str(class_names[class_index]),
                score=float(np.mean(values[start:end, class_index])),
            )
        )
    return ranges
