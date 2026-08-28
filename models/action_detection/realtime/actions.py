"""Shared dual-model execution over one normalized pose stream."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from models.action_detection.config import validate_task_labels
from models.action_detection.preprocessing import FEATURE_CHANNELS, JOINT_NAMES
from models.action_detection.realtime.pose import (
    SelectedPose,
    normalize_selected_pose,
)
from models.action_detection.realtime.windows import TemporalWindowBuffer


@dataclass(frozen=True)
class ActionModelRuntime:
    """Loaded classifier and its training-time preprocessing settings."""

    model_name: str
    classification_task: str
    class_names: tuple[str, ...]
    window_size: int
    confidence_threshold: float
    coordinate_clip: float
    predict_probabilities: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ActionPrediction:
    """One task model's probability result for the current frame."""

    classification_task: str
    model_name: str
    class_names: tuple[str, ...]
    class_name: str
    probability: float
    probabilities: np.ndarray


class DualActionPredictor:
    """Maintain guard/striking histories and execute both classifiers."""

    def __init__(self, runtimes: Sequence[ActionModelRuntime]) -> None:
        """Validate two task runtimes and allocate their temporal buffers.

        Usage: Inference only.
        """

        self.runtimes = tuple(runtimes)
        tasks = tuple(runtime.classification_task for runtime in self.runtimes)
        if set(tasks) != {"guard", "striking"} or len(self.runtimes) != 2:
            raise ValueError(
                "Inference requires exactly one guard and one striking runtime"
            )

        feature_count = len(JOINT_NAMES) * len(FEATURE_CHANNELS)
        self._histories: dict[str, TemporalWindowBuffer] = {}
        for runtime in self.runtimes:
            if runtime.window_size < 1:
                raise ValueError(
                    f"{runtime.model_name} window_size must be at least 1"
                )
            validate_task_labels(
                runtime.classification_task,
                runtime.class_names,
                source=runtime.model_name,
            )
            self._histories[runtime.classification_task] = TemporalWindowBuffer(
                runtime.window_size,
                feature_count,
            )

    def predict(self, pose: SelectedPose) -> tuple[ActionPrediction, ...]:
        """Normalize one pose and return guard and striking predictions.

        Usage: Inference only.

        Normalized vectors are cached by preprocessing configuration so models
        trained with identical settings share exactly one normalization pass.
        """

        normalized: dict[tuple[float, float], np.ndarray] = {}
        predictions = []
        for runtime in self.runtimes:
            preprocessing_key = (
                runtime.confidence_threshold,
                runtime.coordinate_clip,
            )
            current_features = normalized.get(preprocessing_key)
            if current_features is None:
                current_features = normalize_selected_pose(
                    pose,
                    confidence_threshold=runtime.confidence_threshold,
                    coordinate_clip=runtime.coordinate_clip,
                )
                normalized[preprocessing_key] = current_features

            history = self._histories[runtime.classification_task]
            history.append(current_features)
            probabilities = np.asarray(
                runtime.predict_probabilities(history.current_window()),
                dtype=np.float32,
            ).reshape(-1)
            self._validate_probabilities(runtime, probabilities)
            class_index = int(np.argmax(probabilities))
            predictions.append(
                ActionPrediction(
                    classification_task=runtime.classification_task,
                    model_name=runtime.model_name,
                    class_names=runtime.class_names,
                    class_name=runtime.class_names[class_index],
                    probability=float(probabilities[class_index]),
                    probabilities=probabilities,
                )
            )
        return tuple(predictions)

    @staticmethod
    def _validate_probabilities(
        runtime: ActionModelRuntime,
        probabilities: np.ndarray,
    ) -> None:
        """Reject malformed output before it reaches analytics or persistence.

        Usage: Inference only.
        """

        expected_shape = (len(runtime.class_names),)
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"{runtime.model_name} returned {probabilities.shape} "
                f"probabilities for {len(runtime.class_names)} classes"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                f"{runtime.model_name} returned non-finite probabilities"
            )
