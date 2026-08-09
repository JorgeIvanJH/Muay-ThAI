"""Run real-time LightGBM action inference on a video or webcam."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np

from models.action_detection.inference import (
    ActionModelRuntime,
    build_inference_parser,
    run_action_inference,
    validate_model_bundle,
)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
TASK_GUARD_WEIGHTS = WEIGHTS_DIR / "lightgbm_guard.joblib"
LEGACY_GUARD_WEIGHTS = WEIGHTS_DIR / "lightgbm_action.joblib"
DEFAULT_GUARD_WEIGHTS = (
    TASK_GUARD_WEIGHTS
    if TASK_GUARD_WEIGHTS.is_file()
    else LEGACY_GUARD_WEIGHTS
)
DEFAULT_STRIKING_WEIGHTS = WEIGHTS_DIR / "lightgbm_striking.joblib"


def load_runtime(
    weights: Path,
    *,
    classification_task: str,
) -> ActionModelRuntime:
    if not weights.is_file():
        raise FileNotFoundError(f"LightGBM weights not found: {weights}")

    bundle = joblib.load(weights)
    model = bundle["model"]
    class_names = validate_model_bundle(
        bundle,
        classification_task=classification_task,
        source=weights,
    )

    def predict_probabilities(window: np.ndarray) -> np.ndarray:
        probabilities = model.predict_proba(window.reshape(1, -1))[0]
        aligned = np.zeros(len(class_names), dtype=np.float32)
        for model_index, class_index in enumerate(model.classes_):
            aligned[int(class_index)] = probabilities[model_index]
        return aligned

    return ActionModelRuntime(
        model_name="LightGBM",
        classification_task=classification_task,
        class_names=class_names,
        window_size=int(bundle["window_size"]),
        confidence_threshold=float(bundle["confidence_threshold"]),
        coordinate_clip=float(bundle["coordinate_clip"]),
        predict_probabilities=predict_probabilities,
    )


def main() -> None:
    parser = build_inference_parser(
        description=(
            "Run guard and striking LightGBM inference together at 30 FPS CFR."
        ),
        default_guard_weights=DEFAULT_GUARD_WEIGHTS,
        default_striking_weights=DEFAULT_STRIKING_WEIGHTS,
    )
    args = parser.parse_args()
    guard_runtime = load_runtime(
        args.guard_weights,
        classification_task="guard",
    )
    striking_runtime = load_runtime(
        args.striking_weights,
        classification_task="striking",
    )

    run_action_inference(args, (guard_runtime, striking_runtime))


if __name__ == "__main__":
    main()
