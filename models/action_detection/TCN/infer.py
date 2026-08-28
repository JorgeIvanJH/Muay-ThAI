"""Run real-time TCN action inference on a video or webcam."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from models.action_detection.inference import (
    ActionModelRuntime,
    build_inference_parser,
    run_action_inference,
    validate_model_bundle,
)
from models.action_detection.TCN.model import TCNClassifier


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
TASK_GUARD_WEIGHTS = WEIGHTS_DIR / "tcn_guard.pt"
LEGACY_GUARD_WEIGHTS = WEIGHTS_DIR / "tcn_action.pt"
DEFAULT_GUARD_WEIGHTS = (
    TASK_GUARD_WEIGHTS
    if TASK_GUARD_WEIGHTS.is_file()
    else LEGACY_GUARD_WEIGHTS
)
DEFAULT_STRIKING_WEIGHTS = WEIGHTS_DIR / "tcn_striking.pt"


def _select_device(requested: str) -> torch.device:
    """
    Resolve the CPU or CUDA device used for TCN inference.

    Usage: Inference only.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_runtime(
    weights: Path,
    device: torch.device,
    *,
    classification_task: str,
) -> ActionModelRuntime:
    """
    Load one task-specific TCN bundle as a shared inference runtime.

    Usage: Inference only.
    """
    if not weights.is_file():
        raise FileNotFoundError(f"TCN weights not found: {weights}")

    bundle = torch.load(weights, map_location=device, weights_only=False)
    model = TCNClassifier(**bundle["model_config"])
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    class_names = validate_model_bundle(
        bundle,
        classification_task=classification_task,
        source=weights,
    )

    def predict_probabilities(window: np.ndarray) -> np.ndarray:
        """
        Predict a class-probability vector for one causal pose window.

        Usage: Inference only.
        """
        inputs = torch.from_numpy(window[None]).to(device)
        with torch.no_grad():
            logits = model(inputs)
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    return ActionModelRuntime(
        model_name="TCN",
        classification_task=classification_task,
        class_names=class_names,
        window_size=int(bundle["window_size"]),
        confidence_threshold=float(bundle["confidence_threshold"]),
        coordinate_clip=float(bundle["coordinate_clip"]),
        predict_probabilities=predict_probabilities,
    )


def main() -> None:
    """
    Load guard and striking TCNs and start shared video inference.

    Usage: Inference only.
    """
    parser = build_inference_parser(
        description=(
            "Run guard and striking TCN inference together at 30 FPS CFR."
        ),
        default_guard_weights=DEFAULT_GUARD_WEIGHTS,
        default_striking_weights=DEFAULT_STRIKING_WEIGHTS,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for the TCN classifier.",
    )
    args = parser.parse_args()
    device = _select_device(args.device)
    guard_runtime = load_runtime(
        args.guard_weights,
        device,
        classification_task="guard",
    )
    striking_runtime = load_runtime(
        args.striking_weights,
        device,
        classification_task="striking",
    )

    print(f"TCN device: {device}")
    run_action_inference(args, (guard_runtime, striking_runtime))


if __name__ == "__main__":
    main()
"""
conda run -n muay-thai python models/action_detection/TCN/infer.py --source 0 --display

"""
