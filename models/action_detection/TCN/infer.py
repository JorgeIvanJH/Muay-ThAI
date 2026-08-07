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
)
from models.action_detection.TCN.model import TCNClassifier


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "tcn_action.pt"


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_runtime(weights: Path, device: torch.device) -> ActionModelRuntime:
    if not weights.is_file():
        raise FileNotFoundError(f"TCN weights not found: {weights}")

    bundle = torch.load(weights, map_location=device, weights_only=False)
    model = TCNClassifier(**bundle["model_config"])
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    class_names = tuple(str(name) for name in bundle["class_names"])

    def predict_probabilities(window: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(window[None]).to(device)
        with torch.no_grad():
            logits = model(inputs)
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    return ActionModelRuntime(
        model_name="TCN",
        class_names=class_names,
        window_size=int(bundle["window_size"]),
        confidence_threshold=float(bundle["confidence_threshold"]),
        coordinate_clip=float(bundle["coordinate_clip"]),
        predict_probabilities=predict_probabilities,
    )


def main() -> None:
    parser = build_inference_parser(
        description="Run causal TCN action inference at 30 FPS CFR.",
        default_weights=DEFAULT_WEIGHTS,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for the TCN classifier.",
    )
    args = parser.parse_args()
    device = _select_device(args.device)
    runtime = load_runtime(args.weights, device)
    print(f"TCN device: {device}")
    run_action_inference(args, runtime)


if __name__ == "__main__":
    main()
    """
    conda run -n muay-thai python models/action_detection/TCN/infer.py `
    --source 0 `
    --device cuda `
    --yolo-device 0 `
    --display
    """