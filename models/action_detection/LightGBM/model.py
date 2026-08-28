"""LightGBM model definition and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMClassifier


def build_model(
    *,
    class_count: int,
    random_state: int = 42,
    n_estimators: int = 500,
) -> LGBMClassifier:
    """
    Construct the configured LightGBM classifier baseline.

    Usage: Training only.
    """
    if class_count < 2:
        raise ValueError("At least two classes are required")

    objective = "binary" if class_count == 2 else "multiclass"
    parameters: dict[str, Any] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if class_count > 2:
        parameters["num_class"] = class_count
    return LGBMClassifier(**parameters)


def save_model_bundle(bundle: dict[str, Any], output_path: Path | str) -> Path:
    """
    Save a fitted LightGBM model and preprocessing metadata with Joblib.

    Usage: Training only.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path
