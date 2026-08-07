"""Train the LightGBM pose-window baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, classification_report, f1_score

from models.action_detection.LightGBM.model import (
    build_model,
    save_model_bundle,
)
from models.action_detection.preprocessing import (
    DEFAULT_DATASET_DIR,
    build_window_dataset,
    choose_video_split,
    class_names_from_training,
    feature_names,
    load_pose_sequences,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "weights" / "lightgbm_action.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM on causal windows of normalized YOLO poses."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--coordinate-clip", type=float, default=5.0)
    parser.add_argument(
        "--train-videos",
        nargs="+",
        metavar="VIDEO",
    )
    parser.add_argument(
        "--val-videos",
        "--val-video",
        dest="validation_videos",
        nargs="+",
        metavar="VIDEO",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = load_pose_sequences(
        args.dataset_dir,
        confidence_threshold=args.confidence_threshold,
        coordinate_clip=args.coordinate_clip,
    )
    split = choose_video_split(
        sequences,
        train_video_ids=args.train_videos,
        validation_video_ids=args.validation_videos,
        validation_fraction=args.validation_fraction,
    )
    class_names = class_names_from_training(
        sequences, split.train_video_ids
    )

    train_windows, train_targets = build_window_dataset(
        sequences,
        split.train_video_ids,
        window_size=args.window_size,
        class_names=class_names,
    )
    validation_windows, validation_targets = build_window_dataset(
        sequences,
        split.validation_video_ids,
        window_size=args.window_size,
        class_names=class_names,
    )

    train_features = train_windows.reshape(len(train_windows), -1)
    validation_features = validation_windows.reshape(
        len(validation_windows), -1
    )

    print(f"Training videos: {', '.join(split.train_video_ids)}")
    print(f"Validation videos: {', '.join(split.validation_video_ids)}")
    if split.ignored_video_ids:
        print(f"Ignored videos: {', '.join(split.ignored_video_ids)}")
    print(f"Classes: {', '.join(class_names)}")
    print(
        f"Training samples: {len(train_targets):,}; "
        f"validation samples: {len(validation_targets):,}; "
        f"flattened features: {train_features.shape[1]:,}"
    )

    model = build_model(
        class_count=len(class_names),
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    model.fit(
        train_features,
        train_targets,
        eval_set=[(validation_features, validation_targets)],
        eval_metric=(
            "binary_logloss" if len(class_names) == 2 else "multi_logloss"
        ),
        callbacks=[early_stopping(30), log_evaluation(20)],
    )

    predictions = model.predict(validation_features).astype(np.int64)
    accuracy = accuracy_score(validation_targets, predictions)
    macro_f1 = f1_score(
        validation_targets,
        predictions,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0,
    )
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")
    print(
        classification_report(
            validation_targets,
            predictions,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            zero_division=0,
        )
    )

    output_path = save_model_bundle(
        {
            "model": model,
            "class_names": class_names,
            "feature_names": feature_names(),
            "window_size": args.window_size,
            "confidence_threshold": args.confidence_threshold,
            "coordinate_clip": args.coordinate_clip,
            "train_video_ids": split.train_video_ids,
            "validation_video_ids": split.validation_video_ids,
        },
        args.output,
    )
    print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    main()
