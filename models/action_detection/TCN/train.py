"""Train the causal TCN pose classifier."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.action_detection.TCN.model import TCNClassifier
from models.action_detection.preprocessing import (
    DEFAULT_DATASET_DIR,
    balanced_class_weights,
    build_window_dataset,
    choose_video_split,
    class_names_from_training,
    feature_names,
    load_pose_sequences,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "weights" / "tcn_action.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal TCN on normalized YOLO pose windows."
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
    parser.add_argument("--channels", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    targets = []
    predictions = []
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_features)
            loss = loss_function(logits, batch_targets)
            total_loss += loss.item() * len(batch_targets)
            targets.append(batch_targets.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")

    set_seed(args.seed)
    device = select_device(args.device)
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

    train_dataset = TensorDataset(
        torch.from_numpy(train_windows),
        torch.from_numpy(train_targets),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_windows),
        torch.from_numpy(validation_targets),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_config = {
        "input_size": train_windows.shape[2],
        "class_count": len(class_names),
        "channels": tuple(args.channels),
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
    }
    model = TCNClassifier(**model_config).to(device)
    class_weights = torch.from_numpy(
        balanced_class_weights(train_targets, len(class_names))
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Training videos: {', '.join(split.train_video_ids)}")
    print(f"Validation videos: {', '.join(split.validation_video_ids)}")
    if split.ignored_video_ids:
        print(f"Ignored videos: {', '.join(split.ignored_video_ids)}")
    print(f"Classes: {', '.join(class_names)}")
    print(
        f"Training samples: {len(train_dataset):,}; "
        f"validation samples: {len(validation_dataset):,}; "
        f"features per frame: {train_windows.shape[2]}"
    )

    best_macro_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = loss_function(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_targets)

        validation_loss, targets, predictions = evaluate(
            model, validation_loader, loss_function, device
        )
        macro_f1 = f1_score(
            targets,
            predictions,
            labels=list(range(len(class_names))),
            average="macro",
            zero_division=0,
        )
        train_loss = total_train_loss / len(train_dataset)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    validation_loss, targets, predictions = evaluate(
        model, validation_loader, loss_function, device
    )
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0,
    )
    print(f"Best validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")
    print(
        classification_report(
            targets,
            predictions,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            zero_division=0,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in best_state.items()
            },
            "model_config": model_config,
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
    print(f"Saved weights to {args.output}")


if __name__ == "__main__":
    main()
    """
    conda run -n muay-thai python models/action_detection/TCN/train.py `
        --train-videos video_1 video_2 video_3 `
        --val-videos video_4 video_5
    """
