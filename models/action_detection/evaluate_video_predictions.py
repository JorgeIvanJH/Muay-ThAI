"""Compare TCN pre-annotations with manual Label Studio labels, frame by frame.

Usage: Evaluation only.

This runs exactly what the Label Studio pre-annotation backend runs
(``offline.predict_video_probabilities`` with the default TCN bundles and YOLO
settings), converts the argmax into per-frame labels and compares them with the
1-based ranges of a Label Studio JSON-MIN export. It reports per-class
precision/recall/F1, macro F1, accuracy and the wall-clock cost per frame.

Example:

    conda run -n muay-thai python models/action_detection/evaluate_video_predictions.py \
        --task guard --video-id 20260728_130155_30fps

Only frames that OpenCV can decode are compared. Label Studio's timeline may
show a few more frames than OpenCV decodes; those trailing labels are ignored.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from ultralytics import YOLO

from models.action_detection.config import class_names_for_task
from models.action_detection.offline import (
    LabelRange,
    pose_predict_arguments,
    predict_video_probabilities,
    probabilities_to_ranges,
)
from models.action_detection.TCN.infer import (
    DEFAULT_GUARD_WEIGHTS,
    DEFAULT_STRIKING_WEIGHTS,
    load_runtime,
)
from models.yolo import config as yolocfg


DATASET_BUILDER = ROOT_DIR / "dataset" / "build_action_joint_dataset.py"


def _load_dataset_builder():
    """Import the dataset builder by path so its export parsing is reused.

    Usage: Evaluation only.
    """

    spec = importlib.util.spec_from_file_location(
        "build_action_joint_dataset",
        DATASET_BUILDER,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=("guard", "striking"), required=True)
    parser.add_argument(
        "--video-id",
        required=True,
        help="Video stem as used in the CSVs, e.g. 20260728_130155_30fps.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Label Studio JSON-MIN export; default: the single export in "
        "dataset/classification/<task>/.",
    )
    parser.add_argument("--guard-weights", type=Path, default=DEFAULT_GUARD_WEIGHTS)
    parser.add_argument(
        "--striking-weights",
        type=Path,
        default=DEFAULT_STRIKING_WEIGHTS,
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=yolocfg.YOLO_REALTIME_WEIGHTS,
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Ultralytics device, e.g. cpu or 0 (default: Ultralytics choice).",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "output",
        help="Folder for the JSON report.",
    )
    return parser.parse_args()


def _find_task(tasks: list[dict], video_id: str, builder) -> dict:
    """Pick the export task whose video resolves to ``video_id``.

    Usage: Evaluation only.
    """

    video_dir = builder.DEFAULT_VIDEO_DIR
    for task in tasks:
        try:
            path = builder.resolve_video_path(task["video"], video_dir)
        except FileNotFoundError:
            continue
        if path.stem == video_id:
            return task
    raise SystemExit(f"No task in the export resolves to video {video_id!r}")


def _metrics(
    truth: list[str],
    predicted: list[str],
    class_names: tuple[str, ...],
) -> dict:
    """Compute confusion matrix, per-class P/R/F1, macro F1 and accuracy.

    Usage: Evaluation only.
    """

    index = {name: i for i, name in enumerate(class_names)}
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        confusion[index[actual], index[guess]] += 1
    per_class = {}
    f1_values = []
    for name, i in index.items():
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[i, :].sum()),
        }
        f1_values.append(f1)
    return {
        "accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(class_names),
            "rows_true_cols_pred": confusion.tolist(),
        },
    }


def _print_report(report: dict) -> None:
    class_names = report["metrics"]["confusion_matrix"]["labels"]
    print(f"\nTask {report['task']}  video {report['video_id']}")
    print(
        f"Frames compared: {report['frames_compared']:,}  "
        f"(decoded {report['frames_decoded']:,}, labelled to "
        f"{report['last_labelled_frame']:,})"
    )
    print(
        f"Wall time: {report['latency']['total_seconds']:.1f} s  "
        f"({report['latency']['ms_per_frame']:.1f} ms/frame, device "
        f"{report['latency']['yolo_device']})"
    )
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Macro F1: {report['metrics']['macro_f1']:.4f}")
    print(f"\n{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for name in class_names:
        row = report["metrics"]["per_class"][name]
        print(
            f"{name:<12}{row['precision']:>10.3f}{row['recall']:>10.3f}"
            f"{row['f1']:>10.3f}{row['support']:>10,}"
        )
    print("\nConfusion matrix (rows = manual, columns = predicted):")
    width = max(len(name) for name in class_names) + 2
    print(" " * width + "".join(f"{name:>{width}}" for name in class_names))
    for name, row in zip(
        class_names,
        report["metrics"]["confusion_matrix"]["rows_true_cols_pred"],
    ):
        print(f"{name:<{width}}" + "".join(f"{value:>{width},}" for value in row))
    print(f"\nPredicted ranges: {report['predicted_range_count']}")


def main() -> None:
    args = parse_args()
    builder = _load_dataset_builder()
    annotations = args.annotations or builder.resolve_annotations_path(
        args.task,
        None,
    )
    tasks = builder.load_annotation_tasks(annotations)
    task = _find_task(tasks, args.video_id, builder)
    frame_labels, last_labelled_frame = builder.build_frame_labels(task)
    video_path = builder.resolve_video_path(task["video"], builder.DEFAULT_VIDEO_DIR)

    device = torch.device("cpu")
    runtimes = (
        load_runtime(args.guard_weights, device, classification_task="guard"),
        load_runtime(args.striking_weights, device, classification_task="striking"),
    )
    pose_model = YOLO(str(args.pose_model))
    predict_arguments = pose_predict_arguments(args.yolo_device)

    print(f"Running {video_path.name} ...", flush=True)
    started = time.perf_counter()
    result = predict_video_probabilities(
        video_path,
        pose_model=pose_model,
        runtimes=runtimes,
        predict_arguments=predict_arguments,
        max_frames=args.max_frames,
    )
    elapsed = time.perf_counter() - started

    predicted = result.argmax_labels(args.task)
    compared_frames = min(result.frame_count, last_labelled_frame)
    truth = [frame_labels[i + 1] for i in range(compared_frames)]
    predicted = predicted[:compared_frames]
    class_names = class_names_for_task(args.task)
    ranges: list[LabelRange] = probabilities_to_ranges(
        result.probabilities[args.task],
        result.class_names[args.task],
    )

    report = {
        "task": args.task,
        "video_id": args.video_id,
        "video_path": str(video_path),
        "annotations": str(annotations),
        "weights": str(args.guard_weights if args.task == "guard" else args.striking_weights),
        "pose_model": str(args.pose_model),
        "frames_decoded": result.frame_count,
        "last_labelled_frame": last_labelled_frame,
        "frames_compared": compared_frames,
        "latency": {
            "total_seconds": elapsed,
            "ms_per_frame": 1000.0 * elapsed / result.frame_count,
            "yolo_device": str(predict_arguments.get("device", "auto")),
        },
        "metrics": _metrics(truth, predicted, class_names),
        "predicted_range_count": len(ranges),
    }
    _print_report(report)

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / f"preannotation_eval_{args.task}_{args.video_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
