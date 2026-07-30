"""Shared pose preprocessing for action-detection models.

The raw dataset contains one row per YOLO detection per frame. This module
selects the largest detected person, converts pixel coordinates to a
body-centred representation, and creates causal pose windows. Both the
LightGBM and TCN models consume these exact features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ROOT_DIR / "dataset" / "jointswithactionlabels"

JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
LEFT_SHOULDER_INDEX = JOINT_NAMES.index("left_shoulder")
RIGHT_SHOULDER_INDEX = JOINT_NAMES.index("right_shoulder")
LEFT_HIP_INDEX = JOINT_NAMES.index("left_hip")
RIGHT_HIP_INDEX = JOINT_NAMES.index("right_hip")
FEATURE_CHANNELS = ("x_body", "y_body", "confidence", "valid")
FRAME_KEYS = ["video_id", "frame_index"]


@dataclass(frozen=True)
class PoseSequence:
    """Preprocessed frames and labels for one complete video."""

    video_id: str
    features: np.ndarray
    labels: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class VideoSplit:
    """Whole-video training and validation split."""

    train_video_ids: tuple[str, str]
    validation_video_id: str
    ignored_video_ids: tuple[str, ...] = ()


def feature_names() -> list[str]:
    return [
        f"{joint_name}_{channel}"
        for joint_name in JOINT_NAMES
        for channel in FEATURE_CHANNELS
    ]


def required_raw_columns() -> set[str]:
    columns = {
        "video_id",
        "frame_index",
        "time_seconds",
        "action_label",
        "pose_detected",
        "bbox_detected",
        "detection_index",
        "frame_width_px",
        "frame_height_px",
        "bbox_confidence",
        "bbox_x1_px",
        "bbox_y1_px",
        "bbox_x2_px",
        "bbox_y2_px",
    }
    for joint_name in JOINT_NAMES:
        columns.update(
            {
                f"{joint_name}_x_px",
                f"{joint_name}_y_px",
                f"{joint_name}_confidence",
            }
        )
    return columns


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def validate_raw_schema(frames: pd.DataFrame, source: Path) -> None:
    missing = sorted(required_raw_columns() - set(frames.columns))
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"{source} does not use the current raw pose schema. Missing: "
            f"{preview}. Regenerate it with "
            f"'python dataset/build_action_joint_dataset.py --overwrite'."
        )

    conflicting = (
        frames.groupby(FRAME_KEYS, sort=False)["action_label"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicting.any():
        example = conflicting[conflicting].index[0]
        raise ValueError(
            f"{source} has conflicting labels for video/frame {example}"
        )


def select_largest_person(frames: pd.DataFrame) -> pd.DataFrame:
    """Return one selected detection per video frame.

    Rows with a pose are preferred. Among them, the largest bounding-box area
    wins, followed by bounding-box confidence and the original detection index.
    A frame with no detections keeps its empty raw row.
    """

    if frames.empty:
        return frames.copy()

    ranked = frames.copy()
    width = (_numeric(ranked["bbox_x2_px"]) - _numeric(ranked["bbox_x1_px"])).clip(
        lower=0
    )
    height = (
        _numeric(ranked["bbox_y2_px"]) - _numeric(ranked["bbox_y1_px"])
    ).clip(lower=0)
    ranked["_bbox_area"] = (width * height).fillna(-1.0)
    ranked["_pose_priority"] = _numeric(ranked["pose_detected"]).fillna(0)
    ranked["_bbox_priority"] = _numeric(ranked["bbox_detected"]).fillna(0)
    ranked["_bbox_confidence_sort"] = _numeric(
        ranked["bbox_confidence"]
    ).fillna(-1.0)
    ranked["_detection_index_sort"] = _numeric(
        ranked["detection_index"]
    ).fillna(np.inf)

    ranked = ranked.sort_values(
        FRAME_KEYS
        + [
            "_pose_priority",
            "_bbox_priority",
            "_bbox_area",
            "_bbox_confidence_sort",
            "_detection_index_sort",
        ],
        ascending=[True, True, False, False, False, False, True],
        kind="stable",
    )
    selected = ranked.drop_duplicates(FRAME_KEYS, keep="first")
    return selected.drop(
        columns=[
            "_bbox_area",
            "_pose_priority",
            "_bbox_priority",
            "_bbox_confidence_sort",
            "_detection_index_sort",
        ]
    ).sort_values(FRAME_KEYS, kind="stable")


def _mean_available_pair(
    values: np.ndarray,
    valid: np.ndarray,
    first_index: int,
    second_index: int,
) -> np.ndarray:
    pair_values = values[:, [first_index, second_index]]
    pair_valid = valid[:, [first_index, second_index]]
    counts = pair_valid.sum(axis=1)
    totals = np.where(pair_valid, pair_values, 0.0).sum(axis=1)
    return np.divide(
        totals,
        counts,
        out=np.full(values.shape[0], np.nan, dtype=np.float32),
        where=counts > 0,
    )


def normalize_selected_frames(
    frames: pd.DataFrame,
    *,
    confidence_threshold: float = 0.25,
    coordinate_clip: float = 5.0,
) -> np.ndarray:
    """Convert selected raw poses to body-centred joint features."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if coordinate_clip <= 0:
        raise ValueError("coordinate_clip must be positive")

    frame_count = len(frames)
    joint_count = len(JOINT_NAMES)
    x = np.empty((frame_count, joint_count), dtype=np.float32)
    y = np.empty_like(x)
    confidence = np.empty_like(x)

    for joint_index, joint_name in enumerate(JOINT_NAMES):
        x[:, joint_index] = _numeric(frames[f"{joint_name}_x_px"]).to_numpy(
            dtype=np.float32
        )
        y[:, joint_index] = _numeric(frames[f"{joint_name}_y_px"]).to_numpy(
            dtype=np.float32
        )
        confidence[:, joint_index] = _numeric(
            frames[f"{joint_name}_confidence"]
        ).to_numpy(dtype=np.float32)

    pose_detected = (
        _numeric(frames["pose_detected"]).fillna(0).to_numpy(dtype=np.float32)
        > 0
    )
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(confidence)
        & (confidence >= confidence_threshold)
        & pose_detected[:, None]
    )

    hip_x = _mean_available_pair(
        x, valid, LEFT_HIP_INDEX, RIGHT_HIP_INDEX
    )
    hip_y = _mean_available_pair(
        y, valid, LEFT_HIP_INDEX, RIGHT_HIP_INDEX
    )
    shoulder_x = _mean_available_pair(
        x, valid, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX
    )
    shoulder_y = _mean_available_pair(
        y, valid, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX
    )

    valid_counts = valid.sum(axis=1)
    all_joint_x = np.divide(
        np.where(valid, x, 0.0).sum(axis=1),
        valid_counts,
        out=np.full(frame_count, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    )
    all_joint_y = np.divide(
        np.where(valid, y, 0.0).sum(axis=1),
        valid_counts,
        out=np.full(frame_count, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    )

    bbox_x1 = _numeric(frames["bbox_x1_px"]).to_numpy(dtype=np.float32)
    bbox_y1 = _numeric(frames["bbox_y1_px"]).to_numpy(dtype=np.float32)
    bbox_x2 = _numeric(frames["bbox_x2_px"]).to_numpy(dtype=np.float32)
    bbox_y2 = _numeric(frames["bbox_y2_px"]).to_numpy(dtype=np.float32)
    bbox_center_x = (bbox_x1 + bbox_x2) / 2.0
    bbox_center_y = (bbox_y1 + bbox_y2) / 2.0
    bbox_height = np.maximum(0.0, bbox_y2 - bbox_y1)

    center_x = np.where(
        np.isfinite(hip_x),
        hip_x,
        np.where(np.isfinite(all_joint_x), all_joint_x, bbox_center_x),
    )
    center_y = np.where(
        np.isfinite(hip_y),
        hip_y,
        np.where(np.isfinite(all_joint_y), all_joint_y, bbox_center_y),
    )

    torso_length = np.hypot(shoulder_x - hip_x, shoulder_y - hip_y)
    fallback_scale = bbox_height * 0.3
    scale = np.where(
        np.isfinite(torso_length) & (torso_length > 1.0),
        torso_length,
        fallback_scale,
    )
    scale = np.where(np.isfinite(scale) & (scale > 1.0), scale, 1.0)

    body_x = np.clip(
        (x - center_x[:, None]) / scale[:, None],
        -coordinate_clip,
        coordinate_clip,
    )
    body_y = np.clip(
        (y - center_y[:, None]) / scale[:, None],
        -coordinate_clip,
        coordinate_clip,
    )
    body_x = np.where(valid, body_x, 0.0)
    body_y = np.where(valid, body_y, 0.0)
    confidence = np.clip(
        np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )

    channels = np.stack(
        (body_x, body_y, confidence, valid.astype(np.float32)),
        axis=-1,
    )
    return channels.reshape(frame_count, -1).astype(np.float32, copy=False)


def load_pose_sequences(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    *,
    confidence_threshold: float = 0.25,
    coordinate_clip: float = 5.0,
) -> dict[str, PoseSequence]:
    """Load, select, and normalize every raw per-video CSV."""

    dataset_path = Path(dataset_dir)
    csv_paths = sorted(dataset_path.glob("*_joints_labels.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No '*_joints_labels.csv' files found in {dataset_path}"
        )

    sequences: dict[str, PoseSequence] = {}
    for csv_path in csv_paths:
        raw = pd.read_csv(csv_path)
        validate_raw_schema(raw, csv_path)
        selected = select_largest_person(raw)

        for video_id, video_frames in selected.groupby("video_id", sort=False):
            video_id = str(video_id)
            if video_id in sequences:
                raise ValueError(
                    f"Video {video_id!r} occurs in more than one CSV"
                )

            video_frames = video_frames.sort_values(
                "frame_index", kind="stable"
            ).reset_index(drop=True)
            frame_indices = _numeric(video_frames["frame_index"]).to_numpy(
                dtype=np.int64
            )
            if len(frame_indices) > 1 and not np.all(np.diff(frame_indices) == 1):
                raise ValueError(
                    f"Video {video_id!r} does not contain every consecutive frame"
                )

            sequences[video_id] = PoseSequence(
                video_id=video_id,
                features=normalize_selected_frames(
                    video_frames,
                    confidence_threshold=confidence_threshold,
                    coordinate_clip=coordinate_clip,
                ),
                labels=video_frames["action_label"].astype(str).to_numpy(),
                frame_indices=frame_indices,
                timestamps=_numeric(video_frames["time_seconds"]).to_numpy(
                    dtype=np.float32
                ),
            )

    return sequences


def choose_video_split(
    available_video_ids: Iterable[str],
    *,
    train_video_ids: Sequence[str] | None = None,
    validation_video_id: str | None = None,
) -> VideoSplit:
    """Choose exactly two complete videos for training and one for validation."""

    available = tuple(sorted(set(available_video_ids)))
    if len(available) < 3:
        raise ValueError("At least three videos are required")

    if train_video_ids is None and validation_video_id is None:
        train = (available[0], available[1])
        validation = available[2]
    elif train_video_ids is not None and validation_video_id is not None:
        if len(train_video_ids) != 2:
            raise ValueError("Exactly two --train-videos values are required")
        train = (str(train_video_ids[0]), str(train_video_ids[1]))
        validation = str(validation_video_id)
    else:
        raise ValueError(
            "Specify both train_video_ids and validation_video_id, or neither"
        )

    chosen = (*train, validation)
    if len(set(chosen)) != 3:
        raise ValueError("Training and validation videos must be distinct")
    unknown = sorted(set(chosen) - set(available))
    if unknown:
        raise ValueError(f"Unknown video IDs: {', '.join(unknown)}")

    ignored = tuple(video_id for video_id in available if video_id not in chosen)
    return VideoSplit(train, validation, ignored)


def class_names_from_training(
    sequences: dict[str, PoseSequence],
    train_video_ids: Sequence[str],
) -> tuple[str, ...]:
    labels = np.concatenate(
        [sequences[video_id].labels for video_id in train_video_ids]
    )
    return tuple(sorted(np.unique(labels).tolist()))


def encode_labels(labels: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    class_to_index = {
        class_name: class_index
        for class_index, class_name in enumerate(class_names)
    }
    unknown = sorted(set(labels.tolist()) - set(class_to_index))
    if unknown:
        raise ValueError(
            "Labels absent from the training videos: " + ", ".join(unknown)
        )
    return np.asarray(
        [class_to_index[label] for label in labels],
        dtype=np.int64,
    )


def causal_windows(features: np.ndarray, window_size: int) -> np.ndarray:
    """Create left-padded windows ending at each current frame."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if features.ndim != 2:
        raise ValueError("features must have shape [frames, features]")

    frame_count, feature_count = features.shape
    windows = np.zeros(
        (frame_count, window_size, feature_count),
        dtype=np.float32,
    )
    for frame_index in range(frame_count):
        first_frame = max(0, frame_index - window_size + 1)
        history = features[first_frame : frame_index + 1]
        windows[frame_index, -len(history) :] = history
    return windows


def build_window_dataset(
    sequences: dict[str, PoseSequence],
    video_ids: Sequence[str],
    *,
    window_size: int,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    targets = []
    for video_id in video_ids:
        sequence = sequences[video_id]
        windows.append(causal_windows(sequence.features, window_size))
        targets.append(encode_labels(sequence.labels, class_names))
    return np.concatenate(windows), np.concatenate(targets)


def balanced_class_weights(
    encoded_labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    counts = np.bincount(encoded_labels, minlength=class_count)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in the training videos")
    return (
        encoded_labels.size / (class_count * counts.astype(np.float64))
    ).astype(np.float32)
