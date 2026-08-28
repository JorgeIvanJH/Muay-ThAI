"""Fast numeric pose extraction and body-centred inference preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.action_detection.analytics.types import JointPoint
from models.action_detection.preprocessing import (
    FEATURE_CHANNELS,
    JOINT_NAMES,
    LEFT_HIP_INDEX,
    LEFT_SHOULDER_INDEX,
    RIGHT_HIP_INDEX,
    RIGHT_SHOULDER_INDEX,
)


@dataclass(frozen=True)
class SelectedPose:
    """The largest detected person's box and 17 COCO keypoints.

    Usage: Inference only.

    Missing keypoints and boxes contain ``NaN``. Arrays use compact fixed
    shapes so the real-time path does not need dictionaries or pandas.
    """

    pose_detected: bool
    people_detected: int
    boxes_detected: int
    detection_index: int | None
    frame_width_px: int
    frame_height_px: int
    bbox_xyxy: np.ndarray
    bbox_confidence: float
    keypoints_xy: np.ndarray
    keypoint_confidences: np.ndarray


def empty_pose(frame_shape: tuple[int, ...]) -> SelectedPose:
    """Create the fixed-shape representation used when YOLO finds no pose.

    Usage: Inference only.
    """

    height, width = frame_shape[:2]
    return SelectedPose(
        pose_detected=False,
        people_detected=0,
        boxes_detected=0,
        detection_index=None,
        frame_width_px=int(width),
        frame_height_px=int(height),
        bbox_xyxy=np.full(4, np.nan, dtype=np.float32),
        bbox_confidence=float("nan"),
        keypoints_xy=np.full((len(JOINT_NAMES), 2), np.nan, dtype=np.float32),
        keypoint_confidences=np.full(
            len(JOINT_NAMES),
            np.nan,
            dtype=np.float32,
        ),
    )


def extract_largest_pose(result, frame_shape: tuple[int, ...]) -> SelectedPose:
    """Select the largest YOLO pose without building per-frame DataFrames.

    Usage: Inference only.

    Bounding-box area is evaluated on the model tensor. Only the selected box
    and keypoints cross from the inference device to CPU memory.
    """

    height, width = frame_shape[:2]
    boxes = None if result.boxes is None else result.boxes.xyxy
    keypoints = None if result.keypoints is None else result.keypoints.data
    box_count = 0 if boxes is None else int(boxes.shape[0])
    people_count = 0 if keypoints is None else int(keypoints.shape[0])
    if people_count == 0:
        pose = empty_pose(frame_shape)
        return SelectedPose(
            **{
                **pose.__dict__,
                "boxes_detected": box_count,
            }
        )

    # Pose results normally align boxes and keypoints by detection index. If a
    # backend omits boxes, selecting the first pose is the only stable fallback.
    if boxes is not None and box_count:
        candidate_count = min(box_count, people_count)
        candidate_boxes = boxes[:candidate_count]
        widths = (candidate_boxes[:, 2] - candidate_boxes[:, 0]).clamp(min=0)
        heights = (candidate_boxes[:, 3] - candidate_boxes[:, 1]).clamp(min=0)
        selected_index = int((widths * heights).argmax().item())
    else:
        selected_index = 0

    selected_keypoints = (
        keypoints[selected_index].detach().float().cpu().numpy()
    )
    keypoints_xy = np.full((len(JOINT_NAMES), 2), np.nan, dtype=np.float32)
    confidences = np.full(len(JOINT_NAMES), np.nan, dtype=np.float32)
    available = min(len(JOINT_NAMES), selected_keypoints.shape[0])
    keypoints_xy[:available] = selected_keypoints[:available, :2]
    if selected_keypoints.shape[1] >= 3:
        confidences[:available] = selected_keypoints[:available, 2]

    bbox_xyxy = np.full(4, np.nan, dtype=np.float32)
    bbox_confidence = float("nan")
    if boxes is not None and selected_index < box_count:
        bbox_xyxy = boxes[selected_index].detach().float().cpu().numpy()
        if result.boxes.conf is not None:
            bbox_confidence = float(result.boxes.conf[selected_index].item())

    return SelectedPose(
        pose_detected=True,
        people_detected=people_count,
        boxes_detected=box_count,
        detection_index=selected_index,
        frame_width_px=int(width),
        frame_height_px=int(height),
        bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float32),
        bbox_confidence=bbox_confidence,
        keypoints_xy=keypoints_xy,
        keypoint_confidences=confidences,
    )


def _available_pair_mean(
    values: np.ndarray,
    valid: np.ndarray,
    first_index: int,
    second_index: int,
) -> float:
    """Average the available values from a left/right joint pair.

    Usage: Inference only.
    """

    indices = np.asarray((first_index, second_index), dtype=np.int64)
    available = valid[indices]
    if not np.any(available):
        return float("nan")
    return float(np.mean(values[indices][available]))


def normalize_selected_pose(
    pose: SelectedPose,
    *,
    confidence_threshold: float,
    coordinate_clip: float,
) -> np.ndarray:
    """Convert one selected pose to the training-compatible 68 features.

    Usage: Inference only.

    This is the allocation-light equivalent of ``normalize_selected_frames``.
    It intentionally follows the same hip centre, torso scale and bounding-box
    fallback rules used when training the action models.
    """

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if coordinate_clip <= 0.0:
        raise ValueError("coordinate_clip must be positive")

    x = pose.keypoints_xy[:, 0].astype(np.float32, copy=False)
    y = pose.keypoints_xy[:, 1].astype(np.float32, copy=False)
    confidence = pose.keypoint_confidences.astype(np.float32, copy=False)
    valid = (
        pose.pose_detected
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(confidence)
        & (confidence >= confidence_threshold)
    )

    hip_x = _available_pair_mean(x, valid, LEFT_HIP_INDEX, RIGHT_HIP_INDEX)
    hip_y = _available_pair_mean(y, valid, LEFT_HIP_INDEX, RIGHT_HIP_INDEX)
    shoulder_x = _available_pair_mean(
        x,
        valid,
        LEFT_SHOULDER_INDEX,
        RIGHT_SHOULDER_INDEX,
    )
    shoulder_y = _available_pair_mean(
        y,
        valid,
        LEFT_SHOULDER_INDEX,
        RIGHT_SHOULDER_INDEX,
    )

    all_joint_x = float(np.mean(x[valid])) if np.any(valid) else float("nan")
    all_joint_y = float(np.mean(y[valid])) if np.any(valid) else float("nan")
    x1, y1, x2, y2 = pose.bbox_xyxy
    bbox_center_x = float((x1 + x2) / 2.0)
    bbox_center_y = float((y1 + y2) / 2.0)
    bbox_height = float(max(0.0, y2 - y1))

    center_x = hip_x if np.isfinite(hip_x) else all_joint_x
    center_y = hip_y if np.isfinite(hip_y) else all_joint_y
    if not np.isfinite(center_x):
        center_x = bbox_center_x
    if not np.isfinite(center_y):
        center_y = bbox_center_y

    torso_length = float(np.hypot(shoulder_x - hip_x, shoulder_y - hip_y))
    scale = torso_length if np.isfinite(torso_length) and torso_length > 1.0 else bbox_height * 0.3
    if not np.isfinite(scale) or scale <= 1.0:
        scale = 1.0

    body_x = np.clip((x - center_x) / scale, -coordinate_clip, coordinate_clip)
    body_y = np.clip((y - center_y) / scale, -coordinate_clip, coordinate_clip)
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
    expected_features = len(JOINT_NAMES) * len(FEATURE_CHANNELS)
    return channels.reshape(expected_features).astype(np.float32, copy=False)


def pose_points(pose: SelectedPose) -> dict[str, JointPoint]:
    """Expose selected raw joints to the existing analytics package.

    Usage: Inference only.
    """

    if not pose.pose_detected:
        return {}
    return {
        joint_name: JointPoint(
            x_px=float(pose.keypoints_xy[index, 0]),
            y_px=float(pose.keypoints_xy[index, 1]),
            confidence=float(pose.keypoint_confidences[index]),
        )
        for index, joint_name in enumerate(JOINT_NAMES)
    }
