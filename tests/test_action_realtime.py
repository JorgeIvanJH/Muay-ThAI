import queue
import unittest

import numpy as np
import pandas as pd

from models.action_detection.preprocessing import (
    JOINT_NAMES,
    causal_windows,
    normalize_selected_frames,
)
from models.action_detection.realtime.pose import (
    SelectedPose,
    normalize_selected_pose,
)
from models.action_detection.realtime.actions import ActionPrediction
from models.action_detection.realtime.display import (
    METRICS_PANEL_WIDTH,
    fit_inside,
    overlay_metrics_panel,
    render_metrics_panel,
)
from models.action_detection.realtime.queues import put_latest
from models.action_detection.realtime.telemetry import PerformanceSnapshot
from models.action_detection.realtime.windows import TemporalWindowBuffer


def _selected_pose() -> SelectedPose:
    keypoints = np.zeros((len(JOINT_NAMES), 2), dtype=np.float32)
    confidences = np.full(len(JOINT_NAMES), 0.9, dtype=np.float32)
    for index in range(len(JOINT_NAMES)):
        keypoints[index] = (120.0 + index * 7.0, 80.0 + index * 11.0)
    return SelectedPose(
        pose_detected=True,
        people_detected=1,
        boxes_detected=1,
        detection_index=0,
        frame_width_px=640,
        frame_height_px=480,
        bbox_xyxy=np.asarray([80.0, 40.0, 400.0, 460.0], dtype=np.float32),
        bbox_confidence=0.95,
        keypoints_xy=keypoints,
        keypoint_confidences=confidences,
    )


def _pose_dataframe(pose: SelectedPose) -> pd.DataFrame:
    row = {
        "pose_detected": int(pose.pose_detected),
        "bbox_x1_px": pose.bbox_xyxy[0],
        "bbox_y1_px": pose.bbox_xyxy[1],
        "bbox_x2_px": pose.bbox_xyxy[2],
        "bbox_y2_px": pose.bbox_xyxy[3],
    }
    for index, joint_name in enumerate(JOINT_NAMES):
        row[f"{joint_name}_x_px"] = pose.keypoints_xy[index, 0]
        row[f"{joint_name}_y_px"] = pose.keypoints_xy[index, 1]
        row[f"{joint_name}_confidence"] = pose.keypoint_confidences[index]
    return pd.DataFrame([row])


class NumericPosePreprocessingTests(unittest.TestCase):
    def test_matches_training_dataframe_preprocessing(self) -> None:
        pose = _selected_pose()
        expected = normalize_selected_frames(
            _pose_dataframe(pose),
            confidence_threshold=0.25,
            coordinate_clip=5.0,
        )[0]
        actual = normalize_selected_pose(
            pose,
            confidence_threshold=0.25,
            coordinate_clip=5.0,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_missing_joints_match_training_preprocessing(self) -> None:
        original = _selected_pose()
        keypoints = original.keypoints_xy.copy()
        confidences = original.keypoint_confidences.copy()
        keypoints[9] = np.nan
        confidences[10] = 0.05
        pose = SelectedPose(
            **{
                **original.__dict__,
                "keypoints_xy": keypoints,
                "keypoint_confidences": confidences,
            }
        )
        expected = normalize_selected_frames(
            _pose_dataframe(pose),
            confidence_threshold=0.25,
            coordinate_clip=5.0,
        )[0]
        actual = normalize_selected_pose(
            pose,
            confidence_threshold=0.25,
            coordinate_clip=5.0,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class TemporalWindowBufferTests(unittest.TestCase):
    def test_matches_last_training_causal_window(self) -> None:
        buffer = TemporalWindowBuffer(window_size=4, feature_count=3)
        history = []
        for frame_index in range(7):
            features = np.asarray(
                [frame_index, frame_index + 10, frame_index + 20],
                dtype=np.float32,
            )
            history.append(features)
            buffer.append(features)
            expected = causal_windows(np.stack(history), 4)[-1]
            np.testing.assert_array_equal(buffer.current_window(), expected)


class LatestQueueTests(unittest.TestCase):
    def test_full_queue_discards_oldest_item(self) -> None:
        values: queue.Queue[int] = queue.Queue(maxsize=2)
        values.put(1)
        values.put(2)

        dropped = put_latest(values, 3)

        self.assertEqual(dropped, 1)
        self.assertEqual(values.get_nowait(), 2)
        self.assertEqual(values.get_nowait(), 3)


class InferenceDisplayTests(unittest.TestCase):
    def test_window_size_preserves_landscape_and_portrait_ratios(self) -> None:
        self.assertEqual(
            fit_inside(1920, 1080, max_width=720, max_height=720),
            (720, 405),
        )
        self.assertEqual(
            fit_inside(1080, 1920, max_width=720, max_height=720),
            (405, 720),
        )

    def test_metrics_render_separately_and_copy_into_saved_frame(self) -> None:
        predictions = (
            ActionPrediction(
                classification_task="guard",
                model_name="test-guard",
                class_names=("background", "guard_up", "guard_down"),
                class_name="guard_up",
                probability=0.9,
                probabilities=np.asarray([0.05, 0.9, 0.05]),
            ),
            ActionPrediction(
                classification_task="striking",
                model_name="test-striking",
                class_names=("background", "punch", "elbow", "kick", "knee"),
                class_name="punch",
                probability=0.8,
                probabilities=np.asarray([0.1, 0.8, 0.05, 0.03, 0.02]),
            ),
        )
        performance = PerformanceSnapshot(
            capture_fps=30.0,
            pose_fps=29.8,
            action_fps=29.8,
            pose_p95_ms=20.0,
            action_p95_ms=3.0,
            end_to_end_p95_ms=28.0,
            dropped_frames=0,
        )

        panel = render_metrics_panel(
            predictions,
            ("Counts Punch L/R 1/2", "Speed waiting"),
            performance,
        )
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        saved_frame = overlay_metrics_panel(frame, panel)

        self.assertEqual(panel.shape[1], METRICS_PANEL_WIDTH)
        self.assertEqual(saved_frame.shape, frame.shape)
        self.assertTrue(np.all(frame == 127))
        self.assertFalse(np.array_equal(saved_frame, frame))


if __name__ == "__main__":
    unittest.main()
