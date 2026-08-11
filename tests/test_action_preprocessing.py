import unittest

import numpy as np

from models.action_detection.preprocessing import (
    FEATURE_CHANNELS,
    JOINT_NAMES,
    PoseSequence,
    build_window_dataset,
    choose_video_split,
    horizontal_flip_pose_features,
)


def _pose_features(frame_count: int = 2) -> np.ndarray:
    poses = np.zeros(
        (frame_count, len(JOINT_NAMES), len(FEATURE_CHANNELS)),
        dtype=np.float32,
    )
    for joint_index in range(len(JOINT_NAMES)):
        poses[:, joint_index, 0] = joint_index + 1.0
        poses[:, joint_index, 1] = joint_index + 101.0
        poses[:, joint_index, 2] = joint_index / len(JOINT_NAMES)
        poses[:, joint_index, 3] = float(joint_index % 2 == 0)
    return poses.reshape(frame_count, -1)


class HorizontalFlipTests(unittest.TestCase):
    def test_reflects_x_and_swaps_joint_channels(self) -> None:
        features = _pose_features()
        original = features.copy()
        mirrored = horizontal_flip_pose_features(features).reshape(
            len(features), len(JOINT_NAMES), len(FEATURE_CHANNELS)
        )
        poses = original.reshape(
            len(features), len(JOINT_NAMES), len(FEATURE_CHANNELS)
        )

        left_wrist = JOINT_NAMES.index("left_wrist")
        right_wrist = JOINT_NAMES.index("right_wrist")
        np.testing.assert_allclose(
            mirrored[:, left_wrist, 0],
            -poses[:, right_wrist, 0],
        )
        np.testing.assert_allclose(
            mirrored[:, left_wrist, 1:],
            poses[:, right_wrist, 1:],
        )
        np.testing.assert_allclose(mirrored[:, 0, 0], -poses[:, 0, 0])
        np.testing.assert_allclose(mirrored[:, 0, 1:], poses[:, 0, 1:])
        np.testing.assert_array_equal(features, original)

    def test_is_its_own_inverse(self) -> None:
        features = _pose_features()
        restored = horizontal_flip_pose_features(
            horizontal_flip_pose_features(features)
        )
        np.testing.assert_allclose(restored, features)

    def test_rejects_wrong_feature_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "features must have shape"):
            horizontal_flip_pose_features(
                np.zeros((2, 4), dtype=np.float32)
            )

    def test_window_augmentation_doubles_only_requested_dataset(self) -> None:
        features = _pose_features()
        labels = np.asarray(["background", "punch"])
        sequence = PoseSequence(
            video_id="video",
            features=features,
            labels=labels,
            frame_indices=np.asarray([0, 1]),
            timestamps=np.asarray([0.0, 1.0 / 30.0], dtype=np.float32),
        )
        sequences = {sequence.video_id: sequence}

        original_windows, original_targets = build_window_dataset(
            sequences,
            ["video"],
            window_size=2,
            class_names=("background", "punch"),
        )
        augmented_windows, augmented_targets = build_window_dataset(
            sequences,
            ["video"],
            window_size=2,
            class_names=("background", "punch"),
            augment_horizontal_flip=True,
        )

        self.assertEqual(len(original_windows), 2)
        self.assertEqual(len(augmented_windows), 4)
        np.testing.assert_array_equal(
            augmented_windows[:2], original_windows
        )
        np.testing.assert_array_equal(
            original_targets, np.asarray([0, 1])
        )
        np.testing.assert_array_equal(
            augmented_targets,
            np.asarray([0, 1, 0, 1]),
        )


class VideoSplitTests(unittest.TestCase):
    def test_validation_only_uses_all_remaining_videos_for_training(self) -> None:
        split = choose_video_split(
            ["video_3", "video_1", "video_2"],
            validation_video_ids=["video_2"],
        )

        self.assertEqual(split.train_video_ids, ("video_1", "video_3"))
        self.assertEqual(split.validation_video_ids, ("video_2",))
        self.assertEqual(split.ignored_video_ids, ())


if __name__ == "__main__":
    unittest.main()
