# Raw joints with action labels

This directory stores raw per-frame YOLO pose detections joined to manual
Label Studio timeline labels. Coordinates remain in pixels; person selection,
normalization, missing-joint handling, and temporal window creation happen in
the model preprocessing pipeline.

These are pose **detections**, not tracked identities. A `person_index`,
`box_index`, or `detection_index` is only meaningful within its frame and can
change on the next frame.

## Row structure

The CSV uses a long format:

- A frame with one detected person has one row.
- A frame with multiple detected people has one row per detection.
- A frame with no detections retains one row with empty detection and joint
  fields, preserving the temporal sequence.

For an Ultralytics pose model, `result.boxes[i]` is the person bounding box
associated with `result.keypoints[i]`. Therefore, `boxes_detected` and
`people_detected` should normally be equal, and `box_index`, `person_index`,
and `detection_index` should normally refer to the same array position. They
are stored separately because Ultralytics exposes boxes and keypoints as
separate result objects, and the raw exporter preserves a partial result if one
component is unexpectedly absent.

## Columns

| Column | Meaning |
| --- | --- |
| `task_id` | Label Studio task identifier for the annotated video. |
| `video_id` | Video filename stem; groups all frames and detections belonging to one video. |
| `frame_index` | Zero-based decoded video-frame index used by Python/OpenCV. |
| `label_frame` | One-based frame number used to match the Label Studio timeline. Equal to `frame_index + 1`. |
| `time_seconds` | Frame timestamp calculated as `frame_index / fps`. |
| `action_label` | Manual frame label exported from Label Studio, such as `guard_up`, `guard_down`, or `background`. It is repeated for every detection in the frame. |
| `pose_detected` | `1` when this row contains a YOLO keypoint set; otherwise `0`. |
| `bbox_detected` | `1` when this row contains a YOLO bounding box; otherwise `0`. |
| `people_detected` | Number of keypoint/person instances returned by YOLO for the frame. Repeated on every row from that frame. |
| `boxes_detected` | Number of bounding boxes returned by YOLO for the frame. Repeated on every row from that frame. For a pose model this should normally equal `people_detected`. |
| `detection_index` | Builder-assigned zero-based position used to associate `boxes[i]` with `keypoints[i]` inside the frame. Empty when the frame has no detection. Not a tracking ID. |
| `person_index` | Zero-based position of the keypoint set in `result.keypoints` for this frame. Empty if no keypoints are present. Not a tracking ID. |
| `box_index` | Zero-based position of the box in `result.boxes` for this frame. Empty if no box is present. Not a tracking ID. |
| `frame_width_px` | Width in pixels of the decoded frame supplied to YOLO. |
| `frame_height_px` | Height in pixels of the decoded frame supplied to YOLO. |
| `bbox_confidence` | YOLO confidence score for the person bounding box, normally between 0 and 1. |
| `bbox_class_id` | YOLO class index for the bounding box. With the standard human pose checkpoint this is normally `0` for `person`. |
| `bbox_x1_px` | Left edge of the bounding box in raw frame pixels. |
| `bbox_y1_px` | Top edge of the bounding box in raw frame pixels. |
| `bbox_x2_px` | Right edge of the bounding box in raw frame pixels. |
| `bbox_y2_px` | Bottom edge of the bounding box in raw frame pixels. |
| `<joint>_x_px` | Raw horizontal pixel coordinate predicted by YOLO for `<joint>`. |
| `<joint>_y_px` | Raw vertical pixel coordinate predicted by YOLO for `<joint>`. |
| `<joint>_confidence` | YOLO confidence for `<joint>`, normally between 0 and 1. This must be used to mask unreliable coordinates during preprocessing. |

`<joint>` is one of the 17 COCO pose keypoints:

```text
nose
left_eye, right_eye
left_ear, right_ear
left_shoulder, right_shoulder
left_elbow, right_elbow
left_wrist, right_wrist
left_hip, right_hip
left_knee, right_knee
left_ankle, right_ankle
```

## Train/validation/test grouping

Frames and overlapping temporal windows from the same video are highly
correlated. All rows belonging to a `video_id` must remain in the same
training, validation, or test split. Never randomly split individual CSV rows
or frames across those datasets.
