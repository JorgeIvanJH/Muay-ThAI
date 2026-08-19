# Action detection

Guard and striking are separate classification tasks that consume the same
YOLO pose representation:

    guard:     background, guard_up, guard_down
    striking:  background, punch, elbow, kick, knee

Their generated datasets live under:

    dataset/jointswithactionlabels/guard
    dataset/jointswithactionlabels/striking

Both architectures share preprocessing: select the largest person, centre the
joints on the body, scale by torso length, retain confidence/validity channels,
and build left-padded causal windows. Training sequences are horizontally
mirrored by default: normalized x coordinates are negated and anatomical
left/right joint channels are exchanged. Validation sequences are never
augmented. Dataset labels are checked against the selected task before
training.

## Training

The task is inferred from the final dataset folder name. These commands write
distinct weights beneath the same architecture weights folder:

    conda run -n muay-thai python models/action_detection/TCN/train.py --dataset-dir dataset/jointswithactionlabels/guard
    models/action_detection/TCN/weights/tcn_guard.pt

    conda run -n muay-thai python models/action_detection/TCN/train.py --dataset-dir dataset/jointswithactionlabels/striking
    models/action_detection/TCN/weights/tcn_striking.pt

    conda run -n muay-thai python models/action_detection/LightGBM/train.py --dataset-dir dataset/jointswithactionlabels/guard
    models/action_detection/LightGBM/weights/lightgbm_guard.joblib

    conda run -n muay-thai python models/action_detection/LightGBM/train.py --dataset-dir dataset/jointswithactionlabels/striking
    models/action_detection/LightGBM/weights/lightgbm_striking.joblib

For a custom directory name, pass --task guard or --task striking explicitly.
Use --output only when overriding the standard distinctive filename.
Horizontal mirroring can be disabled for an experiment with
`--no-horizontal-flip`; the default is recommended for direction resilience.

By default, sorted whole-video IDs are split deterministically: the final 20%
are validation and all preceding videos are training. At least one video is
kept in each group. An explicit multi-video split can be supplied:

    conda run -n muay-thai python models/action_detection/TCN/train.py --dataset-dir dataset/jointswithactionlabels/guard --train-videos video_1 video_2 video_3 --val-videos video_4 video_5

## Dual-model inference

Inference runs YOLO pose once per frame, then sends the same selected pose
through independent guard and striking preprocessing histories and models.
Both predictions and confidences are drawn above the shared skeleton and
written to the predictions JSONL.

Run from the webcam:

    conda run -n muay-thai python models/action_detection/TCN/infer.py --source 0 --display

Run LightGBM on a video:

    conda run -n muay-thai python models/action_detection/LightGBM/infer.py --source media/videos/30fps/example.mp4

Override either bundle when needed:

    --guard-weights path/to/tcn_guard.pt
    --striking-weights path/to/tcn_striking.pt

Until the striking weights file exists, inference automatically uses the
background-only striking stub and marks its displayed prediction with [STUB].
As soon as the standard striking weights file is trained, the same inference
command loads it automatically.

File inputs are time-resampled to 30 FPS. Annotated MP4 and JSONL predictions
are written beneath output. Webcam capture is requested and paced at 30 FPS,
and unannotated webcam frames are additionally written beneath
media/videos/raw. Actual throughput is limited primarily by YOLO pose
inference. Press q to stop a displayed session.

## Strike analytics

Inference also has a deterministic analytics layer after pose and striking
classification. It smooths keypoints causally, estimates image scale from the
supplied person height and canonical long-bone proportions, assigns each
strike to the faster anatomical limb, and confirms an event with a small state
machine. The initial state-machine thresholds are deliberately kept in
`analytics/strike_events.py` so they can later be tuned against event-level
ground truth without retraining either classifier.

Both strike counts and speed estimates are enabled by default. Select only one
metric or override the default 175 cm height as follows:

    --metrics count
    --metrics speed
    --metrics count speed --person-height-cm 182

Every run writes the annotated MP4 and predictions JSONL plus an events CSV and
summary JSON in the same output folder. The CSV contains event type, anatomical
side, start/apex/end times, classifier confidence and, when requested, sampled,
robust and locally interpolated peak speeds.

Physical speed remains a monocular estimate. Side-on recordings make projected
extension much more informative, but motion towards the camera, keypoint noise
and 30-FPS temporal sampling cannot be fully recovered. The quadratic estimate
is accepted only when its concave peak lies inside the local sample interval
and does not overshoot the observed peak implausibly; otherwise it falls back
to the sampled peak.
