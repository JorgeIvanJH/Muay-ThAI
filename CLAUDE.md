# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Muay-ThAI turns a video or webcam feed of one person shadowboxing into Muay Thai
information: YOLO Pose extracts 17 joints per frame, a temporal classifier (TCN or
LightGBM) labels each frame, and a deterministic analytics layer groups labels into
strike events, counts them and estimates speed. It is written as a teaching project,
so each layer is deliberately separated and documented in its own README. Read the
folder READMEs before changing a layer; they are the design record:

- `models/action_detection/README.md` (start here), then `realtime/`, `analytics/`,
  `TCN/`, `LightGBM/` READMEs
- `dataset/README.md`, `dataset/jointswithactionlabels/README.md`
- `media/videos/README.md`

## Environment and commands

Conda env `muay-thai` (Python 3.11, CUDA 12.8 torch). Run everything from the repo
root; scripts insert the root into `sys.path` and import `models.*` absolutely.
`ultralytics` is used everywhere but is not listed in `environment.yml`; it is
installed manually into the env.

```powershell
# Tests (unittest, no pytest config). Always from repo root.
conda run -n muay-thai python -m unittest discover -s tests -p "test_action*.py"
# Single test module / single test
conda run -n muay-thai python -m unittest tests.test_action_realtime
conda run -n muay-thai python -m unittest tests.test_action_analytics.SomeTestCase.test_name

# Train (task is inferred from the dataset folder name: guard | striking)
conda run -n muay-thai python models/action_detection/LightGBM/train.py --dataset-dir dataset/jointswithactionlabels/guard
conda run -n muay-thai python models/action_detection/TCN/train.py --dataset-dir dataset/jointswithactionlabels/striking --val-videos 20260808_015154_30fps

# Inference: always dual-model (guard + striking) from ONE family. Webcam = --source 0.
conda run -n muay-thai python models/action_detection/TCN/infer.py --source 0 --display
conda run -n muay-thai python models/action_detection/LightGBM/infer.py --source media/videos/30fps/<clip>_30fps.mp4 --metrics count speed
#   low-overhead benchmark: add --no-save-annotated --no-save-raw ; limit with --max-frames N

# Dataset generation (Label Studio JSON export + YOLO -> per-video CSVs)
conda run -n muay-thai python dataset/build_action_joint_dataset.py --task guard   # or --task striking ; --overwrite to regenerate
#   then inspect with dataset/verify_action_joint_dataset.ipynb (set CLASSIFICATION_TASK)

# Video normalisation (ffmpeg/ffprobe on PATH). raw/ -> 30fps/ and 60fps/
bash ./media/videos/preprocess_fps.sh          # --probe to inspect, -f to force re-encode

# Label Studio + TCN ML backend (run from dataset/, needs dataset/.env from .env.example)
docker compose up --build -d      # LS at :8080, TCN backend at :9091 ; Backend URL inside LS is http://tcn-backend:9090
docker compose up --build -d tcn-backend   # rebuild/restart only the backend
conda run -n muay-thai-ls pytest dataset/tcn_backend/test_api.py -q             # backend tests need the ML SDK env, not muay-thai
conda run -n muay-thai python models/action_detection/evaluate_video_predictions.py --task guard --video-id 20260728_130155_30fps --yolo-device 0
```

Trained weights are gitignored; default bundles live at
`models/action_detection/{TCN,LightGBM}/weights/{tcn,lightgbm}_{guard,striking}.{pt,joblib}`,
with timestamped copies in `weights/runs/`. Every inference run writes a timestamped
folder under `output/` (annotated MP4, predictions JSONL, events CSV, summary JSON);
webcam runs also save the raw recording to `media/videos/raw/`.

## Architecture: the things you must not break

**Two independent tasks, never merged.** Guard (`background, guard_up, guard_down`)
and striking (`background, punch, elbow, kick, knee`) are separate Label Studio
projects, separate dataset folders, separate trained models, and are run as a pair at
inference. `models/action_detection/config.py` (`TASK_CLASS_NAMES`,
`validate_task_labels`) is the single vocabulary source; the dataset builder, trainers
and inference all validate against it and reject mixed vocabularies. Bundles store
`classification_task` so guard weights cannot be loaded as striking.

**Train/inference preprocessing must stay numerically identical.** Training uses
pandas (`preprocessing.py`: `normalize_selected_frames`, `causal_windows`,
horizontal-flip augmentation). Live inference re-implements the same maths on NumPy
in `realtime/pose.py` (`SelectedPose`, `normalize_selected_pose`) and
`realtime/windows.py` (`TemporalWindowBuffer`) for speed. `tests/test_action_realtime.py`
asserts equivalence. If you change one side, change the other and the test. Model
bundles (`.pt` / `.joblib`) carry `window_size`, `confidence_threshold`,
`coordinate_clip`, `feature_names`, `class_names` and the video split; inference
reads these from the bundle rather than from CLI flags, so preprocessing changes
require retraining.

Per-frame features are 17 joints x 4 channels (`x_body`, `y_body`, `confidence`,
`valid`) = 68, body-centred on the hip midpoint and scaled by torso length. Default
causal window is 32 frames, left-padded so frame 0 can already predict. TCN consumes
`[32, 68]`; LightGBM flattens to 2176 columns. Missing joints become zeros with
`valid=0`, never dropped.

**Split by whole video, never by frame.** Neighbouring frames are near-duplicates.
`--train-videos` / `--val-videos` take `video_id`s (CSV filename stem without
`_joints_labels`). Default is a deterministic sorted split with the last 20% as
validation. Macro F1 is the project's comparison metric, not accuracy.

**`inference.py` is a library, not an entry point.** `TCN/infer.py` and
`LightGBM/infer.py` each build their two `ActionModelRuntime`s and call the shared
`build_inference_parser` / `run_action_inference`. Model-family-specific code
(softmax for TCN, class-column alignment for LightGBM) belongs in the family's
`infer.py`; anything shared belongs in `inference.py` or `realtime/`.

**Real-time pipeline ownership (`realtime/`).** One capture thread (only caller of
`VideoCapture.read`) -> bounded queue -> one YOLO pose thread (only caller of the
predictor) -> one ordered coordinating thread that owns both temporal buffers and
`StrikeAnalytics` -> async video writer threads. Analytics and window buffers are
stateful and must see frames in timestamp order; do not parallelise them across
frames. Webcam queues drop-oldest and count drops; file queues block and keep every
frame with a synthetic 30 FPS CFR timeline. Display code lives only in
`realtime/display.py` (two independent windows plus the composited saved frame).
TCN `--device auto` resolves to CPU on purpose so the classifiers overlap with GPU
YOLO; `--yolo-device` controls YOLO separately.

**Analytics is post-processing, not a model.** `analytics/pipeline.py`
(`StrikeAnalytics`, `AnalyticsConfig`) receives raw pixel keypoints plus striking
probabilities, estimates pixels-per-metre from `--person-height-cm` and limb
proportions (`anthropometry.py`), smooths and differentiates endpoints
(`kinematics.py`), and runs an `IDLE -> CANDIDATE -> ACTIVE -> IDLE` state machine
(`strike_events.py`, thresholds in `StrikeStateMachineConfig`). Counts are events,
not labelled frames. Known limitation recorded in the README: magnitude-only speed
can double-count retractions, especially elbows.

**Videos must be 30 FPS constant-frame-rate and upright.** Everything downstream
(Label Studio `frameRate="30.0"`, `time_seconds = frame_index / fps`, analytics
velocity) assumes it. Point YOLO/OpenCV at `media/videos/30fps/`, never `raw/`.
Dataset generation uses `models/yolo/config.py:YOLO_WEIGHTS`; live inference uses
`YOLO_REALTIME_WEIGHTS` (overridable with `--pose-model`). Keep dataset generation on
the large model so new CSVs match the pose distribution of existing training data.
Some READMEs still mention `yolo26s-pose.pt` as the live default; `config.py` is the
truth.

**Dataset CSVs are raw and long-format.** `dataset/jointswithactionlabels/<task>/`
holds one row per YOLO detection per frame (a frame with no detection keeps one empty
row), pixel coordinates, and `action_label` repeated per detection. Person selection
(largest box), normalisation and windowing happen later in `preprocessing.py`.
Indices like `person_index` are per-frame positions, not tracking IDs.

## Other parts of the repo

- `models/yolo/capture_joints.py`, `models/mde/capture_depth.py`,
  `models/capture_depth_joints_*.py` and the root notebooks are an earlier
  monocular-depth (Depth-Anything V2) + joints exploration, configured via
  `models/config.py`, `models/yolo/config.py`, `models/mde/config.py`. They are
  not part of the action-detection pipeline.
- `dataset/tcn_backend/` is our Label Studio ML backend (SDK `LabelStudioMLBase`)
  serving the TCN bundles as TimelineLabels pre-annotations. It reuses
  `models/action_detection/offline.py` (sequential whole-video pass built on the
  `realtime/` primitives), picks guard vs striking from the project's labels, posts
  predictions back asynchronously via the LS API, and must never train (`fit` is a
  logged no-op). Its Docker build context is the repo root (root `.dockerignore`);
  weights are bind-mounted. Host runs use the separate conda env `muay-thai-ls`
  because the ML SDK's pins conflict with `muay-thai`.
- `dataset/LSdata/` is Label Studio's persistent database and media; do not edit.
- `output/` is gitignored scratch: run folders, experiment logs and `.joblib`/`.pt`
  copies. `TODOs` lists the next planned work (ML-backend labelling with the real
  models, active learning, InterSystems IRIS integration).

## Repo conventions

- Design choices in the root README: one person per session, side view,
  classifier predicts action only (no left/right in the label; the limb comes from
  YOLO keypoints via analytics), `model.predict()` rather than `model.track()`.
- Commands in docs are PowerShell-style with `conda run -n muay-thai`; the shell
  here is Git Bash, so use forward slashes.
- Working copy is on Windows; some files are LF and Git warns about CRLF
  conversion. Don't "fix" line endings as a side effect of an edit.
