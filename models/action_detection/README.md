# Action detection: a guided tour

Welcome. This folder turns tracked human poses into useful Muay Thai
information. It is organized as a teaching project: pose estimation, temporal
classification and interpretable post-processing are separate, so each idea
can be understood and improved independently.

The system answers three different questions:

1. **Where is the person?** YOLO Pose estimates 17 body joints.
2. **What are they doing?** A TCN or LightGBM model classifies a short history
   of poses.
3. **What can we measure?** Analytics groups predictions into strikes, counts
   them and estimates speed.

A classifier predicts labels; it should not also be responsible for counting
repetitions or converting pixels to metres. That separation is the central
design idea of this folder.

## The complete data flow

```text
video or webcam
      |
      v
YOLO Pose: boxes + 17 keypoints for every detected person
      |
      v
select the largest detected person
      |
      +------------------------------+
      |                              |
      v                              v
body-centred pose features       raw pixel keypoints
17 joints x 4 channels           + CFR timestamp
      |                              |
      v                              v
32-frame causal history          smoothing + physical scale
      |                              |
      +-----------+------------------+
                  v
       guard and striking classifiers
                  |
                  v
       per-frame class probabilities
                  |
                  v
       strike event state machine
                  |
                  v
 annotated video + JSONL + events CSV + summary JSON
```

During inference YOLO runs once per frame. The selected pose is reused by both
classifiers and by analytics.

## Suggested reading order

| Order | File or folder | What to learn there |
|---:|---|---|
| 1 | `config.py` | Task vocabularies and label validation. |
| 2 | `preprocessing.py` | Raw pixel joints to model-ready temporal features. |
| 3 | [`LightGBM/`](LightGBM/README.md) | The simplest trainable baseline. Start here if temporal ML is new to you. |
| 4 | [`TCN/`](TCN/README.md) | A causal neural network that learns motion patterns. |
| 5 | `inference.py` | The shared video loop used by both model families. |
| 6 | [`analytics/`](analytics/README.md) | Counting, limb assignment and approximate speed. |

`inference.py` is a library, not the normal command-line entry point. Run
`TCN/infer.py` or `LightGBM/infer.py`; each loads its models and calls the
shared inference loop.

## The two classification tasks

Guard and striking are deliberately trained as separate classifiers:

```text
guard:     background, guard_up, guard_down
striking:  background, punch, elbow, kick, knee
```

One frame can therefore be both `guard_up` and `punch`. A single combined
vocabulary would make that combination impossible.

Generated datasets live under:

```text
dataset/jointswithactionlabels/guard
dataset/jointswithactionlabels/striking
```

An individual video may omit a class. The complete dataset and training split
must collectively contain every class required by the task. Every frame still
needs a label; use `background` when no relevant action occurs.

## What one model input looks like

YOLO supplies `x_px`, `y_px` and confidence for each joint.
`preprocessing.py` converts every one of the 17 joints into four channels:

| Channel | Meaning |
|---|---|
| `x_body` | Horizontal position relative to the body centre, in torso lengths. |
| `y_body` | Vertical position relative to the body centre, in torso lengths. |
| `confidence` | YOLO confidence clipped to `[0, 1]`. |
| `valid` | `1` when the joint passed confidence checks; otherwise `0`. |

One frame therefore has `17 x 4 = 68` features. With the default 32-frame
window, one example has shape `[32, 68]`. The TCN consumes this history
directly; LightGBM flattens it to `2,176` scalar inputs.

### Why body-centred coordinates?

Pixel coordinates vary with camera resolution, framing and distance. The
preprocessing instead:

1. Uses the midpoint of visible hips as body centre.
2. Falls back to the mean of valid joints, then the bounding-box centre.
3. Uses shoulder-to-hip torso length as scale.
4. Falls back to 30% of bounding-box height when torso scale is unavailable.
5. Replaces invalid coordinates with zero while keeping confidence and
   validity channels.

The model can distinguish a missing joint from a real joint at the origin.
Coordinates are clipped to prevent one bad keypoint dominating the input.

### Why causal windows?

A real-time model cannot use future frames. At frame `t`, its history contains
only frames up to `t`:

```text
t=0: [0,  0,  0, F0]
t=1: [0,  0, F0, F1]
t=2: [0, F0, F1, F2]
t=3: [F0, F1, F2, F3]
```

Left padding permits an immediate prediction, although early predictions have
less motion context.

### Horizontal-flip augmentation

Training poses are mirrored by default. Correct mirroring negates horizontal
coordinates and exchanges anatomical left/right joint channels. Validation is
never augmented. This improves facing-direction resilience without leaking
synthetic copies into validation.

## Training: the shortest useful path

Begin with LightGBM to check whether the dataset is learnable, then compare it
with the TCN.

Train guard models:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/train.py `
  --dataset-dir dataset/jointswithactionlabels/guard

conda run -n muay-thai python models/action_detection/TCN/train.py `
  --dataset-dir dataset/jointswithactionlabels/guard
```

Train striking models:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/train.py `
  --dataset-dir dataset/jointswithactionlabels/striking

conda run -n muay-thai python models/action_detection/TCN/train.py `
  --dataset-dir dataset/jointswithactionlabels/striking
```

The task is inferred from the final folder name. For a differently named
folder, pass `--task guard` or `--task striking`.

### Choosing validation videos

Always split by complete video. A random frame split would place nearly
identical neighbouring frames in training and validation and inflate results.

If only validation IDs are supplied, every other available video trains:

```powershell
conda run -n muay-thai python models/action_detection/TCN/train.py `
  --dataset-dir dataset/jointswithactionlabels/striking `
  --val-videos 20260808_015154_30fps
```

Video IDs are the values inside the CSV, normally the filename without
`_joints_labels.csv`. If no IDs are supplied, sorted videos are split
deterministically using the final 20% for validation. You may also pass both
`--train-videos` and `--val-videos` for complete control.

### Understanding the printed metrics

- **Accuracy** is the fraction of correctly classified frames. It can be
  misleading when `background` dominates.
- **Per-class precision and recall** show which actions are confused.
- **Macro F1** gives every class equal importance and is the main comparison
  metric in this project.

The TCN selects the best epoch by validation macro F1. LightGBM early-stops on
validation log loss and reports macro F1 after fitting.

## Choosing between LightGBM and TCN

| Question | LightGBM | TCN |
|---|---|---|
| Best role | Fast, interpretable baseline | Main temporal experiment |
| Input | Flattened causal window | Causal sequence |
| Learns temporal locality | Indirectly | Explicitly through convolutions |
| GPU required | No | Optional; useful for training |
| Training complexity | Low | Moderate |
| Feature importance | Available, with caveats | Not directly available |

Use identical preprocessing, video splits and window sizes for a fair model
comparison.

## Dual-model inference

Each inference command loads two bundles from one model family: guard and
striking.

TCN on a webcam:

```powershell
conda run -n muay-thai python models/action_detection/TCN/infer.py `
  --source 0 `
  --display
```

LightGBM on a file:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/infer.py `
  --source media/videos/30fps/example.mp4
```

Use `--guard-weights` and `--striking-weights` to override saved bundles.

Shared inference then:

1. Places file input on a 30 FPS constant-rate timeline, or requests 30 FPS
   webcam capture.
2. Runs YOLO Pose and selects the largest person.
3. Normalizes the pose using settings saved in each model bundle.
4. Updates guard and striking histories.
5. Gets both probability vectors.
6. Updates strike analytics.
7. Draws the skeleton, predictions and metrics.
8. Writes the annotated frame and JSONL record.

The video timeline remains 30 FPS even if the computer processes only 8 frames
per wall-clock second. File processing then takes longer than the video. True
live 30 FPS webcam analysis requires the entire pipeline—especially YOLO—to
sustain 30 FPS.

## Strike analytics

Select metrics during either inference entry point:

```powershell
--metrics count
--metrics speed
--metrics count speed --person-height-cm 182
```

Both are enabled by default; default height is 175 cm. Height helps estimate a
pixels-per-metre scale from canonical limb proportions. This is a monocular
estimate, not a laboratory measurement. See the
[analytics guide](analytics/README.md) for the state machine and limitations.

## Outputs from one inference run

Every run gets a timestamped folder beneath `output/`:

| Output | Purpose |
|---|---|
| `*_annotated.mp4` | Skeleton, class predictions, counts and latest speed. |
| `*_predictions.jsonl` | One machine-readable record per frame. |
| `*_events.csv` | One row per completed strike event. |
| `*_summary.json` | Configuration, aggregate counts and speed summary. |

Webcam inference also saves the unannotated recording under
`media/videos/raw/`.

## Tests

Run focused tests from the repository root:

```powershell
conda run -n muay-thai python -m unittest discover `
  -s tests `
  -p "test_action*.py"
```

`test_action_preprocessing.py` checks mirroring and video splits.
`test_action_analytics.py` checks scale, event transitions and quadratic peak
interpolation.

## Common sources of confusion

### “The output says 8 FPS, but my video is 30 FPS”

Eight FPS is processing throughput; 30 FPS is video time. File frames can all
be analysed and written at 30 FPS while taking longer than real time to compute.

### “One video does not contain every class”

That is allowed. The union of training videos must contain every required
class, and validation is most useful when it does too.

### “Some ankles or wrists are missing”

Coordinates become zero, but `confidence` and `valid` tell the classifier they
are missing. Analytics avoids velocity across long gaps, preventing false
speed spikes.

### “Counts do not equal the number of classified frames”

Classification describes frames; counting describes events. The state machine
requires sustained confidence and limb motion, then applies release and
cooldown rules.

### “Estimated speed seems lower than real strike speed”

Only image-plane motion is visible. Smoothing, 30 FPS sampling,
foreshortening and height-proportion error also matter. Treat current speed as
an approximate project metric until validated against a better reference.

## Good student experiments

1. Compare LightGBM and TCN on exactly the same video split.
2. Disable horizontal flips and quantify performance by facing direction.
3. Plot confusion matrices rather than relying on accuracy.
4. Record at 120 FPS, downsample to 30 FPS and measure peak-speed error.
5. Add signed extension/retraction evidence to reduce double counts.
6. Benchmark YOLO, classifiers and analytics separately before optimizing.

The project succeeds not when every metric is perfect, but when each error can
be traced to a clear layer and tested with a focused experiment.
