# LightGBM: the temporal baseline

This folder contains the simplest trainable action classifier in the project.
It is called a baseline not because it is unimportant, but because it answers
a crucial scientific question:

> Can the labels be predicted from our pose representation before we invest in
> a more complex neural network?

LightGBM receives exactly the same causal pose histories as the TCN. It does
not receive RGB images and this implementation does not add handcrafted joint
angles, distances or velocities.

## Files in this folder

| File | Responsibility |
|---|---|
| `model.py` | Creates the `LGBMClassifier` and saves bundles with Joblib. |
| `train.py` | Prepares windows, fits, evaluates and saves a task model. |
| `infer.py` | Loads guard and striking bundles and starts shared inference. |
| `weights/` | Local `.joblib` bundles. Generated files are ignored by Git. |

Shared normalization, splitting and inference live in the parent folder.

## How a temporal window becomes a tree input

The default causal window has shape:

```text
[time, features] = [32, 68]
```

LightGBM expects a flat table, so `train.py` reshapes it:

```text
[32, 68] -> [2,176]
```

Every temporal position remains a distinct column. For example, a wrist's
horizontal coordinate five frames ago is a different feature from the same
coordinate in the current frame. The model can learn relationships between
them, but it does not have the TCN's built-in assumption that nearby moments
should share convolutional patterns.

This is still a joint-only temporal model—not an RGB model and not a single-
frame classifier.

## Reading `model.py`

`build_model` chooses binary or multiclass classification and configures:

| Setting | Teaching interpretation |
|---|---|
| `n_estimators=500` | Maximum number of boosting trees. Early stopping may use fewer. |
| `learning_rate=0.05` | Each new tree makes a small correction. |
| `num_leaves=31` | Controls tree expressiveness. |
| `subsample=0.9` | Samples rows to reduce overfitting. |
| `colsample_bytree=0.9` | Samples feature columns for each tree. |
| `class_weight="balanced"` | Gives rare action classes more influence. |
| `n_jobs=-1` | Uses available CPU cores. |

`save_model_bundle` writes the fitted estimator and preprocessing metadata with
Joblib. As with the TCN, inference should never have to guess how training
prepared the input.

## What `train.py` does

The script:

1. Resolves the classification task from the dataset folder or `--task`.
2. Loads raw pose CSVs through shared preprocessing.
3. Splits complete videos into training and validation.
4. Checks that training collectively contains every required class.
5. Builds causal windows and mirrors training sequences by default.
6. Flattens each window into one feature row.
7. Fits LightGBM with balanced classes.
8. Early-stops after 30 validation rounds without log-loss improvement.
9. Prints accuracy, macro F1 and the per-class report.
10. Saves a `.joblib` model bundle.

## Training commands

Guard:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/train.py `
  --dataset-dir dataset/jointswithactionlabels/guard
```

Striking with a chosen validation video:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/train.py `
  --dataset-dir dataset/jointswithactionlabels/striking `
  --val-videos 20260808_015154_30fps
```

All other available videos become training data when only `--val-videos` is
passed.

Useful options:

```text
--window-size 32
--confidence-threshold 0.25
--coordinate-clip 5.0
--n-estimators 500
--random-state 42
--no-horizontal-flip
--output path/to/model.joblib
```

Default outputs are:

```text
weights/lightgbm_guard.joblib
weights/lightgbm_striking.joblib
```

## What is saved in the bundle?

| Key | Purpose |
|---|---|
| `model` | The fitted `LGBMClassifier`. |
| `class_names` | Converts numeric predictions back to labels. |
| `feature_names` | Documents the per-frame feature ordering. |
| `window_size` | Reconstructs the causal history. |
| `confidence_threshold` | Repeats joint validity decisions. |
| `coordinate_clip` | Repeats normalization limits. |
| `classification_task` | Prevents task/weight mismatches. |
| split IDs | Makes the experiment reproducible. |

## What `infer.py` does

The script loads the guard and striking Joblib bundles. LightGBM may internally
store only the numeric classes observed by the estimator, so
`predict_probabilities` aligns its probability columns with the saved
`class_names` ordering. Both models are then passed to the shared video loop.

Run on a video:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/infer.py `
  --source media/videos/30fps/20260808_015154_30fps.mp4 `
  --metrics count speed
```

Run on the webcam:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/infer.py `
  --source 0 `
  --display
```

The LightGBM classifier itself is usually inexpensive during inference. YOLO
Pose is likely to dominate total frame time.

## How to use the baseline scientifically

Make the comparison fair:

1. Use the same raw CSVs.
2. Use the same whole-video train/validation split.
3. Use the same window size, confidence threshold and coordinate clip.
4. Keep horizontal-flip augmentation consistent.
5. Compare macro F1 and per-class recall, not only accuracy.

Then interpret the result:

- **Both models poor:** inspect labels, class coverage and pose quality.
- **LightGBM good, TCN poor:** check TCN optimization or overfitting.
- **TCN clearly better:** temporal locality is providing useful structure.
- **Both very good:** evaluate new performers and recording conditions before
  concluding the problem is solved.

## Feature importance: useful but easy to misuse

LightGBM exposes feature importance. It can suggest which joints and moments
matter, but correlated temporal features share or compete for importance.
Treat the result as a clue, not proof that one joint causes a prediction.

To interpret flattened indices correctly, combine the saved window size with
`feature_names`. A useful analysis aggregates importance by:

- joint;
- channel (`x_body`, `y_body`, confidence, validity);
- temporal offset;
- left versus right side.

## Common problems

### Training uses too much memory

Flattened windows grow linearly with window size. Reduce `--window-size` or
train on fewer videos while debugging. Do not silently reduce validation only.

### Accuracy is high but strikes are missed

Background dominance can hide poor action recall. Read the per-class report and
macro F1. Balanced class weights help, but they do not replace sufficient
examples.

### Validation is missing a class

Training requires every class. Validation may run without every class, but
macro F1 and early-stopping behavior become less representative. Prefer a
validation collection containing all labels.

### Predictions differ after changing preprocessing

Retrain the model. The bundle preserves its original preprocessing settings,
and inference intentionally follows those settings.

## Suggested exercises

1. Compare a 1-frame model with a 32-frame model to quantify temporal value.
2. Aggregate feature importance by joint and temporal offset.
3. Change `num_leaves` while keeping the split fixed and plot overfitting.
4. Compare class weighting with a deliberately balanced training subset.
5. Measure classifier latency separately from YOLO and video drawing.
