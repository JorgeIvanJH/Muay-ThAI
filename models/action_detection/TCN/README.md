# TCN: learning motion with causal convolutions

This folder contains the neural-network version of the action classifier. A
Temporal Convolutional Network (TCN) sees a short sequence of poses and learns
which changes over time correspond to guard position or a strike.

If this is your first temporal neural network, keep one sentence in mind:

> A TCN is a one-dimensional convolutional network whose horizontal axis is
> time rather than image width.

The TCN never receives RGB pixels. YOLO has already converted the person into
body-joint features.

## Files in this folder

| File | Responsibility |
|---|---|
| `model.py` | Defines the causal convolution, residual block and complete classifier. |
| `train.py` | Loads pose sequences, trains, validates and saves a model bundle. |
| `infer.py` | Loads guard and striking bundles and starts shared video inference. |
| `weights/` | Local `.pt` bundles. Generated weights are ignored by Git. |

The shared preprocessing and video loop live one level above in
`preprocessing.py` and `inference.py`.

## Input and output shapes

With project defaults, a training batch has shape:

```text
[batch, time, features] = [batch, 32, 68]
```

The network returns one logit per class:

```text
guard model:    [batch, 3]
striking model: [batch, 5]
```

The output describes the final frame in the input window. Every earlier frame
provides motion context.

## Reading `model.py`

### `CausalConv1d`

A normal temporal convolution can look both backwards and forwards. That
would leak future frames into the prediction and make offline validation
better than real-time behavior.

`CausalConv1d` adds padding only conceptually to the left. PyTorch's
convolution pads both sides, so the class removes the extra outputs on the
right. The result at time `t` depends only on time `t` and earlier.

### `TemporalResidualBlock`

Each block contains:

```text
causal convolution
batch normalization
ReLU
dropout
causal convolution
batch normalization
ReLU
dropout
+ residual connection
```

The residual connection provides a short path around the two convolutions. It
helps gradients travel through deeper networks and lets a block learn a small
correction rather than reconstructing every useful feature.

If input and output channel counts differ, a `1 x 1` convolution reshapes the
residual path.

### Dilations

Default blocks use dilations `1`, `2` and `4`. A dilated convolution skips
between temporal samples, giving later blocks a wider view without a very deep
network.

With kernel size 3 and two convolutions per default block, the receptive field
is approximately 29 frames. The 32-frame window therefore contains about one
second of context at 30 FPS.

### `TCNClassifier`

PyTorch temporal convolutions expect `[batch, channels, time]`, so the model
transposes the input from `[batch, time, features]`. After the residual stack,
it selects the final temporal position and maps it to class logits with a
linear layer.

Softmax is not part of the model because training cross-entropy expects raw
logits. `infer.py` applies softmax when probabilities are needed.

## What `train.py` does

The training script follows this sequence:

1. Infer `guard` or `striking` from the dataset folder.
2. Load every raw joint-and-label CSV.
3. Select the largest person and normalize all poses.
4. Split complete videos into training and validation.
5. Confirm the training split contains every task class.
6. Build causal windows; mirror training sequences when enabled.
7. Calculate balanced class weights.
8. Train with AdamW and weighted cross-entropy.
9. Measure validation loss and macro F1 after each epoch.
10. Keep the state with the best macro F1 and stop after the patience limit.
11. Print a per-class report and save the complete bundle.

Class weights matter because most videos contain far more `background` frames
than strike frames. Without them, predicting background can become an
attractive shortcut.

## Training commands

Guard:

```powershell
conda run -n muay-thai python models/action_detection/TCN/train.py `
  --dataset-dir dataset/jointswithactionlabels/guard
```

Striking with one explicit validation video:

```powershell
conda run -n muay-thai python models/action_detection/TCN/train.py `
  --dataset-dir dataset/jointswithactionlabels/striking `
  --val-videos 20260808_015154_30fps
```

Only the validation IDs are required. All other CSV videos in the supplied
folder become training data.

Useful experiment options:

```text
--window-size 32
--channels 64 64 64
--kernel-size 3
--dropout 0.2
--epochs 30
--batch-size 256
--learning-rate 0.001
--patience 6
--device auto
--no-horizontal-flip
```

Change one family of settings at a time. For example, compare window sizes
while holding the video split and network channels fixed.

## What is saved in a `.pt` bundle?

The bundle contains more than weights:

| Key | Why it is saved |
|---|---|
| `state_dict` | Learned PyTorch parameters. |
| `model_config` | Reconstructs the exact TCN architecture. |
| `class_names` | Maps output indices back to labels. |
| `feature_names` | Records the expected feature ordering. |
| `window_size` | Recreates the temporal history at inference. |
| `confidence_threshold` | Repeats training-time joint validity rules. |
| `coordinate_clip` | Repeats training-time normalization. |
| `classification_task` | Prevents using guard weights as striking weights. |
| split IDs | Records which videos trained and validated the model. |

Saving preprocessing settings with the model prevents silent train/inference
mismatches.

Default output names are:

```text
weights/tcn_guard.pt
weights/tcn_striking.pt
```

## What `infer.py` does

`infer.py` is deliberately small. It:

1. Selects `cpu` or `cuda`.
2. Loads and validates both `.pt` bundles.
3. Rebuilds both TCN classifiers.
4. Wraps each classifier as an `ActionModelRuntime` probability function.
5. Hands both runtimes to shared `run_action_inference`.

Run on a file:

```powershell
conda run -n muay-thai python models/action_detection/TCN/infer.py `
  --source media/videos/30fps/20260808_015154_30fps.mp4 `
  --metrics count speed `
  --person-height-cm 175
```

Run on a webcam:

```powershell
conda run -n muay-thai python models/action_detection/TCN/infer.py `
  --source 0 `
  --display
```

The `--device` option controls only the TCN. Use `--yolo-device` to override
the Ultralytics pose device.

## When should the TCN outperform LightGBM?

The TCN has an advantage when the ordering and local shape of motion matter:
extension followed by retraction, acceleration before impact, or a guard that
changes over several frames. It may not win when the dataset is tiny, labels
are noisy or most classes can be recognized from one posture.

That is why LightGBM remains valuable: if the TCN cannot beat a simple
baseline on the same split, inspect the data before making the network larger.

## Common problems

### CUDA runs out of memory

Reduce `--batch-size`, then reduce `--channels`. Window size also affects
memory, but changing it alters the amount of motion context.

### Validation F1 is unstable

Use more complete validation videos and confirm every class is represented.
One short video can make macro F1 change sharply from a few frames.

### Training accuracy improves but validation does not

This suggests overfitting. Collect more performers, keep horizontal flips,
increase dropout or reduce channels. Do not move neighbouring frames from
validation into training.

### Early predictions are poor

At the beginning of a stream the causal window is left-padded. The model has
not yet observed a complete history; this is expected.

## Suggested exercises

1. Calculate the receptive field for a different channel-stack length.
2. Compare 16-, 32- and 48-frame histories on the same split.
3. Plot training loss and validation macro F1 by epoch.
4. Compare CPU and CUDA classifier latency without including YOLO.
5. Replace the last-frame readout with temporal average pooling and explain
   why it may or may not be appropriate for frame-wise labels.
