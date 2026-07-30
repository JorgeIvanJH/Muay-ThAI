# Action detection

Both models use the raw per-detection CSV files in
`dataset/jointswithactionlabels` and the same preprocessing pipeline:

1. Select the largest pose detection in each frame.
2. Centre joints on the hips, with safe fallbacks for missing joints.
3. Scale coordinates by torso length.
4. Retain joint confidence and valid/missing masks.
5. Build left-padded causal windows ending at the current frame.

No angles, velocities, or manually engineered joint-distance features are
used. By default, the first two sorted video IDs are used for training and the
third for validation.

Update the Conda environment after dependency changes:

```powershell
conda env update -n muay-thai -f environment.yml
```

Train LightGBM:

```powershell
conda run -n muay-thai python models/action_detection/LightGBM/train.py
```

Train the TCN:

```powershell
conda run -n muay-thai python models/action_detection/TCN/train.py
```

Choose the split explicitly when more videos are available:

```powershell
conda run -n muay-thai python models/action_detection/TCN/train.py `
  --train-videos video_1 video_2 `
  --val-video video_3
```

Weights and the preprocessing/class configuration needed to load them are
written beneath each model's `weights` directory.
