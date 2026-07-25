# media/videos

Video assets for the Muay-ThAI project, organized by processing stage.

| Folder    | Contents                                                       |
|-----------|----------------------------------------------------------------|
| `raw/`    | Original source videos. Mixed frame rates, some variable-frame-rate, some carrying rotation metadata. Source of truth — don't edit in place. |
| `30fps/`  | Normalized copies at a true constant 30 fps.                   |
| `60fps/`  | Normalized copies at a true constant 60 fps.                   |

**Point YOLO / OpenCV at `30fps/` or `60fps/`, never at `raw/`.**

## Why the raw videos need processing

- **Variable frame rate** breaks frame-to-time mapping. `frame_index / fps` only
  equals real seconds when the frame rate is constant; on VFR sources it drifts,
  by up to a second over a clip.
- **Rotation metadata** makes OpenCV read frames sideways. Phone clips store
  landscape pixels plus a "rotate on playback" flag; players honor it, but
  `VideoCapture` may not — and a sideways fighter gives YOLO bad keypoints.
- **Mixed frame rates** (25, 23.976, 29.97, 30 fps) need resampling to a common
  rate so clips are comparable.

`preprocess_fps.sh` fixes all three. Every output is constant frame rate,
physically upright (rotation baked into pixels, flag cleared), square-pixel, and
audio-free.

Reusable script that turns every video in `raw/` into normalized **30 fps** and
**60 fps** copies under `30fps/` and `60fps/`, named
`<original-name>_30fps.mp4` and `<original-name>_60fps.mp4`.

Every output is guaranteed to be:

- **Constant frame rate** — `frame_index / fps` equals real seconds.
- **Physically upright** — rotation baked into pixels, flag cleared, so OpenCV
  cannot read it sideways.
- **Square pixels** (SAR 1:1) — no anamorphic distortion of joint coordinates.
- **Audio-free** by default — CV pipelines never use it; pass `--keep-audio` to retain it.

## Running it

Requires `ffmpeg` and `ffprobe` on your `PATH`. From `media/videos`:

```bash
./preprocess_fps.sh                  # process all videos in raw/
./preprocess_fps.sh --probe          # report fps / VFR / rotation, encode nothing
./preprocess_fps.sh -f               # force re-encode, overwriting existing outputs
./preprocess_fps.sh BagDrill_no1.mp4 # process only the named file(s)
./preprocess_fps.sh --keep-audio     # keep the audio track
./preprocess_fps.sh --help
```

From the repo root, use forward slashes so bash doesn't eat the separators:

```bash
bash ./media/videos/preprocess_fps.sh
```

Outputs are named `<original-name>_30fps.mp4` / `<original-name>_60fps.mp4`.
Existing outputs are skipped, so the script is safe to re-run as new clips land
in `raw/`. After each encode it re-probes the result — you should see
`CFR   upright` on every line.

Supported inputs: `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, `.webm`.

