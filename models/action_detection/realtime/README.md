# Real-time inference pipeline

This package keeps camera input, pose inference, action classification,
analytics and video encoding responsive without hiding their dependencies.
It is intentionally made from small modules rather than one long loop.

## Pipeline ownership

```text
capture thread -> bounded frame queue -> YOLO pose thread
                                              |
                                              v
                                   ordered coordinating thread
                                  /             |              \
                         guard + striking   analytics        display
                                              |
                                              v
                                 asynchronous video writers
```

Only the capture thread calls `VideoCapture.read`. Only the pose thread calls
the YOLO predictor. The coordinating thread owns both temporal histories and
`StrikeAnalytics`, because those objects must observe frames in timestamp
order. Video writers have their own threads so codec latency does not block a
new pose inference.

This is pipeline concurrency: while YOLO handles frame N+1, CPU action models
and analytics can handle frame N, and an encoder can write frame N-1. It does
not run analytics for several frames simultaneously because that would corrupt
velocity history and strike-state transitions.

## Files

| File | Responsibility |
|---|---|
| `types.py` | Immutable timestamped messages between stages. |
| `capture.py` | Truthful webcam timestamps and CFR file resampling. |
| `queues.py` | Drop-oldest live queues and ordered offline queues. |
| `pose.py` | Largest-person selection and numeric body normalization. |
| `windows.py` | One reusable causal window per action model. |
| `actions.py` | Shared normalization and guard/striking execution. |
| `workers.py` | Capture and YOLO worker lifecycle and error propagation. |
| `output.py` | Bounded asynchronous MP4 writers. |
| `telemetry.py` | Rolling FPS, latency and dropped-frame measurements. |

## Webcam versus file semantics

Webcam packets use `time.perf_counter()` measured immediately after a
successful camera read. If YOLO falls behind, the capture queue drops its
oldest waiting frame so latency cannot grow without limit. Every drop is
reported on screen, in JSONL performance data and in the final console
summary.

Video files use their declared FPS to produce a 30-FPS CFR timeline. Their
queues block rather than dropping frames. An offline run may take longer than
the video duration, but it retains every target frame and timestamp.

## Why pandas is absent here

Training works with complete CSV datasets, where pandas is convenient and its
overhead is unimportant. Live inference handles one fixed 17-joint pose at a
time. `SelectedPose` therefore stores compact NumPy arrays and implements the
same normalization math directly. Regression tests compare the numeric output
against `normalize_selected_frames` so training and inference cannot silently
diverge.

## Model placement

YOLO owns the GPU. On the target laptop, the two trained TCNs together measured
about 2.54 ms per frame on CPU and 4.39 ms on CUDA. TCN `--device auto`
therefore uses CPU, allowing action classification to overlap with the next
GPU pose inference. `--device cuda` remains available for other hardware.

Dataset generation continues to use `yolo26l-pose.pt`. Live inference defaults
to `yolo26s-pose.pt`, the largest tested model that sustained the laptop
webcam's measured 29.6 FPS with zero drops while raw and annotated video were
both recorded. Override it when benchmarking another deployment:

```powershell
--pose-model models/yolo/weights/yolo26m-pose.pt
```

Do not select a pose model only from average FPS. Check dropped frames,
end-to-end p95 latency and downstream action accuracy on validation videos.

## Performance controls

Useful inference options include:

```text
--pose-imgsz 640
--pose-precision fp32|fp16
--frame-queue-size 4
--output-queue-size 12
--no-save-annotated
--no-save-raw
```

A 30-FPS camera commonly reports about 29.5–30.0 measured FPS because of
driver timing. The overlay reports separate capture, pose and action rates.
It never describes synthetic timestamps as real inference throughput.

## Safe shutdown

Workers communicate failures through an error queue. A shared stop event asks
all stages to finish, `capture.release()` unblocks a pending camera read, and
video writers drain queued frames before closing their codecs. Live sentinels
may replace stale queued work during shutdown; offline sentinels are ordered
after the final source frame.
