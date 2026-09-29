# TCN pre-annotation backend

A Label Studio ML backend (`label-studio-ml` SDK) that puts the project's own
guard and striking TCN predictions on the TimelineLabels timeline, so the
annotator only corrects boundaries. It never trains: `fit()` logs and returns.
Retraining stays in `models/action_detection/TCN/train.py`.

Read `model.py`'s module docstring for the design (task selection from the
label config, once-per-process model loading, async post-back, exact JSON).

## How this backend was created

It follows the official template from
[Write your own ML backend](https://labelstud.io/guide/ml_create):

1. Install the SDK: clone `HumanSignal/label-studio-ml-backend`, then
   `pip install -e .`
2. Generate the skeleton: `label-studio-ml create my_ml_backend`, which
   produces the files in this folder:
   - `model.py`: the only project-specific code. It contains a class that
     inherits from `LabelStudioMLBase` (here `TCNTimelineModel`).
   - `_wsgi.py`: the SDK's web server (uWSGI). Not modified.
   - `Dockerfile`: for running in Docker. The template's `docker-compose.yml`
     and `.dockerignore` were replaced by the `tcn-backend` service in
     `dataset/compose.yml` and the root `.dockerignore`, because the build
     context is the repo root.
   - `requirements.txt`: our dependencies. `requirements-base.txt` and
     `requirements-test.txt` are the SDK's and are not modified.
   - `test_api.py`: our tests. `README.md`: this file.
3. Override the SDK methods in `model.py`:
   - `predict(tasks, context)`: required. Returns predictions in Label Studio
     JSON. Here it queues a job and posts the result back asynchronously.
   - `fit(event, data)`: optional and called on annotation events. Here it
     only logs.
4. Run it (`docker compose up`, port 9090 inside the container) and connect it
   in **Settings > Model**.

## Run with Docker (normal use)

From `dataset/` (compose reads `dataset/.env` for `LABEL_STUDIO_API_KEY`):

```powershell
docker compose up --build -d tcn-backend
docker compose logs -f tcn-backend
Invoke-RestMethod http://localhost:9091/health
```

Then in each Label Studio project: **Settings > Model > Connect Model**,
Backend URL `http://tcn-backend:9090` (never `localhost`), no auth, interactive
pre-annotations off. See `dataset/README.md` for the full walkthrough.

The service uses the GPU (`gpus: all`) for YOLO and the CPU for the two TCNs.
Weights are bind-mounted from `models/action_detection/TCN/weights` and
`models/yolo/weights`; after retraining, `docker compose restart tcn-backend`.

## Run without Docker (debugging)

Use the `muay-thai-ls` conda env (the ML SDK pins conflict with `muay-thai`):

```powershell
conda run -n muay-thai-ls label-studio-ml start dataset/tcn_backend -p 9090
```

Point Label Studio (in Docker) at `http://host.docker.internal:9090`, and set
`LABEL_STUDIO_URL=http://localhost:8080` plus `LABEL_STUDIO_API_KEY` in the
shell so the backend can download task videos and post predictions.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PREDICT_MODE` | `async` | `async`: schedule and post back via the API; `sync`: answer inline |
| `YOLO_DEVICE` | `0` if CUDA else `cpu` | Ultralytics device for pose |
| `TCN_DEVICE` | `cpu` | Device for the classifiers |
| `GUARD_WEIGHTS`, `STRIKING_WEIGHTS` | `TCN/weights/tcn_{guard,striking}.pt` | Bundles |
| `POSE_WEIGHTS` | `models/yolo/config.py:YOLO_REALTIME_WEIGHTS` | Pose model |
| `MAX_FRAMES` | unlimited | Debug/test limit per video |
| `MODEL_DIR` | `./data` | SDK cache and `results/` prediction cache |
| `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY` | | Download videos, post predictions |

## Tests

```powershell
conda run -n muay-thai-ls pytest dataset/tcn_backend/test_api.py -q
```
