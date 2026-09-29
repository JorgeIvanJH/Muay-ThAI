# Action-classification dataset

This folder contains the tooling used to create pose-only action datasets from
30-FPS CFR videos and Label Studio timeline annotations.

Guard and striking are intentionally separate projects and datasets:

    classification/guard                 Label Studio guard JSON
    classification/striking              Label Studio striking JSON
    jointswithactionlabels/guard          generated guard pose CSVs
    jointswithactionlabels/striking       generated striking pose CSVs

Guard permits background, guard_up, and guard_down. Striking permits
background, punch, elbow, kick, and knee. The exporter rejects mixed,
unexpected, or incomplete vocabularies before running YOLO.

The workflow is:

1. Convert source recordings to exactly 30-FPS CFR videos under
   media/videos/30fps.
2. Label every frame in the appropriate Label Studio project and export that
   project to dataset/classification/guard or striking.
3. Run build_action_joint_dataset.py with --task guard or --task striking.
4. Change CLASSIFICATION_TASK in verify_action_joint_dataset.ipynb and inspect
   the generated skeletons and labels.

# Label Studio setup

## Video timeline labeling stack

Label Studio and the project's TCN pre-annotation backend run together with
[compose.yml](./compose.yml).

The shared `muay-thai-labeling` Docker network gives each service a stable
hostname:

| Connection | URL |
| --- | --- |
| Browser to Label Studio | `http://localhost:8080` |
| Browser to TCN backend health endpoint | `http://localhost:9091/health` |
| Label Studio container to TCN backend | `http://tcn-backend:9090` |
| TCN backend container to Label Studio | `http://label-studio:8080` |

Do not use the Label Studio URL as the Model Backend URL. Port `8080` is Label
Studio; the backend listens on port `9090` inside the network (`9091` from the
host).

### Prerequisites

- Docker Desktop running Linux containers, with GPU support for the backend
- Trained TCN bundles in `models/action_detection/TCN/weights/`
- Videos encoded at a constant frame rate of exactly 30 FPS

### Persistent environment variables

Docker Compose automatically reads `dataset/.env` when it is run from this
directory. The real `.env` is ignored by Git so its token is not committed.
Use [.env.example](./.env.example) as the template:

```dotenv
LABEL_STUDIO_HOST=http://label-studio:8080
LABEL_STUDIO_API_KEY=replace-with-your-label-studio-access-token
```

Find the token in **Label Studio > Account & Settings > Access Token**. If a
token is created or changed while the containers are running, recreate only
the backend so it receives the new value:

```powershell
docker compose up -d --no-deps --force-recreate tcn-backend
```

The backend uses these variables to download the task videos from Label Studio
and to post its predictions back. They are not Basic Authentication
credentials for the Model connection.

### Persistent Storage

The Label Studio database and uploaded media remain in `dataset/LSdata`. The
backend's prediction cache remains in `dataset/tcn_backend/data` (safe to
delete; predictions are recomputed on demand). Model weights are bind-mounted
read-only from `models/`, so retraining never needs an image rebuild.

### Start and stop

Run all Compose commands from this `dataset` directory.

Build and start both services:

```powershell
docker compose up --build -d
```

Subsequent starts can reuse the existing image:

```powershell
docker compose up -d
```

Check status and logs:

```powershell
docker compose ps
docker compose logs --tail 100 label-studio
docker compose logs --tail 100 tcn-backend
Invoke-RestMethod http://localhost:9091/health
```

Stop the services without deleting their persistent data:

```powershell
docker compose down
```

## Labeling interface

Use [timeline-labeling-guard.xml](./timeline-labeling-guard.xml) for the guard
project and [timeline-labeling-striking.xml](./timeline-labeling-striking.xml)
for the striking project in **Project Settings > Labeling Interface > Code**.

The guard configuration is shown below. The striking configuration is the same
but uses the background, punch, elbow, kick, and knee labels.

```xml
<View>
  <TimelineLabels
    name="videoLabels"
    toName="video"
  >
    <Label value="background" background="#a2a2a2"/>
    <Label value="guard_up" background="#1aff00"/>
    <Label value="guard_down" background="#ff0000"/>
  </TimelineLabels>

  <Video
    name="video"
    value="$video"
    height="700"
    frameRate="30.0"
    timelineHeight="200"
  />
</View>
```

`frameRate="30.0"` must match the actual constant frame rate of every uploaded
video. A mismatched or variable frame rate misaligns annotations and model
predictions.

The label set is what tells the TCN backend which task a project is: it must
match `TASK_CLASS_NAMES` in `models/action_detection/config.py` exactly.

Reference:
[Label Studio ML backend Docker networking](https://labelstud.io/guide/ml#localhost-and-Docker-containers)


## TCN pre-annotation backend

[tcn_backend/](./tcn_backend/) is the project's Label Studio ML backend. Its job
is to pre-fill the timeline with suggested labels (pre-annotations) using the
TCN models this project already trained
(`models/action_detection/TCN/weights/tcn_guard.pt` and `tcn_striking.pt`). You
then correct those suggestions instead of labelling every frame from scratch.

Two things to know about it:

1. **It uses the same model and maths as live inference.** It normalises the
   joints the same way, uses the same 32-frame causal window (each frame's
   prediction only sees that frame and the 31 before it), and loads the same
   trained weights. What you see in Label Studio is what the real model would
   predict on that video.
2. **It only predicts and never trains.** Label Studio can ask a backend to
   learn from new annotations by calling `fit()`; here `fit()` just logs a
   message. To improve the model, export the corrected labels, rebuild the
   dataset and retrain with `models/action_detection/TCN/train.py` as usual.

To (re)build and start only the backend (the `.env` token is reused):

```powershell
docker compose up --build -d tcn-backend
docker compose logs -f tcn-backend
Invoke-RestMethod http://localhost:9091/health
```

Connect it in **each** project:

1. Open **Settings > Model > Connect Model**.
2. Set **Name** to `TCN Timeline` and **Backend URL** to
   `http://tcn-backend:9090` (inside the Docker network; never `localhost`).
3. Select no authentication, leave **Interactive preannotations** off, validate
   and save. Label Studio sends the labeling config; the backend reads the label
   set and picks the guard or striking bundle, or rejects a config whose labels
   match neither task.
4. In **Settings > Annotation** enable **Use predictions to prelabel tasks** and
   select the `tcn_guard@...` / `tcn_striking@...` model version.
5. If an older `YOLO Timeline` model (the stock HumanSignal backend this
   project used before) is still listed, delete it: its service no longer
   exists and Label Studio will report connection errors.

How predictions arrive: processing one video takes minutes, because YOLO has to
run on every frame, but Label Studio only waits 100 s for a `/predict` reply.
So the backend doesn't answer straight away:

1. Opening a task, or selecting tasks and choosing
   **Actions > Retrieve predictions**, queues a background job and returns
   immediately.
2. When the job finishes, the backend sends the prediction to Label Studio
   through its API.
3. Reload the task page and the labels appear on the timeline.

What the prediction contains:

- **Every frame is labelled**, including `background`, so there are no gaps on
  the timeline.
- **Frame numbers start at 1 and ranges include both ends**, the same
  convention as Label Studio's exports.
- **Each region has exactly one label.**
- **`model_version` is `<bundle file>@<training timestamp>`** (for example
  `tcn_guard@...`), so you can tell which weights in `weights/runs/` produced a
  given timeline.

To run the backend on the host instead (debugging), see
[tcn_backend/README.md](./tcn_backend/README.md); Label Studio then connects to
`http://host.docker.internal:9090`.

## Videos

Both labeling projects require 30-FPS CFR videos from
[media/videos/30fps](../media/videos/30fps). Convert source recordings with
[preprocess_fps.sh](../media/videos/preprocess_fps.sh) before uploading them to
Label Studio.
