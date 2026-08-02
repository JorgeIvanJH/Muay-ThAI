This folder is dedicated to everything related to the construction of a dataset aimed to train models to classify actions based from YOLO's tracked joints.

To understand the idea, we can simplify the approach in the following steps:

1. Take video sources with CFR (Constant Frame Rate), whose Frame Rate is known, of people performing the actions of interest. (e.g. kicking, punching, or doing nothing). note: some videos taken with modern cameras can have variable frame rates (VFR) to optimize storage. We store our CFR videos at 30FPS at media\videos\30fps, after being preprocessed using media\videos\preprocess_fps.sh.
2. Manually label the frames corresponding to each of the actions (e.g. frames 30 to 45 whole duration of punch, frames 66 to 90 for kicks, frames 46 to 65 doing nothing), and having the resulting ranges somewhere. We label out videos using LabelStudio, locally deployed in a Docker container using dataset\compose.yml, and export out labels in a minimal json format to dataset\classification.
3. For the same videos, run YOLO model and extract the joints of the person performing the actions, ans save them along with the previously labelled classes for each frame. We achieve this running dataset\build_action_joint_dataset.py, and verify the joint YOLO eskeleton drawn and the classes labelled using dataset\verify_action_joint_dataset.ipynb.

This way we could identify actions based only from YOLO's tracked joints. The benefits of this are:
- simpler classification model (much faster training and inference, much less need for ground truth), since we would rely on strongly established YOLO model to detect people, removing surrounding noise. If we would train a classification model taking raw frames, it would have to account for all the pixels as input, and have lots of ground truth to distinguish the person from multiple backgrounds. YOLO would deal extracting the person from any background, and also the input would only likely be 48 variables (the x,y and confidence score of each of the 16 joints) instead of all the pixels in an image (at least 224x224)
- Classification would deal with the action, and YOLO with assigning the action to the limb. In other words, we wouldnt have to duplicate action abels to account for each arm or each leg. e.g. if we label punches regardless of which arm extended, YOLO would provide the right label for the corresponding arm that was extended for the punch.
- Video Augmentation. We can take the same CFR labelled videos and flip or slightly rotate them. YOLO would identify actions similarly but as if the action was made with the other arm or leg, and the classification labels would apply equally.


# Lbel Studio Stup

TODO: Update

## Video timeline labeling stack

Label Studio and its YOLO ML backend run together with
[compose.yml](./compose.yml).

The shared `muay-thai-labeling` Docker network gives each service a stable
hostname:

| Connection | URL |
| --- | --- |
| Browser to Label Studio | `http://localhost:8080` |
| Browser to YOLO health endpoint | `http://localhost:9090/health` |
| Label Studio container to YOLO | `http://yolo:9090` |
| YOLO container to Label Studio | `http://label-studio:8080` |

Do not use the Label Studio URL as the Model Backend URL. Port `8080` is Label
Studio; the ML backend is on port `9090`.

### Prerequisites

- Docker Desktop running Linux containers
- The Label Studio ML backend cloned below this directory
- Videos encoded at a constant frame rate of exactly 30 FPS

Clone the ML backend once from this `dataset` directory:

```powershell
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
```

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
the YOLO service so it receives the new value:

```powershell
docker compose up -d --no-deps --force-recreate yolo
```

The environment variables are used in the YOLO-to-Label-Studio direction so
the backend can download uploaded or protected videos. They are not Basic
Authentication credentials for the Model connection.

### Persistent Storage

The Label Studio database and uploaded media remain in `dataset/LSdata`, while
YOLO models and cached features remain in the bind-mounted directories under
`label-studio-ml-backend/label_studio_ml/examples/yolo`.

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
docker compose logs --tail 100 yolo
Invoke-RestMethod http://localhost:9090/health
```

Stop the services without deleting their persistent data:

```powershell
docker compose down
```

### Windows line-ending safeguard

Windows Git checkouts can convert `start.sh` from LF to CRLF. Linux then
interprets the shebang as `/bin/bash\r` and reports:

```text
exec /app/start.sh: no such file or directory
```

The unified Compose command normalizes the script every time the YOLO
container starts. The YOLO Dockerfile also contains this build-time safeguard
after `COPY . ./`:

```dockerfile
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh
```

## Labeling interface

Use [timeline-labeling-config.xml](./timeline-labeling-config.xml) in
**Project Settings > Labeling Interface > Code**.

The configuration keeps the project-specific labels and enables the trainable
timeline classifier:

TODO: Update, note that guard labels were sepparated from striking labels in diofferent projects, so the xml below does not compile all, but there are 2 different.

```xml
<View>
  <TimelineLabels
    name="videoLabels"
    toName="video"
    model_trainable="true"
    model_classifier_epochs="1000"
    model_classifier_sequence_size="16"
    model_classifier_hidden_size="32"
    model_classifier_num_layers="1"
    model_classifier_f1_threshold="0.95"
    model_classifier_accuracy_threshold="0.99"
    model_score_threshold="0.5"
  >
    <Label value="guard_up" background="#FFA39E"/>
    <Label value="guard_down" background="#D4380D"/>
    <Label value="background" background="#FFA39E"/>
    <Label value="punch" background="#D4380D"/>
    <Label value="elbow" background="#FFC069"/>
    <Label value="kick" background="#AD8B00"/>
    <Label value="knee" background="#D3F261"/>
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

With 30 FPS and `model_classifier_sequence_size="16"`, each classifier
sequence covers approximately 0.53 seconds. Changing the sequence size,
hidden size, number of layers, or set of labels resets the saved classifier.

## Connect and train the model

In the Label Studio project:

1. Open **Settings > Model > Connect Model**.
2. Set **Name** to `YOLO Timeline`.
3. Set **Backend URL** to `http://yolo:9090`.
4. Select no authentication method.
5. Leave **Interactive preannotations** off.
6. Validate and save the connection.

Train and use the model as follows:

1. Manually annotate and submit several representative videos.
2. Each annotation creation or update incrementally trains the LSTM
   classifier.
3. Continue until predictions begin appearing on new tasks.
4. Review, correct, and submit those predictions to continue training.

The tutorial recommends roughly 10–20 well-annotated videos of about 500
frames each before expecting meaningful predictions. This backend is a
demonstration model: it trains the LSTM classifier on YOLO features, not the
YOLO feature extractor itself, and class balance matters.

References:

- [TimelineLabels YOLO tutorial](https://labelstud.io/guide/ml_tutorials/yolo_timeline_labels)
- [Label Studio ML backend Docker networking](https://labelstud.io/guide/ml#localhost-and-Docker-containers)


## Which videos to use:

The configuration above assumes 30fps videos at CFR. found only on [here](../media/videos/30fps) when processed using [this bash file](../media/videos/preprocess_fps.sh)