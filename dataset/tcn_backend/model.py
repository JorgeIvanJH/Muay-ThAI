"""Label Studio ML backend that pre-annotates TimelineLabels with our TCNs.

Usage: Inference only (this backend never trains).

How it fits the project
-----------------------
Label Studio (LS) calls this service for the guard and the striking project.
Both projects use the same ``<TimelineLabels>`` interface but different label
sets, so ``setup()`` reads the project's labels through ``self.label_interface``
and matches them against ``models.action_detection.config.TASK_CLASS_NAMES``
to decide which bundle answers. Both TCN bundles and the YOLO pose model are
loaded once per worker process (``get_engine``) because the SDK instantiates a
fresh model object for every HTTP request.

Inference reuses the live pipeline's building blocks unchanged
(``models/action_detection/offline.py`` -> ``realtime/pose.py``,
``realtime/actions.py``, ``realtime/windows.py``), so the features are exactly
the training features.

Delivery
--------
A full video takes minutes (YOLO dominates) while LS aborts ``/predict`` after
``ML_TIMEOUT_PREDICT`` (100 s by default). Therefore in the default
``PREDICT_MODE=async`` the ``/predict`` handler only schedules a background job
and returns no result; the job runs the video and posts the prediction back to
LS through ``POST /api/predictions`` (SDK ``predictions.create``). The
prediction appears on the timeline when the annotator reloads the task.
``PREDICT_MODE=sync`` answers inline; use it for tests, debugging and short
clips.

Prediction format (per task)
----------------------------
Frames are 1-based and inclusive, matching Label Studio's timeline with
``frameRate="30.0"`` and ``dataset/build_action_joint_dataset.py``::

    {
      "model_version": "tcn_guard@2026-09-06T13:38:59",
      "score": 0.93,
      "result": [
        {"id": "tcn_guard_1_81", "type": "timelinelabels",
         "from_name": "videoLabels", "to_name": "video",
         "value": {"ranges": [{"start": 1, "end": 81}],
                   "timelinelabels": ["background"]},
         "score": 0.97},
        ...
      ]
    }

Regions are the per-frame argmax merged into runs: one label per region,
non-overlapping, covering frames 1..N (``background`` included).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional


# Make ``models.*`` importable both inside the image (/app/models) and when the
# backend runs from the repository checkout (dataset/tcn_backend -> repo root).
BACKEND_DIR = Path(__file__).resolve().parent
for _candidate in (BACKEND_DIR, *BACKEND_DIR.parents):
    if (_candidate / "models" / "action_detection").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk import LabelStudio
from ultralytics import YOLO

from models.action_detection.config import TASK_CLASS_NAMES, validate_task_labels
from models.action_detection.offline import (
    LabelRange,
    pose_predict_arguments,
    predict_video_probabilities,
    probabilities_to_ranges,
)
from models.action_detection.TCN.infer import (
    DEFAULT_GUARD_WEIGHTS,
    DEFAULT_STRIKING_WEIGHTS,
    load_runtime,
)
from models.yolo import config as yolocfg


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration (environment)                                                  #
# --------------------------------------------------------------------------- #

PREDICT_MODE = os.getenv("PREDICT_MODE", "async").strip().lower()
if PREDICT_MODE not in ("async", "sync"):
    raise ValueError("PREDICT_MODE must be 'async' or 'sync'")

GUARD_WEIGHTS = Path(os.getenv("GUARD_WEIGHTS") or DEFAULT_GUARD_WEIGHTS)
STRIKING_WEIGHTS = Path(os.getenv("STRIKING_WEIGHTS") or DEFAULT_STRIKING_WEIGHTS)
POSE_WEIGHTS = Path(os.getenv("POSE_WEIGHTS") or yolocfg.YOLO_REALTIME_WEIGHTS)
TCN_DEVICE = os.getenv("TCN_DEVICE", "cpu")
YOLO_DEVICE = os.getenv("YOLO_DEVICE") or ("0" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "0")) or None
PROGRESS_EVERY_FRAMES = int(os.getenv("PROGRESS_EVERY_FRAMES", "600"))
RESULT_DIR = Path(os.getenv("MODEL_DIR") or (BACKEND_DIR / "data")) / "results"

LABEL_STUDIO_URL = (
    os.getenv("LABEL_STUDIO_URL") or os.getenv("LABEL_STUDIO_HOST") or ""
).rstrip("/")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")


# --------------------------------------------------------------------------- #
# Process-wide engine: models loaded once, one inference job at a time         #
# --------------------------------------------------------------------------- #


def _bundle_version(weights: Path) -> str:
    """Build a traceable model version from the bundle file and its metadata.

    Usage: Inference only.

    The default bundle names are stable (``tcn_guard.pt``), so the training
    timestamp saved by ``TCN/train.py`` is appended; it identifies the copy in
    ``weights/runs/``.
    """

    bundle = torch.load(weights, map_location="cpu", weights_only=False)
    trained_at = bundle.get("trained_at")
    if not trained_at:
        from datetime import datetime

        trained_at = datetime.fromtimestamp(weights.stat().st_mtime).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    return f"{weights.stem}@{trained_at}"


class _Engine:
    """Loaded models plus the single-worker queue for whole-video jobs."""

    def __init__(self) -> None:
        device = torch.device(TCN_DEVICE)
        logger.info(
            "Loading TCN bundles on %s: guard=%s striking=%s", device,
            GUARD_WEIGHTS, STRIKING_WEIGHTS,
        )
        self.runtimes = (
            load_runtime(GUARD_WEIGHTS, device, classification_task="guard"),
            load_runtime(STRIKING_WEIGHTS, device, classification_task="striking"),
        )
        self.model_versions = {
            "guard": _bundle_version(GUARD_WEIGHTS),
            "striking": _bundle_version(STRIKING_WEIGHTS),
        }
        logger.info("Loading pose model %s (device %s)", POSE_WEIGHTS, YOLO_DEVICE)
        self.pose_model = YOLO(str(POSE_WEIGHTS))
        self.predict_arguments = pose_predict_arguments(YOLO_DEVICE)
        # The YOLO predictor must have one user at a time (same rule as the
        # real-time pose worker); the executor serialises whole-video jobs and
        # the lock also protects PREDICT_MODE=sync calls.
        self.inference_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tcn-preannotate",
        )
        self._jobs: Dict[str, Future] = {}
        self._jobs_lock = threading.Lock()
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Engine ready: %s", self.model_versions)

    def submit(self, key: str, function, *args) -> bool:
        """Queue a job unless the same key is already pending or running.

        Usage: Inference only.
        """

        with self._jobs_lock:
            pending = self._jobs.get(key)
            if pending is not None and not pending.done():
                return False
            future = self._executor.submit(_run_logged, key, function, *args)
            self._jobs[key] = future
            return True

    def pending_jobs(self) -> int:
        with self._jobs_lock:
            return sum(1 for future in self._jobs.values() if not future.done())


def _run_logged(key: str, function, *args) -> None:
    """Run one background job and log any failure with its traceback.

    Usage: Inference only.
    """

    logger.info("Pre-annotation job started: %s", key)
    try:
        function(*args)
        logger.info("Pre-annotation job finished: %s", key)
    except Exception:  # noqa: BLE001 - background thread, must not die silently
        logger.error("Pre-annotation job failed: %s\n%s", key, traceback.format_exc())


_ENGINE: Optional[_Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> _Engine:
    """Return the process-wide engine, creating it on first use.

    Usage: Inference only.
    """

    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _Engine()
        return _ENGINE


# --------------------------------------------------------------------------- #
# Label config -> task                                                         #
# --------------------------------------------------------------------------- #


def resolve_task(labels, *, source: str) -> str:
    """Return the task whose vocabulary equals the project's label set.

    Usage: Inference only.

    Exactly one of ``TASK_CLASS_NAMES`` must accept the labels with
    ``require_all=True``; anything else (mixed, partial, unknown) is rejected
    so guard weights are never applied to a striking project or vice versa.
    """

    observed = [str(label) for label in labels]
    matches = []
    errors = []
    for task in TASK_CLASS_NAMES:
        try:
            validate_task_labels(task, observed, source=source, require_all=True)
            matches.append(task)
        except ValueError as error:
            errors.append(str(error))
    if len(matches) != 1:
        raise ValueError(
            f"{source} labels {sorted(observed)} match "
            f"{'no' if not matches else 'several'} classification task(s). "
            + " | ".join(errors)
        )
    return matches[0]


def _timeline_control(label_interface):
    """Return the single TimelineLabels control of the labeling config.

    Usage: Inference only.
    """

    controls = [
        control
        for control in label_interface.controls
        if control.tag == "TimelineLabels"
    ]
    if len(controls) != 1:
        raise ValueError(
            f"Expected exactly one <TimelineLabels> control, found {len(controls)}"
        )
    control = controls[0]
    if not control.to_name or not control.objects:
        raise ValueError("<TimelineLabels> must point to a <Video> via toName")
    return control


# --------------------------------------------------------------------------- #
# The backend                                                                  #
# --------------------------------------------------------------------------- #


class TCNTimelineModel(LabelStudioMLBase):
    """Pre-annotate TimelineLabels with the guard or striking TCN."""

    def setup(self) -> None:
        """Pick the task from the project's labels and attach the shared engine.

        Usage: Inference only.

        Called for every request (the SDK builds a new instance each time), so
        nothing heavy happens here; models live in ``get_engine()``.
        """

        self.engine = get_engine()
        label_interface = getattr(self, "label_interface", None)
        if label_interface is None:
            # ``label-studio-ml start --check`` instantiates without a config.
            logger.warning("No label config provided; task resolution skipped")
            self.task_name = None
            return
        control = _timeline_control(label_interface)
        self.task_name = resolve_task(
            control.labels,
            source=f"Project {self.project_id or '?'} labeling config",
        )
        self.from_name = control.name
        self.to_name = control.to_name[0]
        self.value_key = control.objects[0].value_name
        self.set("model_version", self.engine.model_versions[self.task_name])
        logger.debug(
            "Project %s -> task %s (%s -> %s, data key %s)",
            self.project_id, self.task_name, self.from_name, self.to_name,
            self.value_key,
        )

    # -- predict ------------------------------------------------------------ #

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        """Answer inline (sync) or schedule jobs that post back to LS (async).

        Usage: Inference only.
        """

        if self.task_name is None:
            raise RuntimeError("predict called without a labeling config")
        version = self.engine.model_versions[self.task_name]

        if PREDICT_MODE == "sync":
            predictions = [self._predict_task(task) for task in tasks]
            return ModelResponse(model_version=version, predictions=predictions)

        scheduled = 0
        for task in tasks:
            key = f"{self._cache_key(task)}"
            if self.engine.submit(key, self._predict_and_post, task):
                scheduled += 1
        logger.info(
            "Scheduled %d/%d task(s) for %s pre-annotation (%d pending)",
            scheduled, len(tasks), self.task_name, self.engine.pending_jobs(),
        )
        # Results are delivered through POST /api/predictions by the job, so
        # nothing is returned here: LS would otherwise store a duplicate.
        return ModelResponse(model_version=version, predictions=[])

    def _cache_key(self, task: Dict) -> str:
        video_value = str(task.get("data", {}).get(self.value_key, ""))
        digest = hashlib.sha1(video_value.encode("utf-8")).hexdigest()[:10]
        version = self.engine.model_versions[self.task_name].replace(":", "-")
        return f"task{task.get('id', 'x')}__{self.task_name}__{digest}__{version}"

    def _cache_path(self, task: Dict) -> Path:
        return RESULT_DIR / f"{self._cache_key(task)}.json"

    def _local_video_path(self, task: Dict) -> str:
        """Resolve the task's video to a readable local file.

        Usage: Inference only.

        LS uploads are served under ``/data/upload/...``; the SDK downloads them
        with ``LABEL_STUDIO_URL`` + ``LABEL_STUDIO_API_KEY`` and caches the file.
        Absolute paths that already exist (tests, debugging) are used directly.
        """

        video_value = task["data"][self.value_key]
        if os.path.exists(video_value):
            return video_value
        return self.get_local_path(video_value, task_id=task.get("id"))

    def _predict_task(self, task: Dict) -> Dict:
        """Run the whole video (or load the cached result) for one task.

        Usage: Inference only.
        """

        cache_path = self._cache_path(task)
        if cache_path.is_file():
            logger.info("Using cached prediction %s", cache_path.name)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        version = self.engine.model_versions[self.task_name]
        video_path = self._local_video_path(task)
        logger.info(
            "Running %s TCN on task %s (%s)", self.task_name, task.get("id"),
            video_path,
        )

        def progress(frame_index: int) -> None:
            if PROGRESS_EVERY_FRAMES and frame_index % PROGRESS_EVERY_FRAMES == 0:
                logger.info("task %s: %d frames", task.get("id"), frame_index)

        with self.engine.inference_lock:
            result = predict_video_probabilities(
                video_path,
                pose_model=self.engine.pose_model,
                runtimes=self.engine.runtimes,
                predict_arguments=self.engine.predict_arguments,
                max_frames=MAX_FRAMES,
                progress=progress,
            )
        ranges = probabilities_to_ranges(
            result.probabilities[self.task_name],
            result.class_names[self.task_name],
        )
        prediction = self._format_prediction(ranges, version)
        cache_path.write_text(json.dumps(prediction), encoding="utf-8")
        logger.info(
            "task %s: %d frames -> %d regions (%s)", task.get("id"),
            result.frame_count, len(ranges), version,
        )
        return prediction

    def _format_prediction(self, ranges: List[LabelRange], version: str) -> Dict:
        """Convert label runs into one Label Studio prediction dictionary.

        Usage: Inference only.
        """

        prefix = version.split("@", 1)[0]
        regions = [
            {
                "id": f"{prefix}_{item.start}_{item.end}",
                "type": "timelinelabels",
                "from_name": self.from_name,
                "to_name": self.to_name,
                "value": {
                    "ranges": [{"start": item.start, "end": item.end}],
                    "timelinelabels": [item.label],
                },
                "score": round(float(item.score), 4),
            }
            for item in ranges
        ]
        score = float(np.mean([item.score for item in ranges])) if ranges else 0.0
        return {
            "model_version": version,
            "score": round(score, 4),
            "result": regions,
        }

    # -- async delivery ----------------------------------------------------- #

    def _predict_and_post(self, task: Dict) -> None:
        """Background job: predict (or reuse the cache) and post to LS once.

        Usage: Inference only.
        """

        prediction = self._predict_task(task)
        self._post_prediction(task, prediction)

    def _post_prediction(self, task: Dict, prediction: Dict) -> None:
        """Create the prediction in Label Studio unless this version exists.

        Usage: Inference only.
        """

        if not LABEL_STUDIO_URL or not LABEL_STUDIO_API_KEY:
            raise RuntimeError(
                "LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY are required to post "
                "predictions back to Label Studio"
            )
        task_id = int(task["id"])
        client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
        existing = client.predictions.list(task=task_id)
        if any(item.model_version == prediction["model_version"] for item in existing):
            logger.info(
                "task %s already has a %s prediction; not posting again",
                task_id, prediction["model_version"],
            )
            return
        create_arguments = {
            "task": task_id,
            "result": prediction["result"],
            "score": prediction["score"],
            "model_version": prediction["model_version"],
        }
        if str(self.project_id).isdigit():
            create_arguments["project"] = int(self.project_id)
        created = client.predictions.create(**create_arguments)
        logger.info(
            "Posted prediction %s for task %s (%d regions, %s)",
            getattr(created, "id", "?"), task_id, len(prediction["result"]),
            prediction["model_version"],
        )

    # -- fit ---------------------------------------------------------------- #

    def fit(self, event, data, **kwargs):
        """Do nothing: Label Studio's annotation webhook must not train here.

        Usage: Inference only.

        Retraining stays in ``models/action_detection/TCN/train.py`` from the
        exported labels. An incremental "retrain from new exports" step would
        plug in here: on ``ANNOTATION_UPDATED`` export the project with the SDK,
        run ``dataset/build_action_joint_dataset.py`` and ``TCN/train.py`` in a
        separate process, then drop new bundles into ``TCN/weights`` (the mounted
        folder) and restart the service so ``get_engine()`` reloads them.
        """

        project = (data or {}).get("project") or {}
        logger.info(
            "fit ignored (event=%s, project=%s): this backend never trains",
            event, project.get("id"),
        )
        return {"status": "ignored", "event": event}


# Load the models when the worker process boots (gunicorn --preload imports this
# module) rather than on the first request: Label Studio gives /setup only
# ML_TIMEOUT_SETUP = 3 s, which is less than loading YOLO plus two TCN bundles.
if os.getenv("TCN_BACKEND_LAZY_LOAD", "").strip().lower() not in ("1", "true"):
    get_engine()
