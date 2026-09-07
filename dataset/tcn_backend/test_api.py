"""API tests for the TCN pre-annotation backend.

Run from this directory in the ``muay-thai-ls`` env (needs the Label Studio ML
SDK; ``muay-thai`` does not have it):

    conda run -n muay-thai-ls pytest dataset/tcn_backend/test_api.py -q

The tests run the real YOLO + TCN models on the first ``MAX_FRAMES`` frames of a
local 30 FPS clip from ``media/videos/30fps`` and are skipped when no clip or no
weights are available. ``PREDICT_MODE=sync`` makes ``/predict`` answer inline.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Must be set before ``model`` is imported: it reads the environment at import.
os.environ["PREDICT_MODE"] = "sync"
os.environ.setdefault("MAX_FRAMES", "90")
os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="tcn_backend_test_"))
os.environ.setdefault("LOG_LEVEL", "WARNING")

from model import TCNTimelineModel  # noqa: E402

GUARD_XML = (ROOT_DIR / "dataset" / "timeline-labeling-guard.xml").read_text(encoding="utf-8")
STRIKING_XML = (ROOT_DIR / "dataset" / "timeline-labeling-striking.xml").read_text(encoding="utf-8")
MIXED_XML = """
<View>
  <TimelineLabels name="videoLabels" toName="video">
    <Label value="background"/>
    <Label value="guard_up"/>
    <Label value="punch"/>
  </TimelineLabels>
  <Video name="video" value="$video" frameRate="30.0"/>
</View>
"""
MAX_FRAMES = int(os.environ["MAX_FRAMES"])


def _first_clip() -> str:
    clips = sorted((ROOT_DIR / "media" / "videos" / "30fps").glob("*_30fps.mp4"))
    if not clips:
        pytest.skip("no 30 FPS clip available under media/videos/30fps")
    return str(clips[0])


def _weights_available() -> None:
    from model import GUARD_WEIGHTS, STRIKING_WEIGHTS

    if not (GUARD_WEIGHTS.is_file() and STRIKING_WEIGHTS.is_file()):
        pytest.skip("TCN bundles are not available")


@pytest.fixture(scope="module")
def client():
    from _wsgi import init_app

    app = init_app(model_class=TCNTimelineModel)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _predict(client, label_config: str, task_id: int, project: str = "1.1700000000"):
    request = {
        "tasks": [{"id": task_id, "data": {"video": _first_clip()}}],
        "label_config": label_config,
        "project": project,
    }
    response = client.post(
        "/predict",
        data=json.dumps(request),
        content_type="application/json",
    )
    assert response.status_code == 200, response.data
    return json.loads(response.data)["results"]


def _assert_timeline_prediction(prediction: dict, *, version_prefix: str, labels: set):
    assert prediction["model_version"].startswith(version_prefix + "@")
    regions = prediction["result"]
    assert regions, "expected at least one timeline region"
    expected_start = 1
    for region in regions:
        assert region["type"] == "timelinelabels"
        assert region["from_name"] == "videoLabels"
        assert region["to_name"] == "video"
        assert len(region["value"]["ranges"]) == 1
        assert len(region["value"]["timelinelabels"]) == 1
        assert region["value"]["timelinelabels"][0] in labels
        start = region["value"]["ranges"][0]["start"]
        end = region["value"]["ranges"][0]["end"]
        assert start == expected_start, "ranges must be contiguous and 1-based"
        assert end >= start
        assert 0.0 <= region["score"] <= 1.0
        expected_start = end + 1
    assert regions[-1]["value"]["ranges"][0]["end"] == MAX_FRAMES
    assert 0.0 <= prediction["score"] <= 1.0


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert json.loads(response.data)["status"] == "UP"


def test_predict_guard_project_uses_guard_bundle(client):
    _weights_available()
    results = _predict(client, GUARD_XML, task_id=1, project="1.1700000000")
    assert len(results) == 1
    _assert_timeline_prediction(
        results[0],
        version_prefix="tcn_guard",
        labels={"background", "guard_up", "guard_down"},
    )


def test_predict_striking_project_uses_striking_bundle(client):
    _weights_available()
    results = _predict(client, STRIKING_XML, task_id=2, project="2.1700000000")
    assert len(results) == 1
    _assert_timeline_prediction(
        results[0],
        version_prefix="tcn_striking",
        labels={"background", "punch", "elbow", "kick", "knee"},
    )


def test_predict_is_cached_per_task_and_version(client):
    _weights_available()
    first = _predict(client, GUARD_XML, task_id=1, project="1.1700000000")
    second = _predict(client, GUARD_XML, task_id=1, project="1.1700000000")
    assert first == second


def test_setup_rejects_mixed_vocabulary(client):
    response = client.post(
        "/setup",
        data=json.dumps({"project": "9.1700000000", "schema": MIXED_XML}),
        content_type="application/json",
    )
    assert response.status_code == 500
    assert "match no classification task" in response.get_data(as_text=True)


def test_setup_reports_bundle_model_version(client):
    _weights_available()
    response = client.post(
        "/setup",
        data=json.dumps({"project": "1.1700000000", "schema": GUARD_XML}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert json.loads(response.data)["model_version"].startswith("tcn_guard@")


def test_webhook_fit_is_a_no_op(client):
    _weights_available()
    payload = {
        "action": "ANNOTATION_CREATED",
        "project": {"id": 1, "label_config": GUARD_XML},
        "annotation": {"id": 1, "result": []},
    }
    response = client.post(
        "/webhook",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 201
    body = json.loads(response.data)
    assert body["status"] == "ok"
    assert body["result"] == {"status": "ignored", "event": "ANNOTATION_CREATED"}
