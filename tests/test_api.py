import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # triggers lifespan startup
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_episodes_listed(client):
    eps = client.get("/episodes").json()
    assert len(eps) > 0
    assert {"episode_id", "off_spec"} <= set(eps[0])


def test_predict_contract(client):
    r = client.get("/predict/0", params={"at_step": 200})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["risk_level"] in {"low", "elevated", "high"}
    assert len(body["drivers"]) > 0


def test_predict_unknown_episode_404(client):
    assert client.get("/predict/999999").status_code == 404


def test_recommend_and_feedback_roundtrip(client):
    r = client.get("/recommend/0", params={"at_step": 200}).json()
    if r["action_required"]:
        rec = r["recommendations"][0]
        fb = {**{k: rec[k] for k in
                 ["recommendation_id", "tag", "value", "rationale_tag"]},
              "episode_id": 0, "accepted": True}
        assert client.post("/feedback", json=fb).status_code == 201
        summary = client.get("/feedback/summary").json()
        assert any(row["n"] >= 1 for row in summary)
