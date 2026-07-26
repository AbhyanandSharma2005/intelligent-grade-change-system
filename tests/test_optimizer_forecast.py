import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_optimizer_never_increases_risk(client):
    r = client.get("/optimize/0", params={"at_step": 200})
    assert r.status_code == 200
    body = r.json()
    assert body["risk_after"] <= body["risk_before"] + 1e-6
    assert len(body["setpoints"]) == 4


def test_forecast_contract(client):
    r = client.get("/forecast/0", params={"at_step": 200})
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) == 3
    for p in body["points"]:
        assert p["dev_pct_p10"] <= p["dev_pct_p50"] <= p["dev_pct_p90"]
