"""API integration tests. Requires a Production-stage model to already be
registered in mlruns/ (committed to the repo -- see docs/architecture.md) --
run scripts/train.py first if mlruns/ is missing or empty.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app


@pytest.fixture(scope="module")
def client():
    # Context manager, not a bare TestClient(app) -- only this form runs the
    # lifespan startup/shutdown handlers (see api/main.py), so the model is
    # actually loaded before these tests hit the endpoints.
    with TestClient(app) as c:
        yield c


def _sample_transaction():
    payload = {"Time": 10000.0, "Amount": 49.99}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    return payload


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info_returns_production_model(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "Production"
    assert "pr_auc" in body["val_metrics"]


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=_sample_transaction())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["model_stage"] == "Production"
    assert body["latency_ms"] > 0


def test_predict_rejects_missing_field(client):
    payload = _sample_transaction()
    del payload["V1"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_amount(client):
    payload = _sample_transaction()
    payload["Amount"] = -10.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
