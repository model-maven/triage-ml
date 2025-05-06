import pytest
from fastapi.testclient import TestClient

from triageml.config import settings
from triageml.train_baseline import train


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    # Ensure an artifact exists before the API loads its predictor.
    if not settings.baseline_path.exists():
        train()
    yield


@pytest.fixture(scope="module")
def client(trained_model):
    from triageml.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_backend"] == "baseline"


def test_predict_returns_valid_label(client):
    resp = client.post(
        "/predict",
        json={"text": "I was charged twice and need a refund on my invoice."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {
        "account_access",
        "billing",
        "feature_request",
        "general_inquiry",
        "technical",
    }
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["scores"].values()) - 1.0) < 1e-6


def test_predict_validation_error_on_empty(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_batch_prediction(client):
    resp = client.post(
        "/predict/batch",
        json={"texts": ["app keeps crashing on upload", "please add dark mode"]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_metrics_endpoint(client):
    client.post("/predict", json={"text": "how do I reset my password?"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "triageml_requests_total" in resp.text
