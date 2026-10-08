import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, trained_model_files):
    model_path, meta_path = trained_model_files
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_META_PATH", str(meta_path))
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root(client):
    assert client.get("/").json()["predict"] == "/predict"


def test_predict_returns_prediction_and_confidence(client, sample_record):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_version"] == "test-logreg:0.0.1"
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_without_optional_fields(client, sample_record):
    sample_record.pop("ca")
    sample_record.pop("thal")
    assert client.post("/predict", json=sample_record).status_code == 200


@pytest.mark.parametrize("field,value", [("age", 500), ("cp", 9), ("sex", 2), ("chol", -1)])
def test_predict_rejects_invalid(client, sample_record, field, value):
    sample_record[field] = value
    assert client.post("/predict", json=sample_record).status_code == 422


def test_predict_missing_field(client, sample_record):
    sample_record.pop("age")
    assert client.post("/predict", json=sample_record).status_code == 422


def test_batch(client, sample_record):
    r = client.post("/predict/batch", json={"records": [sample_record] * 3})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_metrics_exposed(client, sample_record):
    client.post("/predict", json=sample_record)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text


def test_service_unavailable_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
