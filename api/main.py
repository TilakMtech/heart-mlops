"""FastAPI service for heart-disease risk prediction.

Endpoints
---------
GET  /            service info
GET  /health      liveness/readiness (model loaded?)
POST /predict     single patient -> prediction + confidence
POST /predict/batch
GET  /metrics     Prometheus exposition format
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)

from api.schemas import BatchPrediction, BatchRequest, PatientRecord, Prediction
from heart import __version__
from heart.config import MODEL_META_PATH
from heart.predict import HeartRiskModel


# ----------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handlers = [handler]
    log_file = os.getenv("LOG_FILE")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        handlers.append(fh)
    logger = logging.getLogger("heart_api")
    logger.handlers = handlers
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    logger.propagate = False
    return logger


log = _setup_logging()

# ----------------------------------------------------------------------------- metrics
REQUESTS = Counter("heart_api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency", ["path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served", ["label"])
PROBABILITY = Histogram("heart_api_disease_probability", "Predicted P(disease)",
                        buckets=tuple(i / 10 for i in range(1, 11)))
MODEL_INFO = Gauge("heart_api_model_info", "Loaded model", ["version"])
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if a model is loaded")

STATE: dict = {"model": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_path = os.getenv("MODEL_PATH")
    meta_path = os.getenv("MODEL_META_PATH", str(MODEL_META_PATH))
    try:
        model = HeartRiskModel(model_path, meta_path) if model_path else HeartRiskModel()
        STATE["model"] = model
        MODEL_INFO.labels(version=model.version).set(1)
        MODEL_LOADED.set(1)
        log.info("model loaded", extra={"extra_fields": {"model_version": model.version}})
    except FileNotFoundError as exc:
        MODEL_LOADED.set(0)
        log.error("model load failed", extra={"extra_fields": {"error": str(exc)}})
    yield
    STATE["model"] = None


app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan,
              description="Predicts presence of heart disease from UCI Cleveland features.")


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = rid
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            log.info("request", extra={"extra_fields": {
                "request_id": rid, "method": request.method, "path": path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _model() -> HeartRiskModel:
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _serve(records: list[PatientRecord], rid: str) -> list[Prediction]:
    model = _model()
    raw = [r.model_dump() for r in records]
    results = model.predict(raw)
    out = []
    for rec, res in zip(raw, results):
        PREDICTIONS.labels(res["label"]).inc()
        PROBABILITY.observe(res["probability_disease"])
        log.info("prediction", extra={"extra_fields": {
            "request_id": rid, "input": rec, **res}})
        out.append(Prediction(**res, model_version=model.version, request_id=rid))
    return out


@app.get("/")
def root():
    model = STATE["model"]
    return {"service": "heart-disease-risk-api", "api_version": __version__,
            "model_version": model.version if model else None,
            "docs": "/docs", "predict": "/predict", "metrics": "/metrics"}


@app.get("/health")
def health():
    ok = STATE["model"] is not None
    if not ok:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": STATE["model"].version}


@app.post("/predict", response_model=Prediction)
def predict(record: PatientRecord, request: Request):
    return _serve([record], request.state.request_id)[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(batch: BatchRequest, request: Request):
    return {"predictions": _serve(batch.records, request.state.request_id)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
