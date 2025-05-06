"""FastAPI serving layer.

Endpoints
---------
GET  /health   liveness + which backend is loaded
POST /predict  classify a single ticket
POST /predict/batch  classify many tickets
GET  /metrics  Prometheus-style counters for scraping

The model is loaded once at startup via the cached predictor and reused across
requests. Every prediction is logged for monitoring/feedback collection.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from .. import __version__
from ..monitoring import log_prediction
from ..predict import get_predictor
from .schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictionResponse,
    TicketRequest,
)

# Minimal in-process counters. In production you'd use prometheus_client, but
# this keeps the dependency surface small while showing the pattern.
_METRICS = {"requests_total": 0, "predictions_total": 0, "needs_review_total": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model at startup so the first request isn't slow.
    get_predictor()
    yield


app = FastAPI(
    title="TriageML",
    description="Production NLP service that auto-classifies support tickets.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model_backend=predictor.kind,
        version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TicketRequest) -> PredictionResponse:
    predictor = get_predictor()
    result = predictor.predict(req.text).as_dict()

    _METRICS["requests_total"] += 1
    _METRICS["predictions_total"] += 1
    if result["needs_review"]:
        _METRICS["needs_review_total"] += 1

    log_prediction(req.text, result)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    predictor = get_predictor()
    _METRICS["requests_total"] += 1
    out: list[PredictionResponse] = []
    for text in req.texts:
        result = predictor.predict(text).as_dict()
        _METRICS["predictions_total"] += 1
        if result["needs_review"]:
            _METRICS["needs_review_total"] += 1
        log_prediction(text, result)
        out.append(PredictionResponse(**result))
    return BatchResponse(predictions=out)


@app.get("/metrics")
def metrics() -> Response:
    lines = [
        "# HELP triageml_requests_total Total API requests.",
        "# TYPE triageml_requests_total counter",
        f"triageml_requests_total {_METRICS['requests_total']}",
        "# HELP triageml_predictions_total Total predictions produced.",
        "# TYPE triageml_predictions_total counter",
        f"triageml_predictions_total {_METRICS['predictions_total']}",
        "# HELP triageml_needs_review_total Predictions flagged for human review.",
        "# TYPE triageml_needs_review_total counter",
        f"triageml_needs_review_total {_METRICS['needs_review_total']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")
