"""API request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=[
        "I was charged twice this month, please refund the duplicate.",
    ])


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128)


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    needs_review: bool
    scores: dict[str, float]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_backend: str
    version: str
