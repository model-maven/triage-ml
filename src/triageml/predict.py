"""Inference layer that abstracts over both model backends.

The API depends only on `Predictor`, so swapping baseline <-> transformer is a
one-line config change with no API code touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib

from .config import settings


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    needs_review: bool = False

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


class Predictor:
    """Loads a model artifact and produces calibrated-ish predictions."""

    def __init__(self, backend: str | None = None, threshold: float | None = None):
        self.backend = backend or settings.model_backend
        self.threshold = (
            threshold if threshold is not None else settings.confidence_threshold
        )
        self._model = None
        self._kind: str | None = None
        self._load()

    def _load(self) -> None:
        if self.backend == "transformer" and settings.transformer_path.exists():
            self._load_transformer(settings.transformer_path)
        elif settings.baseline_path.exists():
            self._model = joblib.load(settings.baseline_path)
            self._kind = "baseline"
        else:
            raise FileNotFoundError(
                "No trained artifact found. Train one with "
                "`python -m triageml.train_baseline`."
            )

    def _load_transformer(self, path: Path) -> None:  # pragma: no cover
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self._model.eval()
        self._torch = torch
        self._kind = "transformer"

    @property
    def kind(self) -> str:
        return self._kind or "unknown"

    def predict(self, text: str) -> Prediction:
        if self._kind == "baseline":
            return self._predict_baseline(text)
        return self._predict_transformer(text)  # pragma: no cover

    def _predict_baseline(self, text: str) -> Prediction:
        proba = self._model.predict_proba([text])[0]
        classes = list(self._model.classes_)
        scores = {c: float(p) for c, p in zip(classes, proba, strict=True)}
        label = max(scores, key=scores.get)
        confidence = scores[label]
        return Prediction(
            label=label,
            confidence=confidence,
            scores=scores,
            needs_review=confidence < self.threshold,
        )

    def _predict_transformer(self, text: str) -> Prediction:  # pragma: no cover
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_length,
        )
        with self._torch.no_grad():
            logits = self._model(**enc).logits
        probs = self._torch.softmax(logits, dim=-1)[0].tolist()
        id2label = self._model.config.id2label
        scores = {id2label[i]: float(p) for i, p in enumerate(probs)}
        label = max(scores, key=scores.get)
        confidence = scores[label]
        return Prediction(
            label=label,
            confidence=confidence,
            scores=scores,
            needs_review=confidence < self.threshold,
        )


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Cached singleton so the model loads once per process."""
    return Predictor()
