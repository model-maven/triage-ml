"""Lightweight production monitoring.

Two things every deployed model needs and most portfolio projects skip:

1. **Prediction logging** — append every request/response to a JSONL file so you
   can audit, build a labelled feedback set, and compute live metrics.
2. **Drift detection** — compare the label distribution of recent traffic against
   the training distribution using Population Stability Index (PSI). A rising PSI
   is an early warning that the world has shifted and the model needs retraining.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from .config import settings


def log_prediction(text: str, prediction: dict, path: Path | None = None) -> None:
    path = path or settings.log_predictions_path
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": text,
        **prediction,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _distribution(labels: Iterable[str], classes: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    total = sum(counts.values()) or 1
    # Laplace smoothing avoids zero buckets that would blow up the log term.
    return {c: (counts.get(c, 0) + 1) / (total + len(classes)) for c in classes}


def population_stability_index(
    reference: Iterable[str],
    current: Iterable[str],
    classes: list[str],
) -> float:
    """PSI over a categorical label distribution.

    Rule of thumb: <0.1 no shift, 0.1-0.25 moderate shift, >0.25 significant.
    """
    ref = _distribution(reference, classes)
    cur = _distribution(current, classes)
    return sum((cur[c] - ref[c]) * math.log(cur[c] / ref[c]) for c in classes)


def drift_report(
    reference_labels: Iterable[str],
    current_labels: Iterable[str],
    classes: list[str],
) -> dict:
    psi = population_stability_index(reference_labels, current_labels, classes)
    if psi < 0.1:
        verdict = "stable"
    elif psi < 0.25:
        verdict = "moderate_drift"
    else:
        verdict = "significant_drift"
    return {"psi": round(psi, 4), "verdict": verdict}
