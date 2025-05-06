"""Train the scikit-learn baseline (TF-IDF + Logistic Regression).

A strong, fast baseline is essential in real ML work: it sets the bar the
heavier transformer must beat, and it's cheap enough to ship as a fallback.

Usage:
    python -m triageml.train_baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from .config import settings
from .data import load_split


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=4.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(artifacts_dir: Path | None = None) -> dict:
    artifacts_dir = artifacts_dir or settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = load_split()
    pipeline = build_pipeline()

    start = time.perf_counter()
    pipeline.fit(data.X_train, data.y_train)
    train_seconds = time.perf_counter() - start

    preds = pipeline.predict(data.X_test)
    macro_f1 = f1_score(data.y_test, preds, average="macro")
    report = classification_report(data.y_test, preds, output_dict=True)

    out_path = artifacts_dir / settings.baseline_filename
    joblib.dump(pipeline, out_path)

    metrics = {
        "model": "baseline_tfidf_logreg",
        "macro_f1": round(float(macro_f1), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "n_train": data.n_train,
        "n_test": data.n_test,
        "train_seconds": round(train_seconds, 3),
        "artifact": str(out_path),
    }
    (artifacts_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
