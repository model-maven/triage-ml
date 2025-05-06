"""Compare logged production predictions against the training distribution.

Run this on a schedule (cron / GitHub Action / Airflow). A 'significant_drift'
verdict is your signal to pull fresh labels and retrain.

Usage:
    python scripts/check_drift.py
"""
from __future__ import annotations

import json
import sys

from triageml import LABELS
from triageml.config import settings
from triageml.data import load_dataframe
from triageml.monitoring import drift_report


def main() -> int:
    reference = load_dataframe()["label"].tolist()

    log_path = settings.log_predictions_path
    if not log_path.exists():
        print("No prediction log yet; nothing to check.")
        return 0

    current = [
        json.loads(line)["label"]
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    if not current:
        print("Prediction log is empty.")
        return 0

    report = drift_report(reference, current, LABELS)
    print(json.dumps({"n_observed": len(current), **report}, indent=2))
    # Non-zero exit on significant drift so CI/cron can alert.
    return 1 if report["verdict"] == "significant_drift" else 0


if __name__ == "__main__":
    sys.exit(main())
