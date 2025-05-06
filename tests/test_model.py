import json

from triageml.monitoring import drift_report, population_stability_index
from triageml.train_baseline import train


def test_baseline_trains_and_beats_random(tmp_path):
    metrics = train(artifacts_dir=tmp_path)
    # 5 classes -> random macro-F1 ~0.2; a real model should clear 0.5 easily.
    assert metrics["macro_f1"] > 0.5
    assert (tmp_path / "baseline.joblib").exists()
    saved = json.loads((tmp_path / "baseline_metrics.json").read_text())
    assert saved["model"] == "baseline_tfidf_logreg"


def test_psi_zero_for_identical_distributions():
    classes = ["a", "b", "c"]
    labels = ["a", "b", "c", "a", "b", "c"]
    assert population_stability_index(labels, labels, classes) == 0.0


def test_drift_report_flags_significant_shift():
    classes = ["a", "b"]
    ref = ["a"] * 90 + ["b"] * 10
    cur = ["b"] * 90 + ["a"] * 10
    report = drift_report(ref, cur, classes)
    assert report["verdict"] == "significant_drift"
    assert report["psi"] > 0.25
