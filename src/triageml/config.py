"""Centralised configuration.

All settings can be overridden via environment variables (12-factor style),
which is how they'd be injected in a container / cloud deployment.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGEML_", env_file=".env")

    # Data
    data_path: Path = PROJECT_ROOT / "data" / "tickets.csv"
    test_size: float = 0.2
    random_state: int = 42

    # Artifacts
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    baseline_filename: str = "baseline.joblib"
    transformer_dirname: str = "transformer"

    # Transformer training
    transformer_base_model: str = "distilbert-base-uncased"
    max_length: int = 64
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5

    # Serving
    model_backend: str = "baseline"  # "baseline" | "transformer"
    confidence_threshold: float = 0.40  # below this -> route to human review
    log_predictions_path: Path = PROJECT_ROOT / "artifacts" / "prediction_log.jsonl"

    @property
    def baseline_path(self) -> Path:
        return self.artifacts_dir / self.baseline_filename

    @property
    def transformer_path(self) -> Path:
        return self.artifacts_dir / self.transformer_dirname


settings = Settings()
