"""Data loading utilities.

Default source is the bundled CSV so everything runs offline. To train on a
real public dataset instead (e.g. Banking77), implement `load_huggingface`
and flip the `source` argument — the downstream pipeline does not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import settings


@dataclass
class Dataset:
    X_train: list[str]
    X_test: list[str]
    y_train: list[str]
    y_test: list[str]

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def load_dataframe(path: Path | None = None) -> pd.DataFrame:
    path = path or settings.data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python scripts/generate_dataset.py` first."
        )
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def load_split(
    path: Path | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> Dataset:
    df = load_dataframe(path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size or settings.test_size,
        random_state=random_state or settings.random_state,
        stratify=df["label"].tolist(),
    )
    return Dataset(X_train, X_test, y_train, y_test)
