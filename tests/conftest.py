"""Shared fixtures. Tests never need network access or the real dataset."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart.config import RAW_COLUMNS
from heart.features import build_pipeline


def _synthetic_raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })
    # signal so models can learn something
    risk = (df["cp"] == 4).astype(int) + df["exang"] + (df["oldpeak"] > 2).astype(int)
    df["num"] = np.where(risk >= 2, rng.integers(1, 5, n), 0)
    df.loc[:2, "ca"] = np.nan
    df.loc[3, "thal"] = np.nan
    return df[RAW_COLUMNS]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return _synthetic_raw()


@pytest.fixture
def raw_csv(tmp_path, raw_df):
    p = tmp_path / "raw.csv"
    raw_df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_record() -> dict:
    return {"age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241, "fbs": 0,
            "restecg": 0, "thalach": 123, "exang": 1, "oldpeak": 0.2, "slope": 2,
            "ca": 0, "thal": 7}


@pytest.fixture
def trained_model_files(tmp_path, raw_df):
    """Fit a small pipeline on synthetic data and write model + meta to tmp."""
    from heart.data import clean, split_xy

    X, y = split_xy(clean(raw_df))
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "model_meta.json"
    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps({"model_name": "test-logreg", "version": "0.0.1",
                                     "threshold": 0.5}))
    return model_path, meta_path
