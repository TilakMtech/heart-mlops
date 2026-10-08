import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heart.data import clean, split_xy
from heart.features import build_pipeline
from heart.predict import HeartRiskModel, _band
from heart.train import candidates


@pytest.mark.parametrize("estimator", [
    LogisticRegression(max_iter=500),
    RandomForestClassifier(n_estimators=20, random_state=0),
])
def test_pipeline_fits_and_predicts_probabilities(raw_df, estimator):
    X, y = split_xy(clean(raw_df))
    pipe = build_pipeline(estimator).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_model_learns_signal(raw_df):
    X, y = split_xy(clean(raw_df))
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    assert pipe.score(X, y) > 0.7


def test_candidates_cover_required_models():
    names = {c.name for c in candidates()}
    assert {"logreg", "random_forest"} <= names
    for c in candidates():
        assert c.grid and c.quick_grid


def test_heart_risk_model_predict(trained_model_files, sample_record):
    model = HeartRiskModel(*trained_model_files)
    [res] = model.predict([sample_record])
    assert res["prediction"] in (0, 1)
    assert 0.5 <= res["confidence"] <= 1.0
    assert 0.0 <= res["probability_disease"] <= 1.0
    assert model.version == "test-logreg:0.0.1"


def test_heart_risk_model_handles_missing_optional(trained_model_files, sample_record):
    rec = dict(sample_record, ca=None, thal=None)
    [res] = HeartRiskModel(*trained_model_files).predict([rec])
    assert res["prediction"] in (0, 1)


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HeartRiskModel(tmp_path / "missing.joblib")


@pytest.mark.parametrize("p,band", [(0.1, "low"), (0.45, "moderate"), (0.9, "high")])
def test_risk_band(p, band):
    assert _band(p) == band
