import numpy as np
import pandas as pd

from heart.config import ENGINEERED_FEATURES, INPUT_FEATURES
from heart.data import clean, split_xy
from heart.features import HeartFeatureBuilder, build_preprocessor


def test_feature_builder_adds_columns(raw_df):
    X, _ = split_xy(clean(raw_df))
    out = HeartFeatureBuilder().fit_transform(X)
    for col in ENGINEERED_FEATURES:
        assert col in out.columns
    assert len(out) == len(X)


def test_hr_reserve_formula():
    X = pd.DataFrame([{c: 0 for c in INPUT_FEATURES}])
    X[["age", "thalach", "oldpeak"]] = [60, 128, 2.5]
    out = HeartFeatureBuilder().transform(X)
    assert np.isclose(out.loc[0, "hr_reserve_pct"], 128 / 160)
    assert np.isclose(out.loc[0, "age_oldpeak"], 150.0)


def test_st_risk_flag():
    base = {c: 1 for c in INPUT_FEATURES}
    rows = [dict(base, oldpeak=2.5, exang=0), dict(base, oldpeak=0.5, exang=1),
            dict(base, oldpeak=0.5, exang=0)]
    out = HeartFeatureBuilder().transform(pd.DataFrame(rows))
    assert out["st_risk_flag"].tolist() == [1.0, 1.0, 0.0]


def test_preprocessor_output_has_no_nans_and_is_scaled(raw_df):
    X, _ = split_xy(clean(raw_df))
    pre = build_preprocessor()
    Z = pre.fit_transform(X)
    assert not np.isnan(Z).any()
    # first block = scaled numeric features -> mean ~0
    assert np.allclose(Z[:, :5].mean(axis=0), 0, atol=1e-6)


def test_preprocessor_handles_unseen_category(raw_df):
    X, _ = split_xy(clean(raw_df))
    pre = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["restecg"] = 5  # never seen during fit
    assert pre.transform(unseen).shape[0] == 1


def test_preprocessor_without_engineering_is_smaller(raw_df):
    X, _ = split_xy(clean(raw_df))
    a = build_preprocessor(True).fit_transform(X).shape[1]
    b = build_preprocessor(False).fit_transform(X).shape[1]
    assert a == b + len(ENGINEERED_FEATURES)


def test_age_oldpeak_missing_zero_and_input_unchanged():
    X = pd.DataFrame([{c: 1.0 for c in INPUT_FEATURES}] * 3)
    X["oldpeak"] = [0.0, np.nan, 2.0]
    X["age"] = [60.0, 60.0, np.nan]
    original = X.copy(deep=True)
    out = HeartFeatureBuilder().transform(X)
    assert out.loc[0, "age_oldpeak"] == 0.0
    assert out["age_oldpeak"].iloc[1:].isna().all()
    pd.testing.assert_frame_equal(X, original)
