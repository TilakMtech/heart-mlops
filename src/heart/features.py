"""Feature engineering + preprocessing pipeline.

The full sklearn Pipeline (feature builder -> ColumnTransformer -> estimator)
is persisted as one object, so training and serving apply identical transforms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES, INPUT_FEATURES, NUMERIC_FEATURES,
)


class HeartFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds clinically-motivated derived features.

    hr_reserve_pct : thalach / (220 - age)  -> share of age-predicted max HR reached
                     during the exercise test (chronotropic response).
    chol_per_age   : chol / age             -> cholesterol load relative to age.
    st_risk_flag   : 1 if oldpeak >= 2.0 or exercise-induced angina, else 0.
    """

    def fit(self, X, y=None):  # stateless
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=INPUT_FEATURES).copy()
        age = X["age"].astype(float)
        predicted_max = (220.0 - age).where(lambda s: s > 0)
        X["hr_reserve_pct"] = X["thalach"] / predicted_max
        X["chol_per_age"] = X["chol"] / age.replace(0, np.nan)
        X["st_risk_flag"] = ((X["oldpeak"] >= 2.0) | (X["exang"] == 1)).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(INPUT_FEATURES + ENGINEERED_FEATURES)


def build_preprocessor(use_engineered: bool = True) -> Pipeline:
    numeric = NUMERIC_FEATURES + (ENGINEERED_FEATURES if use_engineered else [])
    ct = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps = [("features", HeartFeatureBuilder())] if use_engineered else []
    return Pipeline(steps + [("preprocess", ct)])


def build_pipeline(estimator, use_engineered: bool = True) -> Pipeline:
    pre = build_preprocessor(use_engineered)
    return Pipeline(pre.steps + [("model", estimator)])


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
