"""Central configuration: paths, schema and constants shared by every stage."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "heart_raw.csv"
CLEAN_CSV = DATA_DIR / "processed" / "heart_clean.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
MODEL_META_PATH = MODELS_DIR / "model_meta.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-risk")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Schema (UCI Cleveland "processed" file, 14 attributes) -----------------
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous measurements -> impute(median) + scale
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Engineered continuous features (added by HeartFeatureBuilder)
ENGINEERED_FEATURES = ["hr_reserve_pct", "chol_per_age", "st_risk_flag"]
# Multi-level codes -> impute(mode) + one-hot
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
# Already 0/1 -> impute(mode) only
BINARY_FEATURES = ["sex", "fbs", "exang"]

INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Valid domains, used by cleaning and by the API schema
VALID_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (0.0, 10.0),
}
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}
