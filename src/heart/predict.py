"""Inference helper shared by the API, notebooks and CLI.

    python -m heart.predict scripts/sample_request.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from heart.config import INPUT_FEATURES, MODEL_META_PATH, MODEL_PATH


class HeartRiskModel:
    def __init__(self, model_path: Path | str = MODEL_PATH, meta_path: Path | str = MODEL_META_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}; run `make train`")
        self.pipeline = joblib.load(self.model_path)
        meta_path = Path(meta_path)
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.threshold = float(self.meta.get("threshold", 0.5))

    @property
    def version(self) -> str:
        return f"{self.meta.get('model_name', 'unknown')}:{self.meta.get('version', 'dev')}"

    def predict(self, records: list[dict]) -> list[dict]:
        frame = pd.DataFrame(records, columns=INPUT_FEATURES).astype(float)
        proba = self.pipeline.predict_proba(frame)[:, 1]
        out = []
        for p in proba:
            pred = int(p >= self.threshold)
            out.append({
                "prediction": pred,
                "label": "heart_disease" if pred else "no_heart_disease",
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if pred else 1 - p), 4),
                "risk_band": _band(float(p)),
            })
        return out


def _band(p: float) -> str:
    if p < 0.3:
        return "low"
    if p < 0.6:
        return "moderate"
    return "high"


if __name__ == "__main__":
    payload = json.loads(Path(sys.argv[1]).read_text())
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(HeartRiskModel().predict(records), indent=2))
