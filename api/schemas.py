"""Request / response contracts for the prediction API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    """One patient. Codes follow the UCI Cleveland documentation.

    `ca` and `thal` are optional: the pipeline imputes them if missing.
    """

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241, "fbs": 0,
        "restecg": 0, "thalach": 123, "exang": 1, "oldpeak": 0.2, "slope": 2,
        "ca": 0, "thal": 7,
    }})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (1 typical .. 4 asymptomatic)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(None, description="Major vessels coloured (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed, 7 reversible")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_band: str
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    records: list[PatientRecord] = Field(..., min_length=1, max_length=500)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
