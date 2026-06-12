"""
models/schemas.py
-----------------
Pydantic models for all API request/response validation.
FastAPI uses these to auto-generate Swagger docs.
"""

from pydantic import BaseModel, Field
from typing import Any


# ── Request ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="Sepal length in cm", examples=[5.1])
    sepal_width:  float = Field(..., gt=0, lt=20, description="Sepal width in cm",  examples=[3.5])
    petal_length: float = Field(..., gt=0, lt=20, description="Petal length in cm", examples=[1.4])
    petal_width:  float = Field(..., gt=0, lt=20, description="Petal width in cm",  examples=[0.2])

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width":  3.5,
                "petal_length": 1.4,
                "petal_width":  0.2,
            }
        }


# ── Responses ──────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    predicted_class: str
    class_index:     int
    probabilities:   dict[str, float]
    input_features:  dict[str, float]


class MetricsResponse(BaseModel):
    accuracy:         float
    f1_macro:         float
    f1_weighted:      float
    precision_macro:  float
    recall_macro:     float
    confusion_matrix: list[list[int]]
    class_names:      list[str]
    per_class:        dict[str, Any]
    split_info:       dict[str, Any]
    report:           str


class DatasetInfoResponse(BaseModel):
    total_samples:       int
    n_features:          int
    n_classes:           int
    class_names:         list[str]
    feature_names:       list[str]
    class_distribution:  dict[str, int]
    feature_stats:       dict[str, Any]


class HealthResponse(BaseModel):
    status:  str
    app:     str
    version: str
    trained: bool
