"""
api/v1/endpoints/predict.py
----------------------------
POST /api/v1/predict — Classify a flower by its measurements.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictRequest, PredictResponse
from app.services.pipeline import pipeline

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/",
    response_model=PredictResponse,
    summary="Classify an Iris flower",
    description=(
        "Provide the 4 measurements of an Iris flower. "
        "The KNN model will predict whether it's Setosa, Versicolor, or Virginica, "
        "along with per-class probabilities."
    ),
)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Full prediction pipeline:
      1. Accept raw float measurements
      2. Scale using the fitted StandardScaler
      3. Run KNN inference
      4. Return class name + probabilities
    """
    try:
        result = pipeline.predict(
            sepal_length=request.sepal_length,
            sepal_width=request.sepal_width,
            petal_length=request.petal_length,
            petal_width=request.petal_width,
        )
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
