"""
api/v1/endpoints/model_info.py
-------------------------------
GET /api/v1/model/metrics  — Confusion matrix, F1, accuracy
GET /api/v1/model/dataset  — Dataset summary / EDA
"""

from fastapi import APIRouter
from app.models.schemas import MetricsResponse, DatasetInfoResponse
from app.services.pipeline import pipeline

router = APIRouter(prefix="/model", tags=["Model Info"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get model evaluation metrics",
    description="Returns accuracy, F1, confusion matrix, and per-class scores.",
)
async def get_metrics() -> MetricsResponse:
    return MetricsResponse(**pipeline.metrics)


@router.get(
    "/dataset",
    response_model=DatasetInfoResponse,
    summary="Get dataset summary",
    description="Returns EDA stats: class distribution, feature ranges, etc.",
)
async def get_dataset_info() -> DatasetInfoResponse:
    return DatasetInfoResponse(**pipeline.data_summary)
