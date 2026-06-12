"""
backend/app/api/routes/predictions.py
========================================
Delay prediction endpoints.

POST /api/v1/predictions/delay         — single order
POST /api/v1/predictions/delay/batch   — up to 100 orders
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_ml_service
from backend.app.schemas.predictions import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DelayPrediction,
    OrderFeatures,
)
from backend.app.services.ml.service import DelayPredictionService

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/delay", response_model=DelayPrediction)
def predict_delay(
    order: OrderFeatures,
    svc: DelayPredictionService = Depends(get_ml_service),
) -> DelayPrediction:
    return svc.predict(order)


@router.post("/delay/batch", response_model=BatchPredictionResponse)
def predict_delay_batch(
    req: BatchPredictionRequest,
    svc: DelayPredictionService = Depends(get_ml_service),
) -> BatchPredictionResponse:
    predictions = svc.predict_batch(req.orders)
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
