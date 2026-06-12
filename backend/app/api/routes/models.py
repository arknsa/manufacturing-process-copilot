"""
backend/app/api/routes/models.py
===================================
Model metadata endpoints (read-only).

GET /api/v1/models/current             — active champion model metadata
GET /api/v1/models/feature-importance  — global SHAP feature ranking
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_ml_service
from backend.app.schemas.predictions import FeatureImportanceItem, ModelInfo
from backend.app.services.ml.service import DelayPredictionService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/current", response_model=ModelInfo)
def get_current_model(
    svc: DelayPredictionService = Depends(get_ml_service),
) -> ModelInfo:
    info = svc.model_info
    return ModelInfo(
        binary_run_id=info.binary_run_id,
        regression_run_id=info.regression_run_id,
        root_cause_run_id=info.root_cause_run_id,
        feature_count=info.feature_count,
        loaded_at=info.loaded_at.isoformat(),
    )


@router.get("/feature-importance", response_model=List[FeatureImportanceItem])
def get_feature_importance(
    svc: DelayPredictionService = Depends(get_ml_service),
) -> List[FeatureImportanceItem]:
    return svc.global_importance()
