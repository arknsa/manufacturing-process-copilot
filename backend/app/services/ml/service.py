"""
backend/app/services/ml/service.py
=====================================
Orchestrates MLflowModelRegistry + DelayExplainer to serve predictions.

Public API
----------
DelayPredictionService
    predict(order)         → DelayPrediction
    predict_batch(orders)  → List[DelayPrediction]
    global_importance()    → List[FeatureImportanceItem]  (cached after first call)
    model_info             → ModelInfo dataclass
"""

from __future__ import annotations

import logging
from typing import List, Optional

from backend.app.core.config import Settings
from backend.app.schemas.predictions import (
    BatchPredictionResponse,
    DelayPrediction,
    FactorExplanationResponse,
    FeatureImportanceItem,
    OrderFeatures,
)
from backend.app.services.ml.explainability import (
    ExplanationResult,
    FactorExplanation,
    _human_label,
)
from backend.app.services.ml.registry import MLflowModelRegistry, ModelInfo

logger = logging.getLogger(__name__)


def _to_factor_response(f: FactorExplanation) -> FactorExplanationResponse:
    return FactorExplanationResponse(
        feature_name=f.feature_name,
        human_label=f.human_label,
        value=f.value,
        shap_contribution=f.shap_contribution,
        direction=f.direction,
        magnitude=f.magnitude,
    )


def _to_prediction_schema(result: ExplanationResult) -> DelayPrediction:
    return DelayPrediction(
        delay_probability=result.predicted_delay_probability,
        delay_minutes_estimate=result.predicted_delay_minutes,
        root_cause=result.predicted_root_cause,
        confidence=result.confidence,
        top_risk_factors=[_to_factor_response(f) for f in result.top_risk_factors],
        mitigating_factors=[_to_factor_response(f) for f in result.mitigating_factors],
        narrative=result.narrative,
    )


class DelayPredictionService:
    """Single inference service for all ML prediction tasks."""

    def __init__(self, settings: Settings) -> None:
        self._registry = MLflowModelRegistry(
            tracking_uri=settings.MLFLOW_TRACKING_URI,
            binary_run_id=settings.BINARY_CHAMPION_RUN_ID,
            regression_run_id=settings.REGR_CHAMPION_RUN_ID,
            root_cause_run_id=settings.RC_CHAMPION_RUN_ID,
        )
        self._registry.load()
        self._importance_cache: Optional[List[FeatureImportanceItem]] = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, order: OrderFeatures) -> DelayPrediction:
        result = self._registry.explainer.explain_order(order.to_feature_dict())
        logger.debug(
            "predict: prob=%.3f confidence=%s root_cause=%s",
            result.predicted_delay_probability,
            result.confidence,
            result.predicted_root_cause,
        )
        return _to_prediction_schema(result)

    def predict_batch(self, orders: List[OrderFeatures]) -> List[DelayPrediction]:
        return [self.predict(o) for o in orders]

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    def global_importance(self) -> List[FeatureImportanceItem]:
        if self._importance_cache is None:
            logger.info("Computing global SHAP importance (first call, cached after)...")
            df = self._registry.explainer.global_importance()
            self._importance_cache = [
                FeatureImportanceItem(
                    rank=int(row["rank"]),
                    feature_name=row["feature_name"],
                    human_label=_human_label(row["feature_name"]),
                    mean_abs_shap=float(row["mean_abs_shap"]),
                )
                for _, row in df.iterrows()
            ]
        return self._importance_cache

    @property
    def model_info(self) -> ModelInfo:
        return self._registry.model_info
