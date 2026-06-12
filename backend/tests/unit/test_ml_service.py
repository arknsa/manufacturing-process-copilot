"""
backend/tests/unit/test_ml_service.py
========================================
Unit tests for DelayPredictionService.

The MLflowModelRegistry is mocked so no MLflow server or model files
are required to run these tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.schemas.predictions import OrderFeatures
from backend.app.services.ml.explainability import ExplanationResult, FactorExplanation
from tests.conftest import SAMPLE_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_explanation_result(prob: float = 0.72) -> ExplanationResult:
    factor = FactorExplanation(
        feature_name="schedule_tightness_ratio",
        human_label="Schedule tightness",
        value=0.65,
        shap_contribution=1.25,
        direction="increases_risk",
        magnitude="high",
    )
    mit_factor = FactorExplanation(
        feature_name="material_availability_at_release",
        human_label="Material availability",
        value=1,
        shap_contribution=-0.47,
        direction="reduces_risk",
        magnitude="medium",
    )
    return ExplanationResult(
        predicted_delay_probability=prob,
        predicted_delay_minutes=385.0,
        predicted_root_cause="material_unavailability",
        confidence="high" if prob >= 0.65 else "low",
        top_risk_factors=[factor],
        mitigating_factors=[mit_factor],
        narrative=f"This order has a high risk of delay ({prob:.0%} probability).",
        shap_values=np.zeros(41),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("backend.app.services.ml.service.MLflowModelRegistry")
def test_predict_returns_delay_prediction(MockRegistry):
    mock_registry = MockRegistry.return_value
    mock_registry.explainer.explain_order.return_value = _make_explanation_result(0.72)

    from backend.app.core.config import Settings
    from backend.app.services.ml.service import DelayPredictionService

    settings = Settings()
    svc = DelayPredictionService.__new__(DelayPredictionService)
    svc._registry = mock_registry
    svc._importance_cache = None

    order = OrderFeatures(**SAMPLE_ORDER)
    result = svc.predict(order)

    assert result.delay_probability == pytest.approx(0.72)
    assert result.confidence == "high"
    assert result.root_cause == "material_unavailability"
    assert result.delay_minutes_estimate == pytest.approx(385.0)
    assert len(result.top_risk_factors) == 1
    assert result.top_risk_factors[0].feature_name == "schedule_tightness_ratio"
    assert len(result.narrative) > 50


@patch("backend.app.services.ml.service.MLflowModelRegistry")
def test_predict_batch_returns_list(MockRegistry):
    mock_registry = MockRegistry.return_value
    mock_registry.explainer.explain_order.return_value = _make_explanation_result(0.30)

    from backend.app.services.ml.service import DelayPredictionService

    svc = DelayPredictionService.__new__(DelayPredictionService)
    svc._registry = mock_registry
    svc._importance_cache = None

    orders = [OrderFeatures(**SAMPLE_ORDER)] * 3
    results = svc.predict_batch(orders)

    assert len(results) == 3
    assert all(r.delay_probability == pytest.approx(0.30) for r in results)


@patch("backend.app.services.ml.service.MLflowModelRegistry")
def test_cold_start_order_accepted(MockRegistry):
    """OrderFeatures with None rolling features should not raise."""
    mock_registry = MockRegistry.return_value
    mock_registry.explainer.explain_order.return_value = _make_explanation_result(0.45)

    from backend.app.services.ml.service import DelayPredictionService

    svc = DelayPredictionService.__new__(DelayPredictionService)
    svc._registry = mock_registry
    svc._importance_cache = None

    cold_order = {**SAMPLE_ORDER}
    for cold_feat in [
        "product_delay_rate_90d", "machine_delay_rate_90d",
        "operator_delay_rate_90d", "product_x_machine_delay_rate_90d",
        "product_first_pass_yield_90d", "machine_setup_overrun_rate_90d",
        "shift_delay_rate_30d", "machine_avg_delay_minutes_90d",
    ]:
        cold_order[cold_feat] = None

    order = OrderFeatures(**cold_order)
    result = svc.predict(order)
    assert 0.0 <= result.delay_probability <= 1.0

    # Confirm NaN was passed for cold-start features
    call_dict = mock_registry.explainer.explain_order.call_args[0][0]
    import math
    assert math.isnan(call_dict["product_delay_rate_90d"])


@patch("backend.app.services.ml.service.MLflowModelRegistry")
def test_global_importance_caches(MockRegistry):
    """global_importance() should call explainer only once, then return cached."""
    import pandas as pd

    mock_registry = MockRegistry.return_value
    mock_registry.explainer.global_importance.return_value = pd.DataFrame({
        "rank": [1, 2],
        "feature_name": ["schedule_tightness_ratio", "machine_utilization_at_release"],
        "mean_abs_shap": [1.12, 0.84],
    })

    from backend.app.services.ml.service import DelayPredictionService

    svc = DelayPredictionService.__new__(DelayPredictionService)
    svc._registry = mock_registry
    svc._importance_cache = None

    first = svc.global_importance()
    second = svc.global_importance()

    assert first is second  # same object — cache hit
    mock_registry.explainer.global_importance.assert_called_once()
    assert len(first) == 2
    assert first[0].rank == 1
