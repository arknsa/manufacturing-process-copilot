"""
backend/tests/conftest.py
===========================
Shared pytest fixtures.

SAMPLE_ORDER        — valid OrderFeatures dict for a realistic mid-risk order.
MOCK_PREDICTION     — canned DelayPrediction that the mock service returns.
mock_ml_service     — MagicMock standing in for DelayPredictionService.
mock_db             — AsyncMock standing in for AsyncSession.
mock_agent_instance — async-generator stub that yields a fixed agent response.
test_client         — FastAPI TestClient for ML/model/health routes (ML service mocked).
db_test_client      — FastAPI TestClient for DB-backed routes (DB + agent both mocked,
                      ML service + LLMClient patched so no real loading happens).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.schemas.predictions import DelayPrediction, FactorExplanationResponse

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_ORDER: dict = {
    "planned_lead_time_hours": 48.0,
    "release_lag_hours": 4.0,
    "schedule_revision_count": 0.0,
    "is_expedited": 0,
    "priority_encoded": 2,
    "quantity": 50,
    "operation_count": 3,
    "estimated_total_hours": 8.5,
    "schedule_tightness_ratio": 0.65,
    "product_complexity_score": 0.55,
    "material_bom_complexity": 4,
    "is_month_end": 0,
    "is_quarter_end": 0,
    "machine_utilization_at_release": 0.78,
    "work_center_queue_depth_at_release": 1.0,
    "machine_oee_30d": 0.65,
    "machine_unplanned_downtime_hours_30d": 2.0,
    "days_since_last_planned_maintenance": 18.0,
    "maintenance_due_within_order_window": 0,
    "changeover_required": 1,
    "changeover_complexity_score": 2.1,
    "operator_experience_months": 24,
    "operator_skill_tier_encoded": 1.0,
    "operator_concurrent_order_count": 0.0,
    "hours_into_shift_at_start": 3.0,
    "shift_type_encoded": 0,
    "material_availability_at_release": 1,
    "component_shortage_count": 0.0,
    "product_delay_rate_90d": 0.28,
    "machine_delay_rate_90d": 0.32,
    "operator_delay_rate_90d": 0.22,
    "product_x_machine_delay_rate_90d": 0.30,
    "product_first_pass_yield_90d": 0.88,
    "machine_setup_overrun_rate_90d": 0.42,
    "shift_delay_rate_30d": 0.36,
    "machine_avg_delay_minutes_90d": 82.0,
    "planned_start_day_of_week": 1.0,
    "planned_start_hour": 8,
}

MOCK_PREDICTION = DelayPrediction(
    delay_probability=0.72,
    delay_minutes_estimate=385.0,
    root_cause="material_unavailability",
    confidence="high",
    top_risk_factors=[
        FactorExplanationResponse(
            feature_name="schedule_tightness_ratio",
            human_label="Schedule tightness",
            value=0.65,
            shap_contribution=1.25,
            direction="increases_risk",
            magnitude="high",
        ),
        FactorExplanationResponse(
            feature_name="machine_utilization_at_release",
            human_label="Machine utilization",
            value=0.78,
            shap_contribution=0.89,
            direction="increases_risk",
            magnitude="medium",
        ),
    ],
    mitigating_factors=[
        FactorExplanationResponse(
            feature_name="material_availability_at_release",
            human_label="Material availability",
            value=1,
            shap_contribution=-0.47,
            direction="reduces_risk",
            magnitude="medium",
        ),
    ],
    narrative=(
        "This order has a high risk of delay (72% probability).\n\n"
        "The main risk factors are: Schedule tightness (high impact), "
        "Machine utilization (medium impact).\n\n"
        "Risk is partially offset by: material availability.\n\n"
        "If delayed, the predicted root cause is material unavailability "
        "with an estimated delay of 385 minutes."
    ),
)


# ---------------------------------------------------------------------------
# Fixtures — ML service (unchanged from Phase 4)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ml_service() -> MagicMock:
    svc = MagicMock()
    svc.predict.return_value = MOCK_PREDICTION
    svc.predict_batch.side_effect = lambda orders: [MOCK_PREDICTION] * len(orders)
    svc.global_importance.return_value = []
    svc.model_info.binary_run_id = "140ce9025def4436a397ef8333078202"
    svc.model_info.regression_run_id = "d10e7217af3b4b68920d895c244ca1aa"
    svc.model_info.root_cause_run_id = "7cc43338ae434163a2207e052354db1b"
    svc.model_info.feature_count = 41
    svc.model_info.loaded_at.isoformat.return_value = "2026-06-12T00:00:00+00:00"
    return svc


@pytest.fixture()
def test_client(mock_ml_service: MagicMock):
    """FastAPI TestClient with the ML service dependency overridden.

    Both DelayPredictionService and LLMClient are patched inside the TestClient
    context so the lifespan receives mocks instead of the real services
    (which would require shap + asyncpg).
    """
    from backend.app.api.dependencies import get_ml_service
    from backend.app.main import create_app

    mock_llm = MagicMock()
    mock_llm.aclose = AsyncMock()

    with (
        patch("backend.app.main.DelayPredictionService", return_value=mock_ml_service),
        patch("backend.app.main.LLMClient", return_value=mock_llm),
    ):
        app = create_app()
        app.dependency_overrides[get_ml_service] = lambda: mock_ml_service
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


# ---------------------------------------------------------------------------
# Fixtures — DB + Agent (Phase 7 additions)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db() -> AsyncMock:
    """Lightweight AsyncSession stub for routes that call get_db()."""
    session = AsyncMock()

    # Default: execute() returns a result whose scalars().all() is an empty list
    # and scalar_one_or_none() returns None (triggers 404s on lookup routes).
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar_one_or_none.return_value = None
    session.execute.return_value = mock_result

    return session


@pytest.fixture()
def mock_agent_instance() -> MagicMock:
    """CopilotAgent stub whose run() yields a single canned response."""
    agent = MagicMock()

    async def _run(message: str, session_token: str):
        yield "Order ORD-001 currently shows a 72% delay probability."

    agent.run = _run
    return agent


@pytest.fixture()
def db_test_client(mock_ml_service, mock_db, mock_agent_instance):
    """FastAPI TestClient for routes that require a DB session or the agent.

    Both DelayPredictionService and LLMClient are patched so their real
    constructors (MLflow load, httpx client) are never called — making this
    fixture fast regardless of the local environment.
    """
    from backend.app.api.dependencies import get_agent, get_db, get_ml_service
    from backend.app.main import create_app

    mock_llm = MagicMock()
    mock_llm.aclose = AsyncMock()

    async def _get_db_override():
        yield mock_db

    with (
        patch("backend.app.main.DelayPredictionService", return_value=mock_ml_service),
        patch("backend.app.main.LLMClient", return_value=mock_llm),
    ):
        app = create_app()

        app.dependency_overrides[get_ml_service] = lambda: mock_ml_service
        app.dependency_overrides[get_db] = _get_db_override
        app.dependency_overrides[get_agent] = lambda: mock_agent_instance

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
