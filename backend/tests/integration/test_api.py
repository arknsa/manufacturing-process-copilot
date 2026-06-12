"""
backend/tests/integration/test_api.py
========================================
Integration tests for all API endpoints using FastAPI's TestClient.

ML service is replaced by a MagicMock (test_client fixture) so no MLflow
models are loaded for those tests.

DB session and CopilotAgent are replaced by mocks (db_test_client fixture)
so no PostgreSQL connection is needed for the new Phase 7 route tests.

Covers:
  GET  /health
  GET  /ready
  POST /api/v1/predictions/delay
  POST /api/v1/predictions/delay/batch
  GET  /api/v1/models/current
  GET  /api/v1/models/feature-importance
  POST /api/v1/orders/
  GET  /api/v1/orders/today
  PATCH /api/v1/orders/{id}/status
  POST /api/v1/chat/message  (stream=True and stream=False)
  GET  /api/v1/chat/sessions/{token}
  DELETE /api/v1/chat/sessions/{token}
  POST /api/v1/webhooks/order-released
  POST /api/v1/webhooks/shift-end
  POST /api/v1/webhooks/feedback-loop
"""

from __future__ import annotations

import uuid

import pytest

from tests.conftest import MOCK_PREDICTION, SAMPLE_ORDER

# ---------------------------------------------------------------------------
# Helper — minimal valid OrderCreate payload
# ---------------------------------------------------------------------------

SAMPLE_ORDER_CREATE = {
    "order_number": "ORD-TEST-001",
    "planned_start": "2026-06-12T08:00:00Z",
    "planned_end": "2026-06-12T16:00:00Z",
    "quantity": 50,
    "estimated_total_hours": 8.0,
    "planned_lead_time_hours": 48.0,
}

# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------


def test_health(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ready(test_client):
    resp = test_client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ml_service_loaded"] is True
    assert body["status"] == "ready"


# ---------------------------------------------------------------------------
# Predictions — single order
# ---------------------------------------------------------------------------


def test_predict_delay_200(test_client):
    resp = test_client.post("/api/v1/predictions/delay", json=SAMPLE_ORDER)
    assert resp.status_code == 200


def test_predict_delay_response_schema(test_client):
    resp = test_client.post("/api/v1/predictions/delay", json=SAMPLE_ORDER)
    body = resp.json()
    assert "delay_probability" in body
    assert "confidence" in body
    assert "root_cause" in body
    assert "narrative" in body
    assert "top_risk_factors" in body
    assert "mitigating_factors" in body


def test_predict_delay_probability_range(test_client):
    resp = test_client.post("/api/v1/predictions/delay", json=SAMPLE_ORDER)
    body = resp.json()
    assert 0.0 <= body["delay_probability"] <= 1.0


def test_predict_delay_narrative_nonempty(test_client):
    resp = test_client.post("/api/v1/predictions/delay", json=SAMPLE_ORDER)
    body = resp.json()
    assert len(body["narrative"]) > 50


def test_predict_delay_confidence_values(test_client):
    resp = test_client.post("/api/v1/predictions/delay", json=SAMPLE_ORDER)
    body = resp.json()
    assert body["confidence"] in ("high", "medium", "low")


def test_predict_delay_cold_start_order(test_client):
    """Cold-start order (None rolling features) should return 200."""
    cold_order = {**SAMPLE_ORDER}
    for f in [
        "product_delay_rate_90d", "machine_delay_rate_90d",
        "operator_delay_rate_90d", "product_x_machine_delay_rate_90d",
        "product_first_pass_yield_90d", "machine_setup_overrun_rate_90d",
        "shift_delay_rate_30d", "machine_avg_delay_minutes_90d",
    ]:
        cold_order[f] = None
    resp = test_client.post("/api/v1/predictions/delay", json=cold_order)
    assert resp.status_code == 200


def test_predict_delay_missing_required_field(test_client):
    """Missing required field must return 422 Unprocessable Entity."""
    incomplete = {k: v for k, v in SAMPLE_ORDER.items() if k != "quantity"}
    resp = test_client.post("/api/v1/predictions/delay", json=incomplete)
    assert resp.status_code == 422


def test_predict_delay_mock_called(test_client, mock_ml_service):
    test_client.post("/api/v1/predictions/delay", json=SAMPLE_ORDER)
    mock_ml_service.predict.assert_called_once()


# ---------------------------------------------------------------------------
# Predictions — batch
# ---------------------------------------------------------------------------


def test_predict_batch_200(test_client):
    payload = {"orders": [SAMPLE_ORDER, SAMPLE_ORDER]}
    resp = test_client.post("/api/v1/predictions/delay/batch", json=payload)
    assert resp.status_code == 200


def test_predict_batch_count(test_client):
    payload = {"orders": [SAMPLE_ORDER, SAMPLE_ORDER]}
    resp = test_client.post("/api/v1/predictions/delay/batch", json=payload)
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_over_limit(test_client):
    """Batch > 100 orders must return 422."""
    payload = {"orders": [SAMPLE_ORDER] * 101}
    resp = test_client.post("/api/v1/predictions/delay/batch", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


def test_models_current_200(test_client):
    resp = test_client.get("/api/v1/models/current")
    assert resp.status_code == 200


def test_models_current_schema(test_client):
    resp = test_client.get("/api/v1/models/current")
    body = resp.json()
    assert "binary_run_id" in body
    assert "feature_count" in body
    assert "loaded_at" in body


def test_models_feature_importance_200(test_client):
    resp = test_client.get("/api/v1/models/feature-importance")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def test_create_order_201(db_test_client):
    resp = db_test_client.post("/api/v1/orders/", json=SAMPLE_ORDER_CREATE)
    assert resp.status_code == 201
    body = resp.json()
    assert body["order_number"] == "ORD-TEST-001"
    assert body["status"] == "pending"
    assert "id" in body
    assert "created_at" in body


def test_create_order_missing_field_422(db_test_client):
    """quantity is required — omitting it should fail Pydantic validation."""
    incomplete = {k: v for k, v in SAMPLE_ORDER_CREATE.items() if k != "quantity"}
    resp = db_test_client.post("/api/v1/orders/", json=incomplete)
    assert resp.status_code == 422


def test_get_today_orders_200(db_test_client):
    resp = db_test_client.get("/api/v1/orders/today")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_update_order_status_404(db_test_client):
    """Patching a non-existent order ID must return 404."""
    resp = db_test_client.patch(
        f"/api/v1/orders/{uuid.uuid4()}/status",
        json={"status": "in_progress"},
    )
    assert resp.status_code == 404


def test_update_order_status_invalid_value_422(db_test_client):
    """Status values outside the allowed set must return 422."""
    resp = db_test_client.patch(
        f"/api/v1/orders/{uuid.uuid4()}/status",
        json={"status": "exploded"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Chat — non-streaming
# ---------------------------------------------------------------------------


def test_chat_message_non_streaming_200(db_test_client):
    resp = db_test_client.post(
        "/api/v1/chat/message",
        json={
            "message": "What orders are at risk today?",
            "session_token": "test-session-001",
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "content" in body
    assert "session_token" in body
    assert body["role"] == "assistant"


def test_chat_message_non_streaming_content(db_test_client):
    resp = db_test_client.post(
        "/api/v1/chat/message",
        json={
            "message": "How many high-risk orders?",
            "session_token": "test-session-002",
            "stream": False,
        },
    )
    body = resp.json()
    assert len(body["content"]) > 0


# ---------------------------------------------------------------------------
# Chat — streaming (SSE)
# ---------------------------------------------------------------------------


def test_chat_message_streaming_200(db_test_client):
    resp = db_test_client.post(
        "/api/v1/chat/message",
        json={
            "message": "What orders are at risk today?",
            "session_token": "test-session-003",
            "stream": True,
        },
    )
    assert resp.status_code == 200


def test_chat_message_streaming_content_type(db_test_client):
    resp = db_test_client.post(
        "/api/v1/chat/message",
        json={
            "message": "Summarise today's orders.",
            "session_token": "test-session-004",
            "stream": True,
        },
    )
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_chat_message_streaming_done_sentinel(db_test_client):
    """SSE response must end with the [DONE] sentinel."""
    resp = db_test_client.post(
        "/api/v1/chat/message",
        json={
            "message": "Any bottlenecks on Machine-42?",
            "session_token": "test-session-005",
            "stream": True,
        },
    )
    assert b"[DONE]" in resp.content


# ---------------------------------------------------------------------------
# Chat — session history + delete
# ---------------------------------------------------------------------------


def test_get_session_404(db_test_client):
    resp = db_test_client.get("/api/v1/chat/sessions/nonexistent-token")
    assert resp.status_code == 404


def test_delete_session_404(db_test_client):
    resp = db_test_client.delete("/api/v1/chat/sessions/nonexistent-token")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Workflows (n8n webhooks)
# ---------------------------------------------------------------------------


def test_webhook_order_released_202(db_test_client):
    resp = db_test_client.post(
        "/api/v1/webhooks/order-released",
        json={"data": {"order_number": "ORD-001", "risk": "high"}},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["workflow"] == "order-released"


def test_webhook_shift_end_202(db_test_client):
    resp = db_test_client.post("/api/v1/webhooks/shift-end", json={})
    assert resp.status_code == 202
    assert resp.json()["workflow"] == "shift-end"


def test_webhook_feedback_loop_202(db_test_client):
    resp = db_test_client.post("/api/v1/webhooks/feedback-loop", json={})
    assert resp.status_code == 202
    assert resp.json()["workflow"] == "feedback-loop"


def test_webhook_empty_payload_202(db_test_client):
    """Webhooks must accept an empty body (no required fields)."""
    resp = db_test_client.post("/api/v1/webhooks/order-released", json={})
    assert resp.status_code == 202
