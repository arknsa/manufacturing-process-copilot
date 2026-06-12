"""
backend/app/api/routes/workflows.py
=====================================
n8n webhook endpoints.

POST /api/v1/webhooks/order-released  — called by n8n after it scores a new order
                                        via POST /api/v1/predictions/delay.
                                        Sends a Slack alert when delay_probability
                                        meets or exceeds HIGH_RISK_THRESHOLD and
                                        writes a WorkflowExecution audit row.

POST /api/v1/webhooks/shift-end       — shift-end digest trigger (cron 05:55/13:55/21:55)
POST /api/v1/webhooks/feedback-loop   — nightly outcome feedback (cron 23:00)

All endpoints return 202 Accepted immediately.  Notification and DB writes happen
inside an async background task so the caller (n8n) never waits on them.

n8n execution flow for Workflow 1 — High-Risk Order Alert:
  n8n receives new-order event
    → POST /api/v1/predictions/delay  { OrderFeatures }
    → IF delay_probability >= 0.70
        → POST /api/v1/webhooks/order-released  { data: prediction + order meta }
            backend background task:
              1. threshold check
              2. POST to SLACK_WEBHOOK_URL (if configured)
              3. write WorkflowExecution row
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["workflows"])


# ---------------------------------------------------------------------------
# Shared request schema
# ---------------------------------------------------------------------------

class WebhookPayload(BaseModel):
    data: Optional[Dict[str, Any]] = None


class ExecutionSummary(BaseModel):
    id: uuid.UUID
    workflow_name: str
    trigger_type: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------

async def _write_execution(
    workflow_name: str,
    trigger_type: str,
    status: str,
    input_data: dict | None,
    output_data: dict | None = None,
    error_message: str | None = None,
) -> None:
    """Persist a WorkflowExecution audit row.  Never raises — DB errors are logged."""
    try:
        from backend.app.db.models.workflow_execution import WorkflowExecution
        from backend.app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            record = WorkflowExecution(
                workflow_name=workflow_name,
                trigger_type=trigger_type,
                status=status,
                input_data=input_data,
                output_data=output_data,
                error_message=error_message,
            )
            db.add(record)
            await db.commit()
    except Exception as exc:
        logger.error("WorkflowExecution write failed for %s: %s", workflow_name, exc)


async def _send_slack(webhook_url: str, text: str) -> bool:
    """POST a plain-text message to a Slack incoming webhook.  Returns True on success."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json={"text": text})
            resp.raise_for_status()
            return True
    except Exception as exc:
        logger.warning("Slack notification failed: %s", exc)
        return False


def _build_slack_message(data: dict) -> str:
    """Format the high-risk order alert text for Slack."""
    order_number = data.get("order_number", "unknown")
    priority = data.get("priority", "normal").upper()
    prob = data.get("delay_probability", 0.0)
    root_cause = data.get("root_cause", "unknown")
    narrative = data.get("narrative", "")

    top_factors: list[dict] = data.get("top_risk_factors", [])
    factor_lines = "\n".join(
        f"  • {f.get('human_label', f.get('feature_name', '?'))}: "
        f"+{f.get('shap_contribution', 0.0):.2f} SHAP"
        for f in top_factors[:3]
    )

    return (
        f":warning: *High-Risk Order Alert*\n"
        f"*Order:* `{order_number}` | *Priority:* {priority}\n"
        f"*Delay probability:* {prob:.0%}\n"
        f"*Root cause:* {root_cause}\n"
        f"*Top risk factors:*\n{factor_lines}\n"
        f"*Summary:* {narrative[:300]}"
    )


async def _process_order_released(data: dict | None) -> None:
    """Core logic for the order-released webhook — runs in a background task."""
    settings = get_settings()
    input_snapshot = data or {}

    if not data:
        await _write_execution(
            "order_released_alert", "webhook", "failed",
            input_snapshot, error_message="Empty payload — no data field provided",
        )
        return

    delay_probability: float = float(data.get("delay_probability", 0.0))
    alert_sent = False
    error_msg: str | None = None

    if delay_probability >= settings.HIGH_RISK_THRESHOLD:
        logger.info(
            "[WORKFLOW] order-released: delay_probability=%.2f >= threshold=%.2f — sending alert",
            delay_probability, settings.HIGH_RISK_THRESHOLD,
        )
        if settings.SLACK_WEBHOOK_URL:
            message = _build_slack_message(data)
            alert_sent = await _send_slack(settings.SLACK_WEBHOOK_URL, message)
            if not alert_sent:
                error_msg = "Slack delivery failed"
        else:
            logger.info("[WORKFLOW] SLACK_WEBHOOK_URL not configured — alert suppressed")
    else:
        logger.info(
            "[WORKFLOW] order-released: delay_probability=%.2f below threshold — no alert",
            delay_probability,
        )

    output = {
        "delay_probability": delay_probability,
        "threshold": settings.HIGH_RISK_THRESHOLD,
        "alert_triggered": delay_probability >= settings.HIGH_RISK_THRESHOLD,
        "alert_sent": alert_sent,
    }
    status = "failed" if error_msg else "success"
    await _write_execution(
        "order_released_alert", "webhook", status,
        input_snapshot, output_data=output, error_message=error_msg,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/order-released", status_code=202)
async def order_released(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
) -> dict:
    """Called by n8n after it has scored a new order.

    n8n passes the prediction result in ``payload.data``:
      - order_number       (str)
      - priority           (str)
      - delay_probability  (float, 0–1)
      - root_cause         (str)
      - confidence         (str)
      - narrative          (str)
      - top_risk_factors   (list[dict])

    The backend checks the threshold, fires a Slack alert if warranted,
    and writes a WorkflowExecution audit row — all in the background.
    """
    data = payload.data or {}
    delay_probability = float(data.get("delay_probability", 0.0))
    settings = get_settings()
    alert_triggered = delay_probability >= settings.HIGH_RISK_THRESHOLD

    background_tasks.add_task(_process_order_released, payload.data)

    return {
        "status": "accepted",
        "workflow": "order-released",
        "alert_triggered": alert_triggered,
        "delay_probability": delay_probability,
    }


@router.post("/shift-end", status_code=202)
async def shift_end(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
) -> dict:
    """Called by n8n at shift end — generates the shift summary report."""
    background_tasks.add_task(
        _write_execution,
        "daily_digest", "scheduled", "success", payload.data,
    )
    return {"status": "accepted", "workflow": "shift-end"}


@router.post("/feedback-loop", status_code=202)
async def feedback_loop(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
) -> dict:
    """Called by n8n nightly — records model accuracy feedback."""
    background_tasks.add_task(
        _write_execution,
        "outcome_feedback_loop", "scheduled", "success", payload.data,
    )
    return {"status": "accepted", "workflow": "feedback-loop"}


@router.get("/executions", response_model=list[ExecutionSummary])
async def list_executions(
    limit: int = Query(50, ge=1, le=500),
) -> list[ExecutionSummary]:
    """Return recent workflow execution records, newest first."""
    from backend.app.db.models.workflow_execution import WorkflowExecution
    from backend.app.db.session import AsyncSessionLocal
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(WorkflowExecution)
            .order_by(WorkflowExecution.started_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
    return [
        ExecutionSummary(
            id=r.id,
            workflow_name=r.workflow_name,
            trigger_type=r.trigger_type,
            status=r.status,
            created_at=r.started_at,
        )
        for r in rows
    ]
