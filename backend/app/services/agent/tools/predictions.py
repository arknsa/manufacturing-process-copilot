"""
backend/app/services/agent/tools/predictions.py
=================================================
Agent tools for retrieving stored delay predictions and SHAP explanations.

Tools
-----
get_delay_prediction    — latest prediction for a given order_number
get_risk_summary        — aggregate risk counts for active orders
get_feature_explanation — SHAP factor breakdown for a specific order
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import and_, case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.order import ProductionOrder
from backend.app.db.models.prediction import DelayPrediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

GET_DELAY_PREDICTION_SCHEMA: dict[str, Any] = {
    "description": (
        "Retrieve the most recent delay prediction for a production order, "
        "including probability, root cause, confidence, and narrative explanation."
    ),
    "parameters": {
        "order_id": "str — order number, e.g. 'ORD-20260601-001'",
    },
}

GET_RISK_SUMMARY_SCHEMA: dict[str, Any] = {
    "description": (
        "Return an aggregate risk summary for all active orders: "
        "counts of high/medium/low risk and average delay probability."
    ),
    "parameters": {},
}

GET_FEATURE_EXPLANATION_SCHEMA: dict[str, Any] = {
    "description": (
        "Return the SHAP-based feature explanation for a specific order's "
        "latest prediction — top risk factors and mitigating factors."
    ),
    "parameters": {
        "order_id": "str — order number",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _latest_prediction_for(
    db: AsyncSession, order_number: str
) -> tuple[ProductionOrder | None, DelayPrediction | None]:
    """Return (order, latest_prediction) or (None, None) if not found."""
    stmt = (
        select(ProductionOrder, DelayPrediction)
        .join(DelayPrediction, DelayPrediction.production_order_id == ProductionOrder.id)
        .where(ProductionOrder.order_number == order_number)
        .order_by(desc(DelayPrediction.created_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.first()
    if row is None:
        return None, None
    return row[0], row[1]


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def get_delay_prediction(
    db: AsyncSession,
    order_id: str,
) -> dict[str, Any]:
    """Latest prediction for order_id, or error if not found."""
    order, pred = await _latest_prediction_for(db, order_id)
    if pred is None:
        return {"error": f"No prediction found for order '{order_id}'."}

    return {
        "order_number": order.order_number,
        "status": order.status,
        "delay_probability": round(pred.delay_probability, 3),
        "delay_probability_pct": f"{pred.delay_probability:.0%}",
        "delay_minutes_estimate": pred.delay_minutes_estimate,
        "root_cause": pred.root_cause,
        "confidence": pred.confidence,
        "narrative": pred.narrative,
        "model_version": pred.model_version,
        "predicted_at": pred.created_at.isoformat() if pred.created_at else None,
    }


async def get_risk_summary(db: AsyncSession) -> dict[str, Any]:
    """Aggregate risk counts for all pending/in_progress orders."""
    # Subquery: latest prediction per order
    latest_subq = (
        select(
            DelayPrediction.production_order_id,
            func.max(DelayPrediction.created_at).label("max_ts"),
        )
        .group_by(DelayPrediction.production_order_id)
        .subquery()
    )

    stmt = (
        select(
            func.count().label("total"),
            func.sum(
                case((DelayPrediction.delay_probability >= 0.65, 1), else_=0)
            ).label("high_risk"),
            func.sum(
                case(
                    (
                        and_(
                            DelayPrediction.delay_probability >= 0.40,
                            DelayPrediction.delay_probability < 0.65,
                        ),
                        1,
                    ),
                    else_=0,
                )
            ).label("medium_risk"),
            func.sum(
                case((DelayPrediction.delay_probability < 0.40, 1), else_=0)
            ).label("low_risk"),
            func.avg(DelayPrediction.delay_probability).label("avg_probability"),
        )
        .join(latest_subq, latest_subq.c.production_order_id == DelayPrediction.production_order_id)
        .join(
            ProductionOrder,
            and_(
                ProductionOrder.id == DelayPrediction.production_order_id,
                DelayPrediction.created_at == latest_subq.c.max_ts,
            ),
        )
        .where(ProductionOrder.status.in_(["pending", "in_progress"]))
    )

    result = await db.execute(stmt)
    row = result.one()

    total = row.total or 0
    avg_prob = float(row.avg_probability) if row.avg_probability else 0.0

    return {
        "active_orders_with_prediction": total,
        "high_risk": int(row.high_risk or 0),
        "medium_risk": int(row.medium_risk or 0),
        "low_risk": int(row.low_risk or 0),
        "average_delay_probability": round(avg_prob, 3),
        "risk_threshold": 0.65,
    }


async def get_feature_explanation(
    db: AsyncSession,
    order_id: str,
) -> dict[str, Any]:
    """SHAP factor breakdown for the latest prediction of order_id."""
    order, pred = await _latest_prediction_for(db, order_id)
    if pred is None:
        return {"error": f"No prediction found for order '{order_id}'."}

    return {
        "order_number": order.order_number,
        "delay_probability": round(pred.delay_probability, 3),
        "top_risk_factors": pred.top_risk_factors or [],
        "mitigating_factors": pred.mitigating_factors or [],
        "narrative": pred.narrative,
        "root_cause": pred.root_cause,
    }
