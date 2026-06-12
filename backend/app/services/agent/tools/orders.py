"""
backend/app/services/agent/tools/orders.py
============================================
Agent tools for querying production_orders, optionally joined with the
latest delay_prediction.

Tools
-----
get_production_order    — single order by order_number
get_active_orders       — orders with status pending or in_progress
get_orders_at_risk      — active orders whose latest prediction >= threshold
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.order import ProductionOrder
from backend.app.db.models.prediction import DelayPrediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schema declarations (consumed by ToolRegistry.register)
# ---------------------------------------------------------------------------

GET_PRODUCTION_ORDER_SCHEMA: dict[str, Any] = {
    "description": "Get full details for a specific production order by its order number.",
    "parameters": {
        "order_id": "str — order number, e.g. 'ORD-20260601-001'",
    },
}

GET_ACTIVE_ORDERS_SCHEMA: dict[str, Any] = {
    "description": (
        "List production orders currently pending or in-progress, sorted by planned_start."
    ),
    "parameters": {
        "limit": "int (optional, default 20) — maximum number of orders to return",
    },
}

GET_ORDERS_AT_RISK_SCHEMA: dict[str, Any] = {
    "description": (
        "List active orders whose latest delay-probability prediction meets or exceeds "
        "the threshold, sorted by risk descending."
    ),
    "parameters": {
        "threshold": "float (optional, default 0.65) — minimum delay probability",
        "limit": "int (optional, default 10) — maximum rows to return",
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _order_to_dict(order: ProductionOrder) -> dict[str, Any]:
    return {
        "order_number": order.order_number,
        "status": order.status,
        "priority": order.priority,
        "is_expedited": order.is_expedited,
        "quantity": order.quantity,
        "planned_start": order.planned_start.isoformat() if order.planned_start else None,
        "planned_end": order.planned_end.isoformat() if order.planned_end else None,
        "actual_start": order.actual_start.isoformat() if order.actual_start else None,
        "actual_end": order.actual_end.isoformat() if order.actual_end else None,
        "estimated_total_hours": order.estimated_total_hours,
        "planned_lead_time_hours": order.planned_lead_time_hours,
        "material_availability": order.material_availability_at_release,
        "component_shortage_count": order.component_shortage_count,
        "changeover_required": order.changeover_required,
        "schedule_revision_count": order.schedule_revision_count,
    }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def get_production_order(db: AsyncSession, order_id: str) -> dict[str, Any]:
    """Return a single order dict or an ``{"error": ...}`` dict if not found."""
    stmt = select(ProductionOrder).where(ProductionOrder.order_number == order_id)
    result = await db.execute(stmt)
    order = result.scalar_one_or_none()
    if order is None:
        return {"error": f"Order '{order_id}' not found."}
    return _order_to_dict(order)


async def get_active_orders(
    db: AsyncSession,
    limit: int = 20,
) -> dict[str, Any]:
    """Return orders with status ``pending`` or ``in_progress``."""
    stmt = (
        select(ProductionOrder)
        .where(ProductionOrder.status.in_(["pending", "in_progress"]))
        .order_by(ProductionOrder.planned_start)
        .limit(limit)
    )
    result = await db.execute(stmt)
    orders = result.scalars().all()
    return {
        "orders": [_order_to_dict(o) for o in orders],
        "count": len(orders),
    }


async def get_orders_at_risk(
    db: AsyncSession,
    threshold: float = 0.65,
    limit: int = 10,
) -> dict[str, Any]:
    """Return active orders joined with their latest prediction where prob >= threshold."""
    # Subquery: most-recent prediction timestamp per order
    latest_subq = (
        select(
            DelayPrediction.production_order_id,
            func.max(DelayPrediction.created_at).label("max_ts"),
        )
        .group_by(DelayPrediction.production_order_id)
        .subquery()
    )

    stmt = (
        select(ProductionOrder, DelayPrediction)
        .join(latest_subq, latest_subq.c.production_order_id == ProductionOrder.id)
        .join(
            DelayPrediction,
            and_(
                DelayPrediction.production_order_id == ProductionOrder.id,
                DelayPrediction.created_at == latest_subq.c.max_ts,
            ),
        )
        .where(ProductionOrder.status.in_(["pending", "in_progress"]))
        .where(DelayPrediction.delay_probability >= threshold)
        .order_by(desc(DelayPrediction.delay_probability))
        .limit(limit)
    )

    result = await db.execute(stmt)
    rows = result.all()

    at_risk = []
    for order, pred in rows:
        d = _order_to_dict(order)
        d.update(
            {
                "delay_probability": round(pred.delay_probability, 3),
                "root_cause": pred.root_cause,
                "confidence": pred.confidence,
                "narrative": pred.narrative,
            }
        )
        at_risk.append(d)

    return {
        "at_risk_orders": at_risk,
        "count": len(at_risk),
        "threshold": threshold,
    }
