"""
backend/app/api/routes/orders.py
===================================
Production order CRUD endpoints.

POST   /api/v1/orders/                  — create order (returns 201)
GET    /api/v1/orders/today             — today's orders sorted by planned_start
PATCH  /api/v1/orders/{id}/status       — status transition
POST   /api/v1/orders/{id}/evaluate     — assemble ML features from DB and score
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone
from math import log
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import cast, Date, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import get_db, get_ml_service
from backend.app.db.models.machine import Machine, MachineUtilizationLog
from backend.app.db.models.operator import Operator
from backend.app.db.models.order import ProductionOrder
from backend.app.db.models.product import Product
from backend.app.schemas.orders import OrderCreate, OrderResponse, OrderStatusUpdate
from backend.app.schemas.predictions import DelayPrediction, OrderFeatures
from backend.app.services.ml.service import DelayPredictionService

# Encoding contracts must match training constants exactly.
_PRIORITY_ENCODING = {"low": 0, "normal": 1, "high": 2, "critical": 3}
_SKILL_TIER_ENCODING = {"junior": 0.0, "mid": 1.0, "senior": 2.0}
_SHIFT_ENCODING = {"morning": 0, "afternoon": 1, "night": 2}

router = APIRouter(prefix="/orders", tags=["orders"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order_in: OrderCreate,
    db: AsyncSession = Depends(get_db),
) -> OrderResponse:
    """Create a production order.  A delay prediction is fired as a side-effect
    by n8n listening on the /webhooks/order-released endpoint."""
    now = datetime.now(timezone.utc)
    order = ProductionOrder(
        **order_in.model_dump(),
        id=_uuid.uuid4(),
        status="pending",
        created_at=now,
        updated_at=now,
    )
    db.add(order)
    await db.flush()
    return OrderResponse.model_validate(order)


@router.get("/today", response_model=List[OrderResponse])
async def get_today_orders(
    db: AsyncSession = Depends(get_db),
) -> List[OrderResponse]:
    """Return all orders whose planned_start falls today (UTC), newest first."""
    today = datetime.now(timezone.utc).date()
    stmt = (
        select(ProductionOrder)
        .where(cast(ProductionOrder.planned_start, Date) == today)
        .order_by(ProductionOrder.planned_start)
    )
    result = await db.execute(stmt)
    orders = result.scalars().all()
    return [OrderResponse.model_validate(o) for o in orders]


@router.patch("/{order_id}/status", response_model=OrderResponse)
async def update_order_status(
    order_id: _uuid.UUID,
    update: OrderStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> OrderResponse:
    """Transition an order through its lifecycle (pending → in_progress → completed | delayed)."""
    result = await db.execute(
        select(ProductionOrder).where(ProductionOrder.id == order_id)
    )
    order = result.scalar_one_or_none()
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")

    order.status = update.status
    if update.notes is not None:
        order.notes = update.notes
    order.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return OrderResponse.model_validate(order)


# ---------------------------------------------------------------------------
# Evaluate endpoint — assembles ML features from DB and scores the order
# ---------------------------------------------------------------------------

@router.post("/{order_id}/evaluate", response_model=DelayPrediction)
async def evaluate_order(
    order_id: _uuid.UUID,
    db: AsyncSession = Depends(get_db),
    svc: DelayPredictionService = Depends(get_ml_service),
) -> DelayPrediction:
    """Load an order and its related records from postgres, assemble the full
    OrderFeatures vector, and return a DelayPrediction.

    This is the endpoint n8n calls — it requires only the order UUID.
    """
    # ── 1. Load order ────────────────────────────────────────────────────────
    order_row = (await db.execute(
        select(ProductionOrder).where(ProductionOrder.id == order_id)
    )).scalar_one_or_none()
    if order_row is None:
        raise HTTPException(status_code=404, detail="Order not found")

    # ── 2. Load related entities (optional FK — may be NULL) ─────────────────
    machine: Machine | None = None
    if order_row.machine_id:
        machine = (await db.execute(
            select(Machine).where(Machine.id == order_row.machine_id)
        )).scalar_one_or_none()

    operator: Operator | None = None
    if order_row.operator_id:
        operator = (await db.execute(
            select(Operator).where(Operator.id == order_row.operator_id)
        )).scalar_one_or_none()

    product: Product | None = None
    if order_row.product_id:
        product = (await db.execute(
            select(Product).where(Product.id == order_row.product_id)
        )).scalar_one_or_none()

    # ── 3. Machine utilization log — most recent snapshot ────────────────────
    util_log: MachineUtilizationLog | None = None
    if machine is not None:
        util_log = (await db.execute(
            select(MachineUtilizationLog)
            .where(MachineUtilizationLog.machine_id == machine.id)
            .order_by(MachineUtilizationLog.snapshot_at.desc())
            .limit(1)
        )).scalar_one_or_none()

    # ── 4. Concurrent orders on the same operator ─────────────────────────────
    concurrent_count: float = 0.0
    if operator is not None:
        result = await db.execute(
            select(func.count(ProductionOrder.id)).where(
                ProductionOrder.operator_id == operator.id,
                ProductionOrder.status.in_(["pending", "in_progress"]),
                ProductionOrder.id != order_row.id,
            )
        )
        concurrent_count = float(result.scalar() or 0)

    # ── 5. Derive temporal features from planned_start ────────────────────────
    ps = order_row.planned_start
    is_month_end = int(ps.day >= 28)
    is_quarter_end = int(ps.month in (3, 6, 9, 12) and ps.day >= 28)
    planned_start_hour = ps.hour
    planned_start_day_of_week = float(ps.weekday())  # Mon=0, Sun=6

    # hours_into_shift: approximate from planned_start hour
    # morning=0 (06-14h), afternoon=1 (14-22h), night=2 (22-06h)
    if 6 <= ps.hour < 14:
        shift_label = "morning"
        shift_start_hour = 6
    elif 14 <= ps.hour < 22:
        shift_label = "afternoon"
        shift_start_hour = 14
    else:
        shift_label = "night"
        shift_start_hour = 22
    hours_into_shift = float((ps.hour - shift_start_hour) % 24) + ps.minute / 60.0

    # ── 6. Encode categorical features (must match training constants) ─────────
    priority_encoded = _PRIORITY_ENCODING.get(order_row.priority, 1)

    skill_tier_encoded: float = _SKILL_TIER_ENCODING.get(
        operator.skill_tier if isinstance(getattr(operator, "skill_tier", None), str)
        else str(operator.skill_tier) if operator else "mid",
        1.0,
    )
    # operator.skill_tier is stored as int (0/1/2) — use directly if numeric
    if operator is not None and isinstance(operator.skill_tier, int):
        skill_tier_encoded = float(operator.skill_tier)

    operator_shift = "morning"
    if operator is not None:
        operator_shift = operator.shift_type
    shift_type_encoded = _SHIFT_ENCODING.get(operator_shift, 0)

    # ── 7. Schedule tightness ratio ───────────────────────────────────────────
    # Avoid division by zero; clamp to [0, 5]
    lead = max(order_row.planned_lead_time_hours, 1.0)
    schedule_tightness_ratio = min(order_row.estimated_total_hours / lead, 5.0)

    # ── 8. Days since last planned maintenance ────────────────────────────────
    # No maintenance table — use OEE proxy: low OEE → more days overdue
    # Fall back to a neutral 30 days when machine data is absent.
    if machine is not None and util_log is not None:
        oee = 1.0 - util_log.utilization_pct  # rough proxy; real OEE needs more data
        oee_30d = max(0.0, min(1.0, oee))
        days_since_pm = max(1.0, (1.0 - oee_30d) * 60.0)
    else:
        oee_30d = machine.oee_target if machine is not None else 0.80
        days_since_pm = 30.0

    # ── 9. Assemble OrderFeatures ─────────────────────────────────────────────
    features = OrderFeatures(
        # From orders table
        planned_lead_time_hours=order_row.planned_lead_time_hours,
        release_lag_hours=order_row.release_lag_hours,
        schedule_revision_count=float(order_row.schedule_revision_count),
        is_expedited=int(order_row.is_expedited),
        priority_encoded=priority_encoded,
        quantity=order_row.quantity,
        estimated_total_hours=order_row.estimated_total_hours,
        material_availability_at_release=int(order_row.material_availability_at_release),
        component_shortage_count=float(order_row.component_shortage_count),
        changeover_required=int(order_row.changeover_required),
        changeover_complexity_score=order_row.changeover_complexity_score,
        # From products table (cold-start defaults when product_id is NULL)
        product_complexity_score=product.complexity_score if product else 0.5,
        material_bom_complexity=product.material_bom_complexity if product else 1,
        operation_count=product.operation_count if product else 1,
        # From operators table (cold-start defaults when operator_id is NULL)
        operator_experience_months=operator.experience_months if operator else 12,
        operator_skill_tier_encoded=skill_tier_encoded,
        operator_concurrent_order_count=concurrent_count,
        shift_type_encoded=shift_type_encoded,
        hours_into_shift_at_start=hours_into_shift,
        # From machine / utilization logs (cold-start defaults when machine_id is NULL)
        machine_utilization_at_release=util_log.utilization_pct if util_log else 0.75,
        work_center_queue_depth_at_release=float(util_log.queue_depth) if util_log else 3.0,
        machine_oee_30d=oee_30d,
        machine_unplanned_downtime_hours_30d=util_log.unplanned_downtime_hours if util_log else 0.0,
        days_since_last_planned_maintenance=days_since_pm,
        maintenance_due_within_order_window=int(days_since_pm > 30),
        # Derived from order fields
        schedule_tightness_ratio=schedule_tightness_ratio,
        # Derived from planned_start datetime
        is_month_end=is_month_end,
        is_quarter_end=is_quarter_end,
        planned_start_hour=planned_start_hour,
        planned_start_day_of_week=planned_start_day_of_week,
        # Rolling historical features — None triggers cold-start defaults in pipeline
        product_delay_rate_90d=None,
        machine_delay_rate_90d=None,
        operator_delay_rate_90d=None,
        product_x_machine_delay_rate_90d=None,
        product_first_pass_yield_90d=None,
        machine_setup_overrun_rate_90d=None,
        shift_delay_rate_30d=None,
        machine_avg_delay_minutes_90d=None,
    )

    return svc.predict(features)
