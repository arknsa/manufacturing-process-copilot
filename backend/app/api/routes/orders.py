"""
backend/app/api/routes/orders.py
===================================
Production order CRUD endpoints.

POST   /api/v1/orders/            — create order (returns 201)
GET    /api/v1/orders/today       — today's orders sorted by planned_start
PATCH  /api/v1/orders/{id}/status — status transition
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import cast, Date, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import get_db
from backend.app.db.models.order import ProductionOrder
from backend.app.schemas.orders import OrderCreate, OrderResponse, OrderStatusUpdate

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
