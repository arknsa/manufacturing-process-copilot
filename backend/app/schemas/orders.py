"""
backend/app/schemas/orders.py
================================
Pydantic schemas for the production orders resource.

OrderCreate        — inbound payload when creating an order.
OrderResponse      — outbound single-order representation.
OrderWithPrediction — full view returned by GET /api/v1/orders/today.
OrderStatusUpdate  — PATCH body for status transitions.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.app.schemas.predictions import DelayPrediction


class OrderCreate(BaseModel):
    order_number: str = Field(..., max_length=50)
    product_id: Optional[uuid.UUID] = None
    machine_id: Optional[uuid.UUID] = None
    operator_id: Optional[uuid.UUID] = None
    planned_start: datetime
    planned_end: datetime
    quantity: int = Field(..., gt=0)
    is_expedited: bool = False
    priority: str = Field("normal", pattern="^(normal|high|critical)$")
    estimated_total_hours: float = Field(..., gt=0)
    planned_lead_time_hours: float = Field(..., gt=0)
    release_lag_hours: float = 0.0
    schedule_revision_count: int = 0
    material_availability_at_release: bool = True
    component_shortage_count: int = 0
    changeover_required: bool = False
    changeover_complexity_score: float = 0.0
    notes: Optional[str] = None


class OrderResponse(BaseModel):
    id: uuid.UUID
    order_number: str
    product_id: Optional[uuid.UUID]
    machine_id: Optional[uuid.UUID]
    operator_id: Optional[uuid.UUID]
    planned_start: datetime
    planned_end: datetime
    actual_start: Optional[datetime]
    actual_end: Optional[datetime]
    quantity: int
    is_expedited: bool
    priority: str
    estimated_total_hours: float
    planned_lead_time_hours: float
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OrderWithPrediction(OrderResponse):
    latest_prediction: Optional[DelayPrediction] = None


class OrderStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(pending|in_progress|completed|delayed)$")
    notes: Optional[str] = None
