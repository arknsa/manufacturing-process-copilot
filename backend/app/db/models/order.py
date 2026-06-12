"""backend/app/db/models/order.py — production_orders table (central operational table)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db._declarative import Base


class ProductionOrder(Base):
    __tablename__ = "production_orders"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    order_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)

    # Foreign keys to reference tables
    product_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("products.id", ondelete="SET NULL"), nullable=True, index=True
    )
    machine_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("machines.id", ondelete="SET NULL"), nullable=True, index=True
    )
    operator_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("operators.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Scheduling fields
    planned_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    planned_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actual_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Order attributes
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    is_expedited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False, default="normal")
    estimated_total_hours: Mapped[float] = mapped_column(Float, nullable=False)
    planned_lead_time_hours: Mapped[float] = mapped_column(Float, nullable=False)
    release_lag_hours: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    schedule_revision_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Material & changeover
    material_availability_at_release: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    component_shortage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    changeover_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    changeover_complexity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Status lifecycle: pending → in_progress → completed | delayed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    predictions: Mapped[list["DelayPrediction"]] = relationship(  # type: ignore[name-defined]
        "DelayPrediction", back_populates="order", cascade="all, delete-orphan"
    )
