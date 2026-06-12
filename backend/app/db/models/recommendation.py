"""backend/app/db/models/recommendation.py — recommendations table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db._declarative import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # category: schedule_change | resource_reallocation | maintenance | escalation | other
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    # urgency: low | medium | high | critical
    urgency: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    order_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("production_orders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    bottleneck_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("bottleneck_detections.id", ondelete="SET NULL"), nullable=True
    )
    # status: open | acknowledged | actioned | dismissed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open", index=True)
    actioned_by: Mapped[str | None] = mapped_column(String(150), nullable=True)
    actioned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
