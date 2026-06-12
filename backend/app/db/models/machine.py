"""backend/app/db/models/machine.py — machines + machine_utilization_logs tables."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db._declarative import Base


class Machine(Base):
    __tablename__ = "machines"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    machine_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    machine_type: Mapped[str] = mapped_column(String(100), nullable=False)
    work_center: Mapped[str] = mapped_column(String(100), nullable=False)
    oee_target: Mapped[float] = mapped_column(Float, nullable=False, default=0.85)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    utilization_logs: Mapped[list["MachineUtilizationLog"]] = relationship(
        back_populates="machine", cascade="all, delete-orphan"
    )


class MachineUtilizationLog(Base):
    __tablename__ = "machine_utilization_logs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    machine_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    utilization_pct: Mapped[float] = mapped_column(Float, nullable=False)
    queue_depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unplanned_downtime_hours: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    machine: Mapped["Machine"] = relationship(back_populates="utilization_logs")
