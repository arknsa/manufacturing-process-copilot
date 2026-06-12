"""backend/app/db/models/operator.py — operators table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db._declarative import Base


class Operator(Base):
    __tablename__ = "operators"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    employee_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    skill_tier: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    experience_months: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    shift_type: Mapped[str] = mapped_column(String(20), nullable=False, default="day")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
