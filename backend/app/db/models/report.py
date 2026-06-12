"""backend/app/db/models/report.py — operational_reports table."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, String, Text, func
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db._declarative import Base


class OperationalReport(Base):
    __tablename__ = "operational_reports"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # report_type: shift_summary | handover_brief | daily_digest
    report_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    report_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    shift: Mapped[str | None] = mapped_column(String(20), nullable=True)
    data_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    html_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
