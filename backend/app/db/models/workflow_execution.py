"""backend/app/db/models/workflow_execution.py — workflow_executions table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db._declarative import Base


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workflow_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # trigger_type: webhook | scheduled | manual
    trigger_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # status: running | success | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running", index=True)
    input_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
