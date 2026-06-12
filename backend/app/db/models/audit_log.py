"""backend/app/db/models/audit_log.py — audit_logs table (append-only)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db._declarative import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # entity_type: order | prediction | recommendation | model | chat_session
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # operation: create | update | delete | predict | promote | acknowledge | action
    operation: Mapped[str] = mapped_column(String(30), nullable=False)
    actor: Mapped[str] = mapped_column(String(150), nullable=False, default="system")
    data_before: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    data_after: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
