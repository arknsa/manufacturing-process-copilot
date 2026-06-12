"""backend/app/db/models/product.py — products table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db._declarative import Base


class Product(Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    sku: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    product_family: Mapped[str] = mapped_column(String(100), nullable=False)
    complexity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    operation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    standard_hours: Mapped[float] = mapped_column(Float, nullable=False, default=8.0)
    material_bom_complexity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
