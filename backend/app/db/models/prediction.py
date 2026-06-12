"""
backend/app/db/models/prediction.py
— delay_predictions, ml_model_registry, benchmark_results tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db._declarative import Base


class DelayPrediction(Base):
    __tablename__ = "delay_predictions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    production_order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("production_orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    delay_probability: Mapped[float] = mapped_column(Float, nullable=False)
    delay_minutes_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    root_cause: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    top_risk_factors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mitigating_factors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    shap_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    order: Mapped["ProductionOrder"] = relationship(  # type: ignore[name-defined]
        "ProductionOrder", back_populates="predictions"
    )


class MlModelRegistry(Base):
    __tablename__ = "ml_model_registry"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    binary_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    regression_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    root_cause_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    is_champion: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    loaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    auc_roc: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_80_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    ece: Mapped[float | None] = mapped_column(Float, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
