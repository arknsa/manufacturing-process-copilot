"""
backend/tests/unit/test_db_models.py
=======================================
Verifies that:
1. All 15 expected table names are registered in Base.metadata after
   importing base.py (which imports all 12 model files).
2. SQLAlchemy can create all tables against an in-memory SQLite database
   without errors.

No PostgreSQL or live DB required — uses sync SQLite via create_engine().
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text


EXPECTED_TABLES = {
    "products",
    "machines",
    "machine_utilization_logs",
    "operators",
    "production_orders",
    "delay_predictions",
    "ml_model_registry",
    "benchmark_results",
    "bottleneck_detections",
    "recommendations",
    "chat_sessions",
    "chat_messages",
    "operational_reports",
    "workflow_executions",
    "audit_logs",
}


def test_base_metadata_registers_all_tables():
    from backend.app.db.base import Base  # noqa: F401

    registered = set(Base.metadata.tables.keys())
    missing = EXPECTED_TABLES - registered
    assert not missing, f"Missing tables in Base.metadata: {missing}"


def test_all_tables_creatable_on_sqlite():
    """Create all ORM tables on an in-memory SQLite DB; verify via inspect."""
    from backend.app.db.base import Base

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    existing = set(inspector.get_table_names())

    missing = EXPECTED_TABLES - existing
    assert not missing, f"Tables not created in SQLite: {missing}"

    engine.dispose()


def test_production_orders_has_expected_columns():
    from backend.app.db.base import Base

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("production_orders")}

    required = {
        "id", "order_number", "planned_start", "planned_end",
        "quantity", "is_expedited", "priority", "status",
        "estimated_total_hours", "planned_lead_time_hours", "created_at",
    }
    missing = required - cols
    assert not missing, f"Missing columns in production_orders: {missing}"
    engine.dispose()


def test_delay_predictions_fk_column_exists():
    from backend.app.db.base import Base

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("delay_predictions")}
    assert "production_order_id" in cols
    assert "delay_probability" in cols
    assert "shap_values" in cols
    engine.dispose()
