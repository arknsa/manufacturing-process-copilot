"""
backend/app/db/base.py
========================
SQLAlchemy DeclarativeBase. Imports every model so Alembic autogenerate
discovers all tables. This is the only file that imports all models together.
"""

from __future__ import annotations

# Base is defined in _declarative.py to break the circular import that
# arises when model files import Base while base.py is loading models.
from backend.app.db._declarative import Base  # noqa: F401


# Import all models so Base.metadata is fully populated before Alembic runs.
# Order matters: referenced tables must be imported before referencing tables.
from backend.app.db.models.product import Product  # noqa: E402, F401
from backend.app.db.models.machine import Machine, MachineUtilizationLog  # noqa: E402, F401
from backend.app.db.models.operator import Operator  # noqa: E402, F401
from backend.app.db.models.order import ProductionOrder  # noqa: E402, F401
from backend.app.db.models.prediction import (  # noqa: E402, F401
    DelayPrediction,
    MlModelRegistry,
    BenchmarkResult,
)
from backend.app.db.models.bottleneck import BottleneckDetection  # noqa: E402, F401
from backend.app.db.models.recommendation import Recommendation  # noqa: E402, F401
from backend.app.db.models.chat_session import ChatSession  # noqa: E402, F401
from backend.app.db.models.chat_message import ChatMessage  # noqa: E402, F401
from backend.app.db.models.report import OperationalReport  # noqa: E402, F401
from backend.app.db.models.workflow_execution import WorkflowExecution  # noqa: E402, F401
from backend.app.db.models.audit_log import AuditLog  # noqa: E402, F401
