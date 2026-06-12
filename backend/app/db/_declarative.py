"""
backend/app/db/_declarative.py
================================
Single source of truth for the SQLAlchemy DeclarativeBase.

Model files import Base from HERE — never from base.py — to avoid
the circular import that arises when base.py (which imports all models
for Alembic discovery) is loaded during a model-file import.

Outside code (Alembic env.py, session.py, tests) should import Base
from backend.app.db.base, which re-exports it.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
