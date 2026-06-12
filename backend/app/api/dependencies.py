"""
backend/app/api/dependencies.py
==================================
FastAPI Depends() providers.

get_db           — yields an AsyncSession; lazy import keeps asyncpg optional
                   at collection time (only required when a route is served)
get_ml_service   — returns the singleton DelayPredictionService from app.state
get_llm_client   — returns the singleton LLMClient from app.state
get_agent        — builds a per-request CopilotAgent (db + llm + registry)
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.agent.agent import CopilotAgent, build_registry
from backend.app.services.llm.client import LLMClient
from backend.app.services.ml.service import DelayPredictionService


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session.  session.py is lazy-imported so that asyncpg
    is not required at import time — only when a DB-backed route is served."""
    from backend.app.db.session import AsyncSessionLocal  # noqa: PLC0415

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_ml_service(request: Request) -> DelayPredictionService:
    return request.app.state.ml_service


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_agent(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CopilotAgent:
    """Build a per-request CopilotAgent using the singleton LLM client."""
    llm: LLMClient = request.app.state.llm_client
    registry = build_registry(db)
    return CopilotAgent(db=db, llm=llm, registry=registry)
