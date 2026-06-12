"""
backend/app/api/routes/health.py
===================================
Health and readiness endpoints.

GET /health  — liveness probe (always 200 if process is running)
GET /ready   — readiness probe (200 only after ML service is loaded)
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    ml_service_loaded: bool


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    ml_loaded = (
        hasattr(request.app.state, "ml_service")
        and request.app.state.ml_service is not None
    )
    return ReadyResponse(
        status="ready" if ml_loaded else "initializing",
        ml_service_loaded=ml_loaded,
    )
