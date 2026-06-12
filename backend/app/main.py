"""
backend/app/main.py
=====================
FastAPI application factory.

lifespan()   — loads ML service + LLM client at startup; cleans up on shutdown.
create_app() — assembles the FastAPI app with all routers and middleware.

Entry point for uvicorn:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import chat, health, models, orders, predictions, workflows
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.llm.client import LLMClient
from backend.app.services.ml.service import DelayPredictionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    app.state.ml_service = DelayPredictionService(settings)
    app.state.llm_client = LLMClient(settings)
    yield
    await app.state.llm_client.aclose()
    app.state.ml_service = None
    app.state.llm_client = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Manufacturing Process Copilot",
        description=(
            "ML-powered delay prediction, root-cause diagnosis, "
            "SHAP explainability, and AI copilot API for production planning."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predictions.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(orders.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(workflows.router, prefix="/api/v1")

    return app


app = create_app()
