"""
backend/app/core/config.py
============================
Application settings loaded from environment variables.
All other modules call get_settings() — never os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Path to manufacturing-process-copilot/ root
_PROJ_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://mpc:mpc@localhost:5432/mpc"

    # ── MLflow ─────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = _PROJ_ROOT.joinpath("mlruns").resolve().as_uri()

    # ── Champion run IDs (Day 7 binary + regression, Day 8 root cause) ────
    BINARY_CHAMPION_RUN_ID: str = "140ce9025def4436a397ef8333078202"
    REGR_CHAMPION_RUN_ID: str = "d10e7217af3b4b68920d895c244ca1aa"
    RC_CHAMPION_RUN_ID: str = "7cc43338ae434163a2207e052354db1b"

    # ── LLM routing ────────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    OPENROUTER_FALLBACK_MODEL: str = "openai/gpt-oss-120b:free"
    OPENROUTER_CODER_MODEL: str = "qwen/qwen3-coder:free"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    LLM_TIMEOUT_SECONDS: float = 10.0

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── Business logic ─────────────────────────────────────────────────────
    PREDICTION_THRESHOLD: float = 0.65
    MODEL_NAME: str = "delay_predictor"

    # ── Notifications (n8n workflow alerts) ───────────────────────────────
    SLACK_WEBHOOK_URL: str = ""          # leave empty to disable Slack alerts
    ALERT_EMAIL_TO: str = ""             # leave empty to disable email alerts
    HIGH_RISK_THRESHOLD: float = 0.70    # delay_probability >= this triggers alert

    # ── App ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
