FROM python:3.12-slim

WORKDIR /app

# Build tools required by asyncpg (C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless these files change) ─────────────────────
COPY backend/pyproject.toml backend/pyproject.toml
# mpc_ml is a local package required by the pickled sklearn pipelines at load time
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/src ml/src

# Install mpc_ml and its declared deps (lightgbm, xgboost, shap, etc.)
RUN pip install --no-cache-dir --timeout 120 "./ml"

# Install runtime deps declared in [project].dependencies
# --timeout 120  → survives slow PyPI mirrors (default 15 s trips ReadTimeoutError)
# --no-cache-dir → keeps image small
RUN pip install --no-cache-dir --timeout 120 \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.29" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.3" \
        "mlflow>=2.14" \
        "sqlalchemy[asyncio]>=2.0" \
        "asyncpg>=0.29" \
        "alembic>=1.13" \
        "httpx>=0.27" \
        "aiohttp>=3.9"

# ── Application source ───────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
