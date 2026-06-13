# ── Stage 1: builder ────────────────────────────────────────────────────────
# Contains gcc and libpq-dev so C extensions (asyncpg) can compile.
# This stage is discarded after the final COPY; build tools never ship.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first so pip layers are cached independently of source code.
COPY backend/pyproject.toml backend/pyproject.toml
COPY ml/pyproject.toml      ml/pyproject.toml
COPY ml/src                 ml/src

# Install mpc_ml first — its ==2.22.0 pin on mlflow locks the version before
# the backend's >=2.14 floor runs, preventing an accidental mlflow 3.x upgrade.
RUN pip install --no-cache-dir --timeout 120 "./ml"

# Install backend runtime deps.
# asyncpg compiles a C extension here; gcc is available in this stage only.
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


# ── Stage 2: runtime ────────────────────────────────────────────────────────
# Clean python:3.12-slim with only the runtime shared library (libpq5).
# gcc, libpq-dev, and all build-time headers are absent.
FROM python:3.12-slim AS runtime

# Runtime shared libraries required by compiled wheels:
#   libpq5     — PostgreSQL client (asyncpg)
#   libgomp1   — OpenMP runtime (LightGBM, Numba/SHAP)
#   libtbb12   — Intel TBB runtime (Numba tbbpool, pulled in by SHAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        libtbb12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring across all installed packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (tiny layer — bust cache only on code changes).
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
