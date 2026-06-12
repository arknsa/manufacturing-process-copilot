# Manufacturing Process Copilot
## Document 05 — Repository Structure

**Status:** Authoritative Reference  
**Version:** 1.0  
**Role:** Staff ML Engineer + Senior Python Architect

---

## Architectural Principles

Five decisions shape every structural choice in this repository.

**1. One shared ML package, two consumers.** The feature engineering pipeline that trains the model must be byte-for-byte identical to the pipeline that serves predictions. The `mpc_ml` package solves this: it is defined once in `ml/`, installed as a dependency in `backend/`, and the serialised Pipeline object travels with every MLflow artifact. Training-serving skew is structurally impossible.

**2. Services own behaviour; routes own transport.** `backend/app/services/` contains all business logic — ML inference, LLM routing, agent orchestration. `backend/app/api/routes/` is thin: validate input, call service, return output. Tests target the service layer, not the routes.

**3. The agent lives in the backend, not the frontend.** The ReAct agent is a stateless function that receives a message and a memory snapshot, calls tools, and returns a response. Streamlit renders the result. This makes the agent independently testable and prevents state management leaking into the UI.

**4. The database is the source of truth for production state; MLflow is the source of truth for model state.** PostgreSQL stores orders, predictions, recommendations, and chat history. MLflow stores experiments, runs, artifacts, and the model registry. Neither owns the other's domain.

**5. Every external entry point is explicitly declared.** n8n calls FastAPI webhooks — those routes are isolated in `routes/workflows.py`. Streamlit calls the FastAPI REST API through a typed client. Nothing calls the database directly from the frontend. All cross-service communication is HTTP.

---

## Repository Tree

```
mpc/                                      ← Repository root
│
├── .github/
│   └── workflows/
│       └── ci.yml                        ← Lint + test on push
│
├── backend/                              ← FastAPI service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── health.py
│   │   │       ├── predictions.py
│   │   │       ├── models.py
│   │   │       ├── orders.py
│   │   │       ├── chat.py
│   │   │       └── workflows.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── session.py
│   │   │   ├── base.py
│   │   │   └── models/
│   │   │       ├── __init__.py
│   │   │       ├── product.py
│   │   │       ├── machine.py
│   │   │       ├── operator.py
│   │   │       ├── order.py
│   │   │       ├── prediction.py
│   │   │       ├── bottleneck.py
│   │   │       ├── recommendation.py
│   │   │       ├── chat_session.py
│   │   │       ├── chat_message.py
│   │   │       ├── report.py
│   │   │       ├── workflow_execution.py
│   │   │       └── audit_log.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── orders.py
│   │   │   ├── predictions.py
│   │   │   ├── chat.py
│   │   │   └── recommendations.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── ml/
│   │       │   ├── __init__.py
│   │       │   ├── service.py
│   │       │   ├── registry.py
│   │       │   └── explainability.py
│   │       ├── llm/
│   │       │   ├── __init__.py
│   │       │   ├── client.py
│   │       │   ├── prompts.py
│   │       │   └── streaming.py
│   │       └── agent/
│   │           ├── __init__.py
│   │           ├── agent.py
│   │           ├── memory.py
│   │           ├── tool_registry.py
│   │           └── tools/
│   │               ├── __init__.py
│   │               ├── orders.py
│   │               ├── predictions.py
│   │               ├── analytics.py
│   │               └── recommendations.py
│   ├── alembic/
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   │       ├── 001_factory_entities.py
│   │       ├── 002_ml_tables.py
│   │       └── 003_llm_agent_tables.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── unit/
│   │   │   ├── __init__.py
│   │   │   ├── test_ml_service.py
│   │   │   ├── test_explainability.py
│   │   │   └── test_agent.py
│   │   └── integration/
│   │       ├── __init__.py
│   │       └── test_api.py
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── alembic.ini
│
├── ml/                                   ← Training + simulation
│   ├── src/
│   │   └── mpc_ml/                       ← Installable shared package
│   │       ├── __init__.py
│   │       ├── features/
│   │       │   ├── __init__.py
│   │       │   ├── constants.py
│   │       │   ├── pipeline.py
│   │       │   └── transformers.py
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   ├── baseline.py
│   │       │   ├── tuning.py
│   │       │   └── evaluation.py
│   │       └── tracking/
│   │           ├── __init__.py
│   │           └── mlflow_utils.py
│   ├── scripts/
│   │   ├── generate_synthetic_data.py    ← DONE
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_baseline.ipynb
│   │   ├── 03_tuning.ipynb
│   │   └── 04_evaluation.ipynb
│   ├── data/
│   │   ├── raw/
│   │   │   └── README.md
│   │   └── processed/
│   │       └── README.md
│   ├── models/
│   │   └── README.md
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_pipeline.py
│   │   └── test_evaluation.py
│   ├── Dockerfile
│   └── pyproject.toml
│
├── frontend/                             ← Streamlit dashboard + chat
│   ├── app.py
│   ├── pages/
│   │   ├── 1_copilot_chat.py
│   │   ├── 2_risk_board.py
│   │   ├── 3_order_detail.py
│   │   └── 4_model_performance.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat_window.py
│   │   ├── risk_gauge.py
│   │   ├── shap_chart.py
│   │   └── metrics_table.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── api_client.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── formatting.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── n8n/
│   ├── workflows/
│   │   ├── 01_order_released_predict.json
│   │   ├── 02_daily_digest.json
│   │   └── 03_outcome_feedback_loop.json
│   └── README.md
│
├── docker/
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   └── .env.example
│
├── infra/
│   └── postgres/
│       └── init.sql
│
├── docs/
│   ├── 01_simulation_architecture.md     ← DONE
│   ├── 02_dataset_schema.md              ← DONE
│   ├── 03_feature_dictionary.md          ← DONE
│   └── 04_implementation_roadmap.md      ← DONE
│
├── scripts/
│   ├── setup_dev.sh
│   ├── run_simulation.sh
│   └── train_and_register.sh
│
├── .gitignore
├── Makefile
└── README.md
```

---

## Directory Catalogue

### `/backend/`
The FastAPI application. Contains the ML inference service, LLM routing layer, ReAct agent, REST API, database models, and all Alembic migrations. This is the only service that touches PostgreSQL directly. Packaged as a Docker image and exposed on port 8000.

### `/backend/app/`
The Python application package. Everything inside here is importable as `app.*`. Nothing outside this directory should be imported by the application.

### `/backend/app/api/`
HTTP transport layer only. Routes validate incoming Pydantic schemas, call the appropriate service method, and serialize the response. No business logic. No database queries.

### `/backend/app/api/routes/`
One file per resource domain. Each file registers a FastAPI `APIRouter` that is included in `main.py`.

### `/backend/app/core/`
Cross-cutting infrastructure: application settings and structured logging. Imported by everything else; imports nothing from `app.*`.

### `/backend/app/db/`
All SQLAlchemy infrastructure: session factory, declarative base, and one ORM model file per database table. Alembic migrations reference these models directly.

### `/backend/app/db/models/`
Twelve ORM model files, one per table. Each file is a thin SQLAlchemy declarative class that mirrors the schema in `infra/postgres/init.sql`. Business logic does not belong here.

### `/backend/app/schemas/`
Pydantic request and response schemas for the API layer. Separate from the ORM models — the API schema is what the outside world sees; the ORM model is what the database sees. The service layer maps between them.

### `/backend/app/services/`
All business logic lives here. Services are plain Python classes, dependency-injected via FastAPI's `Depends()`. They do not import from `app/api/`.

### `/backend/app/services/ml/`
ML inference: loads the production model from MLflow, runs feature preprocessing via the `mpc_ml` pipeline, computes predictions, and generates SHAP explanations. This is the performance-critical path — target <50ms per prediction.

### `/backend/app/services/llm/`
LLM abstraction layer. Handles provider selection (OpenRouter primary, Ollama fallback), request formatting, token counting, and SSE streaming. The agent and the report generator both call this; neither calls an LLM directly.

### `/backend/app/services/agent/`
ReAct agent implementation. Receives a user message and a memory snapshot, runs the think-act-observe loop, and returns a final response string. Calls tools via the tool registry. Reads and writes session memory via the memory module. Stateless between invocations.

### `/backend/app/services/agent/tools/`
One file per tool domain. Each file defines one or more async tool functions that the agent can call. Tools are pure async functions: they accept typed arguments and return structured results. They query the database or call internal services — never external APIs.

### `/backend/alembic/`
Database migration infrastructure. `env.py` is configured to discover all ORM models and apply migrations against the database URL from `config.py`. Versions are numbered sequentially and never modified after creation.

### `/backend/alembic/versions/`
Three migration files, each corresponding to a logical group of related tables. The grouping prevents a monolithic first migration that becomes impossible to debug.

### `/backend/tests/`
Pytest test suite for the backend. Unit tests mock external dependencies (MLflow, PostgreSQL, LLM). Integration tests run against a real PostgreSQL test database provisioned in `conftest.py`.

---

### `/ml/`
Everything related to dataset generation, ML training, and the shared feature engineering package. Has its own `pyproject.toml` and can run entirely independently of the backend during the ML development phase.

### `/ml/src/mpc_ml/`
The installable Python package. This is the single most architecturally important decision: the `mpc_ml` package is installed as `mpc-ml` in both `ml/pyproject.toml` (for training) and `backend/pyproject.toml` (for serving). Any change to the feature pipeline is made here and propagates to both consumers automatically. Using `src/` layout prevents accidental relative imports.

### `/ml/src/mpc_ml/features/`
Everything needed to transform a raw order record (37 columns) into a model-ready feature matrix. The `pipeline.py` file exposes a single public function, `build_pipeline()`, which returns a fitted or unfitted sklearn `Pipeline` object. The pipeline is the artifact that travels in MLflow alongside the model.

### `/ml/src/mpc_ml/models/`
Scikit-learn and boosting model factories, Optuna hyperparameter search, and evaluation utilities. All model definitions are here, not scattered across notebooks. Notebooks call functions from this module — they do not define models inline.

### `/ml/src/mpc_ml/tracking/`
Thin wrappers around MLflow's Python API. Standardises what gets logged (params, metrics, artifacts, tags) so every training run produces a consistent, queryable experiment. Includes the `promote_model()` function that transitions a run to production stage in the MLflow Model Registry.

### `/ml/scripts/`
Executable entry points. These are called by the `Makefile` and shell scripts. They are not importable modules — they do not define reusable functions. They parse CLI arguments, call `mpc_ml` functions, and handle top-level error reporting.

### `/ml/notebooks/`
Four Jupyter notebooks forming the analytical narrative of the ML development process. These are portfolio artefacts: they must be narrated, have clean outputs, and tell a story a non-ML reader can follow. They call functions from `mpc_ml` — they do not define logic inline.

### `/ml/data/`
Gitignored. Contains the raw and processed CSV files produced by `generate_synthetic_data.py`. Only `README.md` files are tracked, describing how to regenerate the data.

### `/ml/models/`
Gitignored. Local model artifact cache. MLflow is the authoritative model store; this directory is for local development convenience only.

### `/ml/tests/`
Pytest tests for the `mpc_ml` package. Contract tests for the feature pipeline (verify output shape, column names, no NaN) and unit tests for evaluation metrics. These run in CI without a database or MLflow server.

---

### `/frontend/`
Streamlit multi-page application. Contains four pages, four reusable UI components, a typed API client, and formatting utilities. Has no direct database access and no ML logic — it is a pure presentation layer.

### `/frontend/pages/`
Streamlit's file-based page router: files prefixed with numbers appear in the sidebar in order. Each page file is responsible for one view. Pages call `api_client.py` and `components/` — they do not call FastAPI directly with raw `requests`.

### `/frontend/components/`
Reusable Streamlit/Plotly UI elements. Each component is a function that accepts data and renders a visual element. Keeping them here prevents copy-paste across pages and makes the visual language consistent.

### `/frontend/services/`
The `api_client.py` file is the only place that knows the FastAPI base URL and authentication headers. All pages and components go through this client. If the API changes, only this file changes.

### `/frontend/utils/`
Pure Python functions: probability to risk label, minutes to human-readable duration, timestamp to display string. No Streamlit or HTTP imports allowed here.

---

### `/n8n/`
Exported n8n workflow JSON files. These are imported into the n8n UI via the import workflow function. The `README.md` documents the import procedure and describes each workflow's trigger, steps, and expected outcomes.

### `/docker/`
All Docker Compose configuration. The main `docker-compose.yml` defines the production stack of eight services. The `docker-compose.dev.yml` override adds volume mounts for live reload, exposes additional debug ports, and uses development-mode startup commands. `.env.example` is the canonical list of all required environment variables.

### `/infra/postgres/`
A single `init.sql` file that is mounted into the PostgreSQL container and executed on first start. It creates all tables exactly as specified in the architecture document. Alembic migrations are for subsequent schema changes; `init.sql` is only for the initial cold start.

### `/scripts/`
Three shell scripts invoked by `make` targets. They are human-readable wrappers around longer Python commands. Not part of the application — part of the developer experience.

---

## File Catalogue

### Root level

| File | Purpose |
|---|---|
| `Makefile` | Unified developer interface. Targets: `make simulate`, `make train`, `make up`, `make test`, `make migrate`, `make shell`. All CI commands run `make` targets — never raw scripts. |
| `README.md` | Portfolio entry point. Sections: what it is, why it's hard, architecture overview, how to run, results, known limitations. A technical reviewer should understand the project in 10 minutes. |
| `.gitignore` | Excludes: `ml/data/`, `ml/models/`, `mlruns/`, `__pycache__/`, `.env`, `*.pkl`, `*.pyc`, `.ipynb_checkpoints/`, `node_modules/`. |

---

### `/backend/app/`

| File | Purpose |
|---|---|
| `main.py` | FastAPI application factory. Defines `lifespan()` context manager: loads ML model on startup, releases on shutdown. Registers all APIRouters with their URL prefixes. Configures CORS and middleware. This is the Docker entrypoint. |
| `api/dependencies.py` | FastAPI `Depends()` providers: `get_db()` yields a SQLAlchemy session; `get_ml_service()` returns the singleton `DelayPredictionService`; `get_agent()` returns the singleton `CopilotAgent`. Centralising DI here means every route uses the same injection pattern. |
| `api/routes/health.py` | `GET /health` returns 200 + version string. `GET /ready` checks database connectivity and model load status. Used by Docker healthcheck and load balancer probes. |
| `api/routes/predictions.py` | `POST /api/v1/predictions/delay` — single-order prediction with SHAP explanation. `POST /api/v1/predictions/delay/batch` — up to 100 orders. Both store the prediction in `delay_predictions` table and return the serialised `DelayPrediction` schema. |
| `api/routes/models.py` | `GET /api/v1/models/current` — returns active model metadata from `ml_model_registry`. `GET /api/v1/models/feature-importance` — returns global SHAP importance ranking. Read-only; model promotion happens via MLflow CLI, not this API. |
| `api/routes/orders.py` | CRUD for `production_orders`: `POST /api/v1/orders/` creates an order and immediately fires a prediction. `GET /api/v1/orders/today` returns today's orders with their latest predictions. `PATCH /api/v1/orders/{id}/status` updates order status. |
| `api/routes/chat.py` | `POST /api/v1/chat/message` — sends a message to the ReAct agent and streams the response via SSE. `GET /api/v1/chat/sessions/{token}` — returns session history. `DELETE /api/v1/chat/sessions/{token}` — clears session memory. |
| `api/routes/workflows.py` | Webhook endpoints called by n8n: `POST /api/v1/webhooks/order-released`, `POST /api/v1/webhooks/shift-end`, `POST /api/v1/webhooks/feedback-loop`. These trigger async background tasks and return 202 Accepted immediately. |

---

### `/backend/app/core/`

| File | Purpose |
|---|---|
| `config.py` | `Settings(BaseSettings)` — reads from environment variables with `.env` fallback. Contains: `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`, `REDIS_URL`, `LOG_LEVEL`, `MODEL_NAME`, `SLACK_WEBHOOK_URL`. Exposes a `get_settings()` cached function used by `dependencies.py`. |
| `logging.py` | Configures `structlog` with JSON output in production and console output in development (determined by `LOG_LEVEL`). All service classes use `structlog.get_logger(__name__)`. Never use `print()` in the application. |

---

### `/backend/app/db/`

| File | Purpose |
|---|---|
| `session.py` | Creates the async SQLAlchemy engine and `AsyncSessionLocal` factory. Exposes `get_db()` async generator used by FastAPI's dependency injection. |
| `base.py` | SQLAlchemy `DeclarativeBase` subclass. Imports all model files so Alembic's autogenerate can discover them. This file is the only place all models are imported together. |

### `/backend/app/db/models/`

| File | Table(s) | Purpose |
|---|---|---|
| `product.py` | `products` | Product SKU, complexity, routing metadata. Static reference data loaded from simulation output. |
| `machine.py` | `machines`, `machine_utilization_logs` | Machine identity + type. Utilisation log records hourly snapshots for the utilisation feature. |
| `operator.py` | `operators` | Operator identity, skill tier, shift assignment. Reference data. |
| `order.py` | `production_orders` | The central operational table. Every incoming order lands here. Status transitions from `pending` through `in_progress` to `completed` or `delayed`. |
| `prediction.py` | `delay_predictions`, `ml_model_registry`, `benchmark_results` | Stores every prediction with its feature snapshot and SHAP values as JSONB. The `ml_model_registry` table mirrors MLflow's registry for fast API queries without calling MLflow. |
| `bottleneck.py` | `bottleneck_detections` | Records when the agent detects a machine bottleneck, with severity and affected orders. |
| `recommendation.py` | `recommendations` | Agent-generated action recommendations with full lifecycle tracking (open → acknowledged → actioned). |
| `chat_session.py` | `chat_sessions` | One row per conversation. Stores the context snapshot at session start and an LLM-generated summary for context compression. |
| `chat_message.py` | `chat_messages` | One row per message turn. Stores role, content, tool calls/results, model used, and token counts for cost tracking. |
| `report.py` | `operational_reports` | Shift summaries and handover briefs generated by n8n workflows. Stored as both structured JSON and rendered HTML. |
| `workflow_execution.py` | `workflow_executions` | Audit trail for every n8n workflow run. Enables debugging automation failures without accessing n8n's internal logs. |
| `audit_log.py` | `audit_logs` | Append-only audit trail for all state-changing operations. Every prediction, recommendation action, and model promotion creates a record here. |

---

### `/backend/app/schemas/`

| File | Purpose |
|---|---|
| `orders.py` | `OrderCreate` (inbound), `OrderResponse` (outbound), `OrderWithPrediction` (full view returned by `GET /orders/today`). |
| `predictions.py` | `OrderFeatures` (37-field input schema; used by both the API and the ML service), `DelayPrediction` (full prediction response with SHAP explanation), `FactorExplanation`, `BatchPredictionRequest`, `BatchPredictionResponse`. |
| `chat.py` | `ChatMessageRequest` (user message + session token), `ChatMessageResponse` (streaming SSE frame), `ChatSessionResponse` (session history). |
| `recommendations.py` | `RecommendationResponse`, `RecommendationStatusUpdate` (PATCH body for acknowledging/actioning). |

---

### `/backend/app/services/ml/`

| File | Purpose |
|---|---|
| `registry.py` | `MLflowModelRegistry` — loads the production-stage model and pipeline from MLflow on startup. Polls every 60 seconds for a version bump and hot-reloads if found. Caches both model and pipeline in memory. Exposes `get_model()`, `get_pipeline()`, `get_model_info()`. |
| `explainability.py` | `DelayExplainer` — wraps `shap.TreeExplainer`. Exposes `explain_order(features_dict) → ExplanationResult` and `global_importance() → List[FeatureImportance]`. Generates human-readable narratives by mapping SHAP values through feature label templates and a root-cause-specific sentence library. |
| `service.py` | `DelayPredictionService` — orchestrates `registry.py` and `explainability.py`. Accepts an `OrderFeatures` schema, runs the pipeline transform, calls the model, calls the explainer, persists the prediction to the database, and returns a `DelayPrediction`. Single public method: `predict(order: OrderFeatures) → DelayPrediction`. |

---

### `/backend/app/services/llm/`

| File | Purpose |
|---|---|
| `client.py` | `LLMClient` — handles provider routing. Primary path: OpenRouter API with `qwen/qwen3-80b-a3b:free` (reasoning) or `qwen/qwen3-coder:free` (structured output). Fallback path: Ollama local at `llama3.2:3b`. Circuit-breaker: if OpenRouter returns 3 consecutive errors or >10s timeout, switch to Ollama for the duration of the session. |
| `prompts.py` | All prompt templates as module-level constants. Includes: `SYSTEM_PROMPT` (copilot identity + factory context), `TOOL_SELECTION_PROMPT` (ReAct reasoning step), `EXPLANATION_NARRATIVE_PROMPT` (delay explanation generation), `SHIFT_HANDOVER_PROMPT`, `ROOT_CAUSE_ANALYSIS_PROMPT`. Prompts are strings with `{placeholder}` slots — no f-strings, no logic. |
| `streaming.py` | Utilities for FastAPI SSE streaming responses. `stream_llm_response(generator) → AsyncGenerator[str, None]` converts a streaming LLM response into SSE-formatted chunks. Used by `routes/chat.py` to stream agent responses to the Streamlit frontend. |

---

### `/backend/app/services/agent/`

| File | Purpose |
|---|---|
| `agent.py` | `CopilotAgent` — the ReAct loop. `run(message: str, session_token: str) → AsyncGenerator[str, None]`. Each iteration: (1) load memory, (2) build prompt with context, (3) call LLM for reasoning, (4) parse tool call if present, (5) execute tool via registry, (6) append observation, (7) repeat or return final answer. Maximum 5 iterations per turn to bound token usage. |
| `memory.py` | `SessionMemory` — reads and writes `chat_sessions` + `chat_messages` tables. `load(session_token) → List[Message]` returns the last N messages (configurable, default 10). `save(session_token, role, content, tool_data)` appends a message row. `compress(session_token)` calls the LLM to summarise and truncate old messages when the session exceeds a token budget. |
| `tool_registry.py` | `ToolRegistry` — a mapping from tool name strings to async callable functions. `register(name, func, schema)` adds a tool. `dispatch(tool_name, arguments) → str` executes a tool and returns its JSON-serialised result. The registry is the only code that calls tool functions; the agent calls the registry. |

### `/backend/app/services/agent/tools/`

| File | Tools defined | Purpose |
|---|---|---|
| `orders.py` | `get_production_order`, `get_active_orders`, `get_orders_at_risk` | Query `production_orders` and join with `delay_predictions`. Return structured summaries the agent can reason about. |
| `predictions.py` | `get_delay_prediction`, `get_risk_summary`, `get_feature_explanation` | Retrieve stored predictions and SHAP explanations from `delay_predictions`. Allow the agent to answer "why is this order at risk?" |
| `analytics.py` | `get_machine_history`, `get_bottlenecks`, `get_shift_summary`, `get_kpi_dashboard` | Aggregate queries for machine performance, active bottlenecks, and shift-level KPIs. These are the data retrieval tools for analytical queries. |
| `recommendations.py` | `create_recommendation`, `get_recommendations`, `update_recommendation_status` | Write and read from `recommendations` table. The agent creates recommendations via this tool; supervisors acknowledge them via the API. |

---

### `/backend/alembic/versions/`

| File | Purpose |
|---|---|
| `001_factory_entities.py` | Creates `products`, `machines`, `operators`, `production_orders`, `machine_utilization_logs`. These are the operational tables that exist independent of ML. |
| `002_ml_tables.py` | Creates `ml_model_registry`, `benchmark_results`, `delay_predictions`, `bottleneck_detections`. Depends on `001` (references `production_orders`, `machines`). |
| `003_llm_agent_tables.py` | Creates `chat_sessions`, `chat_messages`, `recommendations`, `operational_reports`, `workflow_executions`, `audit_logs`. Depends on `001` and `002`. |

---

### `/ml/src/mpc_ml/features/`

| File | Purpose |
|---|---|
| `constants.py` | Module-level constants that define the contract between simulation and ML: `FEATURE_COLS` (ordered list of 37 feature names), `TARGET_COLS`, `LOG_FEATURES` (columns requiring log1p), `BINARY_FEATURES`, `ORDINAL_FEATURES`, `SCALE_FEATURES`, `PASSTHROUGH_FEATURES`, `INTERACTION_FEATURES`. This file is the single source of truth for what the pipeline processes. |
| `transformers.py` | Two custom sklearn transformers: `InteractionFeatureAdder` (adds the four derived interaction columns defined in Doc 04) and `ColumnSelector` (selects exactly `FEATURE_COLS` from an arbitrary DataFrame, handles cold-start defaults). Both implement `fit()`, `transform()`, and `get_feature_names_out()` for sklearn Pipeline compatibility. |
| `pipeline.py` | Exports `build_pipeline(fitted: bool = False) → Pipeline`. Assembles a `ColumnTransformer` wrapping all preprocessing steps (log+scale, scale-only, passthrough) inside a final `Pipeline` with `InteractionFeatureAdder` as the first step. When `fitted=False`, returns an unfitted pipeline ready for `fit_transform(train_df)`. The fitted pipeline is serialised alongside the model in MLflow. |

### `/ml/src/mpc_ml/models/`

| File | Purpose |
|---|---|
| `baseline.py` | `get_baseline_models() → Dict[str, Any]` returns the five baseline estimators (LogisticRegression, DecisionTree, RandomForest, XGBoost, LightGBM) each pre-configured with appropriate class weights and random state. `get_task_models(task: str) → Dict` returns the appropriate estimators for each of the four prediction tasks (binary, regression, ordinal, multi-class). |
| `tuning.py` | `build_optuna_objective(X_train, y_train, model_type) → Callable` — factory that returns an Optuna objective function. `run_study(objective, n_trials=100) → Study` — runs the Optuna study with MedianPruner. `best_params_to_model(study, model_type) → Estimator` — reconstructs the champion model from the best trial's params. |
| `evaluation.py` | `evaluate_model(model, pipeline, X, y, task) → MetricsDict` — computes the full metric set for a given task. `calibration_report(model, X, y) → CalibrationResult` — ECE + reliability diagram data. `confusion_matrix_annotated(model, X, y) → DataFrame` — confusion matrix with precision/recall per class. `precision_at_recall(model, X, y, target_recall=0.80) → float` — finds the threshold that achieves target recall and returns the corresponding precision. |

### `/ml/src/mpc_ml/tracking/`

| File | Purpose |
|---|---|
| `mlflow_utils.py` | Wrappers that enforce consistent MLflow logging: `start_run(experiment, run_name)` context manager; `log_pipeline(pipeline)` serialises and logs the sklearn Pipeline as an artifact; `log_model_with_signature(model, pipeline, sample_input)` logs the model with an inferred MLflow signature; `log_standard_metrics(metrics_dict)` logs the full evaluation dict; `promote_to_production(run_id, model_name)` transitions the model to Production stage and archives the previous version. |

---

### `/ml/scripts/`

| File | Purpose |
|---|---|
| `generate_synthetic_data.py` | **DONE.** 540-day factory simulation. CLI: `--days`, `--orders-per-day`, `--machines`, `--seed`, `--output-dir`. Outputs `synthetic_factory_data.csv`, `train/val/test.csv`, `simulation_report.json`. |
| `train.py` | CLI entry point for the full training pipeline. Loads train/val/test CSVs, fits the `mpc_ml` pipeline on train, trains all four task models, evaluates on val, logs everything to MLflow, and optionally promotes the champion to production. Accepts `--experiment`, `--promote`, `--task` flags. |
| `evaluate.py` | CLI entry point for test-set evaluation. Loads the production model from MLflow, evaluates on `test.csv`, prints a full report, and logs final test metrics as a new MLflow run tagged `evaluation_type=final_test`. Run exactly once per model version before promotion. |

---

### `/ml/notebooks/`

| File | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory analysis. Sections: dataset overview, target distribution, causal relationship validation (4 plots), feature distributions by class, correlation heatmap, cold-start analysis for rolling features, skewness table, temporal stability check. Output is the analytical basis for all preprocessing decisions in `pipeline.py`. |
| `02_baseline.ipynb` | Baseline model comparison. Trains all five models from `baseline.py` using the fitted pipeline from `pipeline.py`. Produces a comparison table of AUC, AP, F1 across all models. Identifies the champion. Logged to MLflow under the `baseline` experiment. |
| `03_tuning.ipynb` | Optuna hyperparameter optimisation for XGBoost and LightGBM. Runs 100 trials. Shows optimisation history, parameter importance, and the best trial. Final champion params are saved as a cell output and fed into `train.py`. |
| `04_evaluation.ipynb` | Final model evaluation. SHAP beeswarm plot (top 20 features), calibration curve, confusion matrix at operational threshold, precision-at-80%-recall, example explanations for high-risk and low-risk orders. This notebook is the ML section of the portfolio README. |

---

### `/ml/tests/`

| File | Purpose |
|---|---|
| `conftest.py` | Loads a 50-row sample of `test.csv` as a pytest fixture. Builds an unfitted pipeline as a fixture. Defines a mock MLflow client fixture. |
| `test_pipeline.py` | Contract tests: verify `build_pipeline().fit_transform(train_df)` produces a matrix with the expected shape; verify no NaN in output; verify column count equals `len(FEATURE_COLS)`; verify `get_feature_names_out()` returns the correct names after log+scale transforms. |
| `test_evaluation.py` | Tests `evaluate_model()` with a dummy classifier. Verifies the metrics dict has all required keys. Tests `precision_at_recall()` boundary conditions. |

---

### `/frontend/`

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point. Configures page layout, sidebar navigation, and session state initialisation (API base URL, auth token placeholder, active session token). Does not render content directly — delegates to pages. |

### `/frontend/pages/`

| File | Purpose |
|---|---|
| `1_copilot_chat.py` | Chat interface. Renders the `chat_window` component, handles user input, calls `POST /api/v1/chat/message` with streaming enabled, appends assistant response chunks to the message history as they arrive. Initialises a session token on first load and persists it in `st.session_state`. |
| `2_risk_board.py` | Live order risk board. Calls `GET /api/v1/orders/today`, renders a colour-coded table sorted by delay probability descending. Row click navigates to the Order Detail page with the selected order ID. Refreshes every 60 seconds via `st.rerun()`. |
| `3_order_detail.py` | Single-order view. Receives an `order_id` via query param. Calls `GET /api/v1/predictions/delay/{id}`. Renders: delay probability gauge, top risk factors bar chart, root cause prediction, narrative explanation, recommended actions, and a link to open the chat asking about this specific order. |
| `4_model_performance.py` | MLops dashboard. Calls `GET /api/v1/models/current` and `GET /api/v1/models/feature-importance`. Renders: current model metrics card, feature importance bar chart, rolling 30-day precision/recall chart (from completed orders in the database), and a placeholder for PSI drift indicators. |

### `/frontend/components/`

| File | Purpose |
|---|---|
| `chat_window.py` | `render_chat_messages(messages: List[dict])` — renders a list of chat messages as styled bubbles (user right, assistant left). Handles tool call display as collapsible expanders. Used by page 1. |
| `risk_gauge.py` | `render_risk_gauge(probability: float, label: str)` — Plotly gauge chart with three zones: green (<40%), amber (40–65%), red (>65%). Consistent visual language across the application. |
| `shap_chart.py` | `render_shap_waterfall(factors: List[FactorExplanation])` — horizontal Plotly bar chart of SHAP contributions. Positive values (increase risk) in red, negative (reduce risk) in green. Feature labels use the human-readable names from Doc 03. |
| `metrics_table.py` | `render_metrics_comparison(models: List[dict])` — styled `st.dataframe` with conditional cell colouring for AUC, F1, and precision columns. Used in both notebook 02 and the model performance page. |

### `/frontend/services/`

| File | Purpose |
|---|---|
| `api_client.py` | `MpcApiClient` — single `httpx.AsyncClient` instance shared across pages via `st.session_state`. Methods: `predict(order_features)`, `predict_batch(orders)`, `get_today_orders()`, `get_order_detail(order_id)`, `stream_chat(message, session_token)`, `get_model_info()`, `get_feature_importance()`. This is the only file that constructs URLs or sets headers. |

### `/frontend/utils/`

| File | Purpose |
|---|---|
| `formatting.py` | Pure functions, no imports from `streamlit` or `httpx`: `prob_to_risk_label(p: float) → str`, `minutes_to_display(m: int) → str` ("4h 12m"), `root_cause_to_display(rc: str) → str` ("Setup Overrun"), `risk_colour(p: float) → str` ("#e74c3c"). Tested independently, no side effects. |

---

### `/n8n/workflows/`

| File | Purpose |
|---|---|
| `01_order_released_predict.json` | Triggered by webhook from `POST /webhooks/order-released`. Calls the delay prediction endpoint, checks if `probability > 0.65`, and if so sends a Slack notification with the order number, probability, and top risk factor. Error handling: retry 3× on HTTP failure, log to `workflow_executions` table. |
| `02_daily_digest.json` | Scheduled at 06:00 Monday–Friday. Calls `GET /api/v1/orders/today`, aggregates orders by risk level, calls `POST /api/v1/chat/message` with a shift summary prompt, and emails the resulting HTML report to a configurable recipient list. |
| `03_outcome_feedback_loop.json` | Scheduled at 23:00 Monday–Friday. Queries completed orders from the last 24 hours, calls `POST /api/v1/predictions/feedback` with actual vs predicted outcomes, and updates the rolling precision/recall metrics used by the model performance dashboard. |

---

### `/docker/`

| File | Purpose |
|---|---|
| `docker-compose.yml` | Defines eight services: `db` (postgres:16, port 5432), `redis` (redis:7, port 6379), `mlflow` (ghcr.io/mlflow/mlflow, port 5000), `api` (custom Dockerfile, port 8000), `frontend` (custom Dockerfile, port 8501), `n8n` (n8nio/n8n, port 5678), `ollama` (ollama/ollama, port 11434). Named volumes for postgres data, mlflow artifacts, and n8n data. Health checks on all services. `api` depends_on `db`, `redis`, `mlflow`. |
| `docker-compose.dev.yml` | Overrides for local development: mounts `backend/app/` and `frontend/` as volumes for live reload, exposes `5432` and `6379` directly for database tooling, sets `RELOAD=true` on the API service, sets `LOG_LEVEL=DEBUG`. |
| `.env.example` | Documents every environment variable with a placeholder value and a one-line description. Variables: `DATABASE_URL`, `REDIS_URL`, `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_ROOT`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `SLACK_WEBHOOK_URL`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_RECIPIENTS`, `MODEL_NAME`, `LOG_LEVEL`, `PREDICTION_THRESHOLD`. |

---

### `/infra/postgres/`

| File | Purpose |
|---|---|
| `init.sql` | Creates all 15 tables exactly as defined in the architecture document's schema section. Mounted as `/docker-entrypoint-initdb.d/init.sql` in the postgres container — runs automatically on first start. Does not use Alembic; Alembic migrations are for post-init changes only. Also creates indexes on the most-queried columns: `production_orders(planned_start)`, `delay_predictions(production_order_id)`, `chat_messages(session_id, created_at)`. |

---

### `/.github/workflows/`

| File | Purpose |
|---|---|
| `ci.yml` | Two jobs: `test-ml` (installs `mpc_ml`, runs `ml/tests/` with pytest) and `test-backend` (starts a postgres:16 container, installs backend deps, runs `backend/tests/` with pytest). Triggers on push to `main` and on all pull requests. Fast: targets <3 minutes total. |

---

## Development Order

The sequence is driven by one constraint: the `mpc_ml` feature pipeline must be implemented before anything that uses it, and the database schema must exist before services that query it.

### Phase 0 — Foundation ✓ Done

```
ml/scripts/generate_synthetic_data.py       ✓ Implemented + calibrated
docs/01–04_*.md                             ✓ Finalised
```

### Phase 1 — ML Package (Day 3, today)

Implement the `mpc_ml` package. This is the critical-path deliverable — nothing downstream works without it.

```
ml/pyproject.toml                           Define package, deps, entry points
ml/src/mpc_ml/features/constants.py        Feature column lists (FIRST file written)
ml/src/mpc_ml/features/transformers.py     InteractionFeatureAdder, ColumnSelector
ml/src/mpc_ml/features/pipeline.py         build_pipeline() function
ml/src/mpc_ml/models/evaluation.py         Metrics helpers (no model deps)
ml/src/mpc_ml/tracking/mlflow_utils.py     MLflow wrappers
ml/src/mpc_ml/models/baseline.py           Model factories
ml/src/mpc_ml/models/tuning.py             Optuna objective builder
ml/tests/conftest.py
ml/tests/test_pipeline.py                  Write tests alongside pipeline
ml/tests/test_evaluation.py
```

### Phase 2 — ML Notebooks (Days 4–5)

With the package installed, the notebooks are driving code — they discover problems in constants and pipeline assumptions.

```
ml/notebooks/01_eda.ipynb                  EDA; confirm causal relationships
ml/notebooks/02_baseline.ipynb             5-model comparison; select champion family
ml/notebooks/03_tuning.ipynb               Optuna; find best hyperparameters
ml/scripts/train.py                        Promote best run to MLflow production
ml/notebooks/04_evaluation.ipynb           Final test-set evaluation + SHAP
ml/scripts/evaluate.py                     Record final test metrics
```

**Gate:** Val AUC ≥ 0.75 before proceeding. If below target, return to `constants.py` and add interaction features.

### Phase 3 — Backend Foundation (Days 6–7)

Set up the database and the infrastructure the service layer depends on.

```
backend/pyproject.toml                     Declare deps including mpc-ml
backend/app/core/config.py                 Settings; validates env vars on startup
backend/app/core/logging.py               Structured logging
backend/app/db/base.py                     DeclarativeBase
backend/app/db/models/product.py           Reference tables first
backend/app/db/models/machine.py
backend/app/db/models/operator.py
backend/app/db/models/order.py             Core operational table
backend/app/db/models/prediction.py        ML tables
backend/app/db/models/bottleneck.py
backend/app/db/models/recommendation.py
backend/app/db/models/chat_session.py
backend/app/db/models/chat_message.py
backend/app/db/models/report.py
backend/app/db/models/workflow_execution.py
backend/app/db/models/audit_log.py
backend/app/db/session.py                  Session factory (after models defined)
backend/alembic.ini
backend/alembic/env.py
backend/alembic/versions/001_factory_entities.py
backend/alembic/versions/002_ml_tables.py
backend/alembic/versions/003_llm_agent_tables.py
infra/postgres/init.sql                    Cold-start SQL (mirrors ORM models)
```

### Phase 4 — ML Service (Day 7)

Now the database exists, the model is in MLflow, and `mpc_ml` is installed.

```
backend/app/schemas/predictions.py         OrderFeatures, DelayPrediction schemas
backend/app/schemas/orders.py
backend/app/services/ml/registry.py        Load model from MLflow
backend/app/services/ml/explainability.py  SHAP service
backend/app/services/ml/service.py         Orchestrates registry + explainability
backend/tests/unit/test_ml_service.py      Test with mocked MLflow
backend/tests/unit/test_explainability.py
```

### Phase 5 — LLM Service (Day 8)

```
backend/app/services/llm/prompts.py        Prompts first (no deps)
backend/app/services/llm/streaming.py      SSE helpers
backend/app/services/llm/client.py         OpenRouter + Ollama routing
```

### Phase 6 — Agent Layer (Day 8–9)

```
backend/app/services/agent/tools/orders.py         Simplest tools first
backend/app/services/agent/tools/predictions.py
backend/app/services/agent/tools/analytics.py
backend/app/services/agent/tools/recommendations.py
backend/app/services/agent/tool_registry.py        After all tools defined
backend/app/services/agent/memory.py               DB-backed session memory
backend/app/services/agent/agent.py                ReAct loop (last, deps on all above)
backend/tests/unit/test_agent.py
```

### Phase 7 — API Routes + App Assembly (Day 9–10)

```
backend/app/api/dependencies.py            DI providers
backend/app/api/routes/health.py           First route; confirms stack works
backend/app/schemas/chat.py
backend/app/schemas/recommendations.py
backend/app/api/routes/predictions.py
backend/app/api/routes/models.py
backend/app/api/routes/orders.py
backend/app/api/routes/chat.py
backend/app/api/routes/workflows.py
backend/app/main.py                        Assembles all routers; lifespan hooks
backend/tests/conftest.py                  Test DB, test client
backend/tests/integration/test_api.py      Smoke tests for all routes
```

**Gate:** All integration tests pass. `GET /health` and `POST /predictions/delay` work end-to-end.

### Phase 8 — Frontend (Days 11–13)

```
frontend/requirements.txt                  Streamlit, httpx, plotly
frontend/utils/formatting.py               Pure functions first
frontend/services/api_client.py            Typed API client (before pages)
frontend/components/risk_gauge.py
frontend/components/shap_chart.py
frontend/components/chat_window.py
frontend/components/metrics_table.py
frontend/app.py                            Entry point + session state init
frontend/pages/2_risk_board.py             Simplest page first (read-only)
frontend/pages/3_order_detail.py
frontend/pages/4_model_performance.py
frontend/pages/1_copilot_chat.py           Last (most complex; streaming SSE)
frontend/Dockerfile
```

### Phase 9 — Automation + Infrastructure (Day 14–15)

```
n8n/workflows/01_order_released_predict.json
n8n/workflows/02_daily_digest.json
n8n/workflows/03_outcome_feedback_loop.json
n8n/README.md
docker/docker-compose.yml
docker/docker-compose.dev.yml
docker/.env.example
```

### Phase 10 — Polish (Week 4)

```
scripts/setup_dev.sh
scripts/run_simulation.sh
scripts/train_and_register.sh
Makefile
.github/workflows/ci.yml
README.md                                   Final portfolio README
```

---

## Service Dependency Map

The table below reads as "this service requires X to be running / available."

| Service | Requires at startup | Requires at runtime |
|---|---|---|
| `api` (FastAPI) | `db`, `mlflow` | `db`, `redis`, `mlflow`, `ollama` (or OpenRouter) |
| `frontend` (Streamlit) | `api` | `api` |
| `mlflow` | — | `db` (as artifact store backend, via S3/local) |
| `n8n` | — | `api` (all workflow steps call FastAPI) |
| `ollama` | — | — (stateless) |
| `redis` | — | — (stateless) |
| `db` (postgres) | — | — |

**Cold start order:** `db` → `redis` + `mlflow` + `ollama` → `api` → `frontend` + `n8n`

---

## Package Dependency Map

| Package | Depends on `mpc_ml` | Depends on `backend` | Notes |
|---|---|---|---|
| `mpc_ml` | — | — | No dependencies on application code |
| `backend` | ✓ (for serving) | — | Installs `mpc-ml` from `ml/` path |
| `ml` scripts | ✓ (for training) | — | Installs `mpc-ml` from `ml/` path |
| `frontend` | — | — | Calls FastAPI via HTTP only; no Python imports |

`mpc_ml` depends only on: `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap`, `pandas`, `numpy`, `mlflow`. It has no FastAPI, Streamlit, or SQLAlchemy dependencies.

---

*Document 05 of 05 — Manufacturing Process Copilot Technical Series*
