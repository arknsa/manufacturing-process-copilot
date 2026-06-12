# Manufacturing Process Copilot
## Document 04 — Final Implementation Roadmap

**Status:** Active  
**Phase:** Post-simulation — ML pipeline begins now  
**Completed:** Factory simulation, synthetic dataset generation  
**Next milestone:** EDA notebook + baseline ML model

---

## Roadmap Overview

```
Week 1  ──── Data & Baseline
  Day 1  [DONE] Simulation architecture + data strategy
  Day 2  [DONE] Factory simulation implementation + calibration
  Day 3  Feature engineering pipeline (sklearn Pipeline)
  Day 4  EDA notebook (01_eda.ipynb)
  Day 5  Baseline model + experiment tracking setup

Week 2  ──── ML Core
  Day 6  Benchmarking framework (5 algorithms)
  Day 7  Hyperparameter tuning (Optuna)
  Day 8  SHAP explainability layer
  Day 9  ML service API (FastAPI)
  Day 10  Model deployment + MLflow integration

Week 3  ──── Product
  Day 11  Delay prediction endpoint + business logic
  Day 12  Root cause explanation engine
  Day 13  Streamlit dashboard V1
  Day 14  n8n workflow automation
  Day 15  End-to-end integration test

Week 4  ──── Portfolio Polish
  Day 16  Docker Compose full stack
  Day 17  Performance benchmarking + documentation
  Day 18  Demo data + walkthrough recording
  Day 19  GitHub README + architecture diagrams
  Day 20  Portfolio submission
```

---

## Week 1, Day 3 — Feature Engineering Pipeline

**Deliverable:** `backend/app/services/ml/pipeline.py`

The feature engineering pipeline is a scikit-learn `Pipeline` object that transforms the raw CSV columns into a model-ready matrix. It must be:
- Fit only on train split (never val or test)
- Reproducible: no randomness inside transformers
- Serialisable: saved as a pickle alongside the trained model in MLflow

### Preprocessing specification by feature group

**Log-normal skewed features** — apply `log1p` via `FunctionTransformer` before standard scaling:
`planned_lead_time_hours`, `release_lag_hours`, `estimated_total_hours`, `quantity`, `operator_experience_months`, `machine_unplanned_downtime_hours_30d`

**Already-bounded features** — standard scale only (no log transform):
`schedule_tightness_ratio`, `machine_utilization_at_release`, `machine_oee_30d`, `product_delay_rate_90d`, `machine_delay_rate_90d`, `operator_delay_rate_90d`, `product_x_machine_delay_rate_90d`, `product_first_pass_yield_90d`, `machine_setup_overrun_rate_90d`, `shift_delay_rate_30d`

**Binary/boolean features** — pass through unchanged:
`is_expedited`, `is_month_end`, `is_quarter_end`, `material_availability_at_release`, `maintenance_due_within_order_window`, `changeover_required`

**Low-cardinality ordinal** — pass through unchanged (tree models handle these natively):
`priority_encoded`, `operator_skill_tier_encoded`, `shift_type_encoded`, `planned_start_day_of_week`

**Integer count features** — pass through (already on sensible scale):
`operation_count`, `component_shortage_count`, `material_bom_complexity`, `work_center_queue_depth_at_release`, `operator_concurrent_order_count`

**Float features** — standard scale:
`product_complexity_score`, `changeover_complexity_score`, `hours_into_shift_at_start`, `days_since_last_planned_maintenance`, `schedule_revision_count`, `planned_start_hour`

### Pipeline structure

```python
# ColumnTransformer with named transformers:
# 'log_scale'      → FunctionTransformer(log1p) → StandardScaler
# 'scale_only'     → StandardScaler
# 'passthrough'    → remainder='passthrough'
```

### Derived interaction features to engineer

Add these as a `FunctionTransformer` step before the column transformer:

| Derived feature | Formula | Rationale |
|---|---|---|
| `tightness_x_queue` | `schedule_tightness_ratio × work_center_queue_depth_at_release` | Multiplicative congestion risk |
| `lag_as_pct_of_window` | `release_lag_hours / planned_lead_time_hours` | How much of the lead time is consumed before work starts |
| `log_experience_x_concurrent` | `log1p(operator_experience_months) × operator_concurrent_order_count` | Zero for all current orders; ready for future model |
| `oee_x_maintenance_overdue` | `machine_oee_30d × (1 / max(1, days_since_last_planned_maintenance / 30))` | OEE adjusted by maintenance recency |

**Note:** Do not add interaction features blindly. Compute them, test with and without via cross-validation on train split, and keep only those that improve validation AUC. Tree models may not need them at all.

---

## Week 1, Day 4 — EDA Notebook

**Deliverable:** `ml/notebooks/01_eda.ipynb`

The EDA notebook serves two purposes: confirming simulation calibration and discovering ML-relevant patterns. It is a portfolio artefact — it must be clean, narrated, and explainable to a non-ML reader.

### Required sections

**1. Dataset overview**
Row count, class balance, missing value check (must confirm 0%), split sizes and temporal continuity.

**2. Target distribution**
`is_delayed` bar chart. `delay_minutes` histogram with log x-axis. `delay_category` pie chart. `delay_root_cause` bar chart (delayed orders only).

**3. Causal relationship validation**
Four plots confirming the simulation's causal design:
- Delay rate vs `schedule_tightness_ratio` quartile (monotone increase expected)
- Delay rate by `material_availability_at_release` (shortage >> available)
- Delay rate by `priority_encoded` (critical >> high >> normal)
- Delay rate vs `machine_utilization_at_release` buckets (high util > low util, structurally)

**4. Feature distributions**
Histograms and KDE plots for all 37 features, coloured by `is_delayed`. Focus commentary on the 6 Tier 1 features.

**5. Correlation analysis**
Full correlation heatmap (37×37). Highlight: top 10 features by |r| with `is_delayed`. Flag pairs with |r|>0.80 as potential multicollinearity risks (`product_delay_rate_90d` vs `product_x_machine_delay_rate_90d`).

**6. Historical feature cold-start analysis**
Plot rolling feature values vs simulation day. Quantify what fraction of the training set has default (cold-start) values. This informs how the model handles unseen product-machine pairs in production.

**7. Temporal stability check**
Plot weekly delay rate over the 540-day simulation. Confirm no extreme drift (>15pp swing within any 90-day window). If drift exists, note it for the ML evaluation strategy.

**8. Skewness and transform decisions**
Skewness table for all continuous features. Confirm which features benefit from log1p (skewness >1.5 pre-transform).

---

## Week 1, Day 5 — Baseline Model + Experiment Tracking

**Deliverable:** `ml/notebooks/02_baseline.ipynb` + MLflow experiment `mpc/delay_prediction/baseline`

### Baseline algorithms

Run in this order. Each run is a tracked MLflow experiment. Every run must log: all hyperparameters, train/val AUC, val precision/recall/F1, confusion matrix, and the fitted Pipeline object.

| Model | Library | Why this model |
|---|---|---|
| Logistic Regression | sklearn | Linear baseline; establishes minimum bar |
| Decision Tree (depth=5) | sklearn | Human-interpretable tree for stakeholder demos |
| Random Forest | sklearn | Strong ensemble baseline; handles mixed types well |
| **XGBoost** | xgboost | **Primary model candidate** |
| LightGBM | lightgbm | **Primary model candidate** |

For delay prediction, the primary metric is **ROC-AUC** on the validation split. Secondary metrics: precision at 80% recall (operational threshold for alert generation) and F1 at default 0.5 threshold.

### Class weight handling

Set `class_weight='balanced'` for sklearn models. Set `scale_pos_weight = (n_negative / n_positive)` for XGBoost and LightGBM. This is required because the 63/37 imbalance creates subtle calibration issues in probability outputs.

### Baseline performance expectations

Based on feature correlations and domain analogues:

| Model | Expected val AUC range |
|---|---|
| Logistic Regression | 0.68–0.75 |
| Decision Tree | 0.65–0.72 |
| Random Forest | 0.78–0.84 |
| XGBoost | 0.82–0.88 |
| LightGBM | 0.82–0.88 |

If XGBoost/LightGBM AUC is below 0.75, check: (1) data leakage absence confirmed, (2) train/val split is truly temporal, (3) no target encoding on training data.

---

## Week 2, Day 6 — Benchmarking Framework

**Deliverable:** `backend/app/services/ml/benchmarking.py`

A reproducible benchmarking harness that evaluates all five models on a consistent evaluation protocol and produces a comparison report.

### Evaluation protocol

- **Cross-validation:** TimeSeriesSplit with 5 folds on train set (preserves temporal order)
- **Final evaluation:** Train on full train set, evaluate on held-out val set (model selection), final AUC reported on test set (single evaluation, never touched during development)
- **Metrics logged per model:** AUC, AP (Average Precision), PR-AUC, Brier score, calibration ECE, confusion matrix at threshold 0.40

### Multi-task framing

The copilot supports four prediction tasks. Benchmark all four:

| Task | Target | Algorithm | Primary metric |
|---|---|---|---|
| Delay binary | `is_delayed` | XGBoost classifier | ROC-AUC |
| Delay regression | `delay_minutes` | XGBoost regressor (log1p target) | MAE, RMSE |
| Delay category | `delay_category` | LightGBM ordinal classifier | Weighted F1 |
| Root cause | `delay_root_cause` | LightGBM multi-class | Macro F1 |

Train separate models for each task. Do not attempt multi-task learning without evidence it helps on this dataset size.

---

## Week 2, Day 7 — Hyperparameter Tuning

**Deliverable:** `ml/notebooks/03_tuning.ipynb` + MLflow runs under `mpc/delay_prediction/tuning`

Use **Optuna** for Bayesian hyperparameter optimisation. Limit to the two primary candidates (XGBoost and LightGBM).

### Search spaces

**XGBoost:**
```
n_estimators:     [200, 1000]  (int, log)
max_depth:        [3, 8]       (int)
learning_rate:    [0.01, 0.30] (float, log)
subsample:        [0.6, 1.0]   (float)
colsample_bytree: [0.6, 1.0]   (float)
min_child_weight: [1, 10]      (int)
reg_alpha:        [1e-8, 1.0]  (float, log)
reg_lambda:       [1e-8, 1.0]  (float, log)
```

**LightGBM:**
```
n_estimators:     [200, 1000]  (int, log)
num_leaves:       [20, 200]    (int)
learning_rate:    [0.01, 0.30] (float, log)
subsample:        [0.6, 1.0]   (float)
colsample_bytree: [0.6, 1.0]   (float)
min_child_samples:[5, 50]      (int)
reg_alpha:        [1e-8, 1.0]  (float, log)
reg_lambda:       [1e-8, 1.0]  (float, log)
```

**Optuna settings:** 100 trials, pruning enabled (MedianPruner, 5-trial warmup), objective = val AUC, direction = maximize. Use `TimeSeriesSplit(5)` inside the objective function.

**Accepted outcome:** Select the model+hyperparameters with the highest val AUC. Expected post-tuning AUC: 0.85–0.91.

---

## Week 2, Day 8 — SHAP Explainability

**Deliverable:** `backend/app/services/ml/explainability.py`

SHAP is not optional — it is a core product feature. The copilot must explain every prediction to the production planner in plain English. This service computes SHAP values for individual orders and global feature importance.

### Implementation requirements

```python
class DelayExplainer:
    def explain_order(self, order_features: dict) -> ExplanationResult:
        # Returns: predicted_prob, top_3_positive_factors, top_3_mitigating_factors, shap_values
    
    def global_importance(self) -> pd.DataFrame:
        # Returns: feature_name, mean_abs_shap ranked table

class ExplanationResult:
    predicted_delay_probability: float
    predicted_delay_minutes: Optional[float]
    predicted_root_cause: str
    confidence: str          # 'high' | 'medium' | 'low'
    top_risk_factors: List[FactorExplanation]
    mitigating_factors: List[FactorExplanation]
    narrative: str           # Plain English, generated from SHAP + templates

class FactorExplanation:
    feature_name: str
    human_label: str         # e.g. "Machine Queue Depth"
    value: Any
    shap_contribution: float # positive = increases delay risk
    direction: str           # 'increases_risk' | 'reduces_risk'
    magnitude: str           # 'high' | 'medium' | 'low'
```

### Human-readable feature labels (mapping)

| Feature name | Human label |
|---|---|
| `work_center_queue_depth_at_release` | Machine queue congestion |
| `release_lag_hours` | Late material release |
| `schedule_tightness_ratio` | Schedule tightness |
| `material_availability_at_release` | Material availability |
| `priority_encoded` | Order priority |
| `machine_oee_30d` | Machine performance |
| `product_delay_rate_90d` | Product delay history |
| `operator_skill_tier_encoded` | Operator experience level |
| `days_since_last_planned_maintenance` | Maintenance recency |
| `changeover_complexity_score` | Changeover complexity |

### Narrative template

```
"This order has a {confidence} risk of delay ({prob:.0%} probability).

The main risk factors are: {top_factor_1} ({impact_1}), {top_factor_2} ({impact_2}).

{mitigation_sentence if mitigating_factors else "No significant risk mitigators were identified."}

If delayed, the predicted root cause is {root_cause} with an estimated delay of {delay_est}."
```

---

## Week 2, Day 9 — ML Service API

**Deliverable:** `backend/app/services/ml/service.py` + `backend/app/api/routes/predictions.py`

### API endpoints

```
POST   /api/v1/predictions/delay
       Input:  OrderFeatures (37 fields)
       Output: DelayPrediction {probability, category, root_cause, minutes_est, explanation}

POST   /api/v1/predictions/delay/batch
       Input:  List[OrderFeatures] (up to 100 orders)
       Output: List[DelayPrediction]

GET    /api/v1/models/current
       Output: ModelInfo {model_id, version, val_auc, trained_at, feature_count}

GET    /api/v1/models/feature-importance
       Output: List[{feature, importance, rank}]
```

### Service design

```python
class DelayPredictionService:
    def __init__(self, model_registry: MLflowModelRegistry):
        self.model = model_registry.load_production_model("delay_classifier")
        self.pipeline = model_registry.load_production_pipeline("feature_pipeline")
        self.explainer = DelayExplainer(self.model, self.pipeline)
    
    def predict(self, order: OrderFeatures) -> DelayPrediction:
        X = self.pipeline.transform(order.to_dataframe())
        prob = self.model.predict_proba(X)[0, 1]
        explanation = self.explainer.explain_order(order.dict())
        return DelayPrediction(probability=prob, explanation=explanation, ...)
```

**Model loading:** Use MLflow's `mlflow.sklearn.load_model()`. Load at service startup, not per-request. Cache in memory; hot-reload on model version bump via polling or webhook.

**Latency target:** <50ms p95 for single-order prediction. <500ms for batch of 100.

---

## Week 2, Day 10 — MLflow Integration

**Deliverable:** `backend/app/services/ml/registry.py` + MLflow tracking server config

### Experiment structure

```
mlflow/
  experiments/
    mpc/delay_prediction/
      baseline/        ← Week 1, Day 5 runs
      tuning/          ← Week 2, Day 7 runs
      production_v1/   ← Final selected model
    mpc/root_cause/
      baseline/
      production_v1/
```

### Model promotion protocol

1. All models trained → logged to MLflow with full metrics
2. Champion model selected by val AUC (delay_classifier) and macro-F1 (root_cause_classifier)
3. Champion evaluated **once** on test set → final AUC logged as `test_auc` tag
4. Model promoted to `production` stage in MLflow Model Registry
5. FastAPI service reloads production model on next health check

### Artifacts logged per run (mandatory)

- Fitted `Pipeline` object (as `pipeline.pkl`)
- Fitted model (as `model.pkl`)
- `classification_report.txt` (train and val)
- `confusion_matrix.png`
- `shap_beeswarm.png` (top 20 features)
- `calibration_curve.png`
- Feature importance table (as `feature_importance.csv`)
- All hyperparameters (as MLflow params)

---

## Week 3, Days 11–15 — Product Layer

### Day 11 — Delay prediction endpoint + business logic

Integrate the ML service into the production order workflow. When an order is released in the system, automatically trigger a delay prediction. Store the prediction alongside the order record. Flag orders with `probability > 0.65` as "high risk" in the database.

Business rules to implement:
- Orders predicted high-risk are surfaced to the planner before start
- If `root_cause = material_unavailability` predicted: auto-trigger inventory check
- If `root_cause = machine_breakdown` predicted: surface machine maintenance status

### Day 12 — Root cause explanation engine

The explanation engine translates ML output into production planner language. Build the `narrative` generation function that produces actionable sentences from SHAP values + template library.

Required narratives per root cause:
- `setup_overrun`: "Consider scheduling a pre-run changeover review. Setup history on [machine] shows [overrun_rate]% overrun rate in the last 90 days."
- `machine_breakdown`: "Machine [id] is [days_since_pm] days since last maintenance (interval: [interval] days). Consider expediting PM before this order."
- `material_unavailability`: "[shortage_count] components are unconfirmed. Verify [component_list] availability before releasing."
- `quality_failure_rework`: "FPY for [product] on [machine] is [fpy]% over 90 days. Consider pre-run quality setup inspection."

### Day 13 — Streamlit Dashboard V1

**Deliverable:** `frontend/app.py`

Three pages:

**Page 1 — Live Order Risk Board**
Table of today's orders sorted by delay probability (desc). Colour-coded risk: red >65%, amber 40–65%, green <40%. Clicking an order shows the SHAP explanation panel.

**Page 2 — Prediction Detail**
Single-order view: probability gauge, top risk factors bar chart, root cause distribution, narrative explanation, recommended actions.

**Page 3 — Performance Metrics**
Model performance dashboard: rolling 30-day precision/recall on completed orders where actual outcome is known. Feature importance chart. Dataset drift indicators (PSI for top 10 features).

### Day 14 — n8n Workflow Automation

Three workflows:

1. **Order released → predict** — Triggered by webhook from FastAPI when an order is released. Calls prediction endpoint. If high risk, posts to Slack with explanation.

2. **Daily digest** — Runs at 06:00. Fetches all orders starting today. Aggregates predicted risk. Sends HTML summary email with orders requiring attention.

3. **Outcome feedback loop** — Runs at 23:00. For all orders completed today, records whether prediction was correct. Updates precision/recall dashboard. Detects if model accuracy has degraded below threshold (triggers retraining alert).

### Day 15 — End-to-End Integration Test

Full smoke test of the complete system:
1. Run factory simulation (540 days, seed=99 for test data)
2. Train model (30 min)
3. Start Docker Compose stack
4. POST 10 sample orders to prediction API
5. Verify response schema, SHAP values, and narrative text
6. Trigger n8n digest workflow manually
7. Verify Streamlit dashboard renders

---

## Week 4 — Portfolio Polish

### Day 16 — Docker Compose Full Stack

```yaml
services:
  postgres:     # Production order database
  redis:        # Prediction cache (TTL: 10 minutes)
  mlflow:       # Model registry + experiment tracking
  fastapi:      # ML service + REST API
  streamlit:    # Dashboard
  n8n:          # Workflow automation
  ollama:       # Local LLM for explanation narrative
```

All services run on a single machine. Target: `docker compose up` completes cold start in under 3 minutes on a 2021 MacBook Pro (M1).

### Day 17 — Performance Benchmarking

Document and test:
- Single prediction latency: target p50<20ms, p99<100ms
- Batch prediction (100 orders): target p95<500ms
- Dashboard initial load: <3 seconds
- Simulation (540 days): document runtime (expected ~30 seconds)
- MLflow experiment page load: <5 seconds

Memory footprint at steady state: target <4GB RAM total for all services.

### Day 18 — Demo Data + Walkthrough

Prepare a scripted 8-minute demo:
1. Show the simulation running and calibration report (1 min)
2. Show MLflow experiment tracking — 5 model comparison (2 min)
3. Show the SHAP explanation for a delayed order (2 min)
4. Show the dashboard live order risk board (1 min)
5. Trigger the n8n Slack alert (1 min)
6. Show the architecture diagram and GitHub README (1 min)

Prepare two demo order scenarios:
- **Scenario A — High risk:** tight schedule, shortage, high-utilisation machine, junior operator
- **Scenario B — Low risk:** ample lead time, material confirmed, senior operator, well-maintained machine

### Days 19–20 — Documentation + Submission

GitHub README structure:
```
README.md
├── What this is (product summary)
├── Why it's hard (data strategy decision)
├── Architecture (diagram + 6-layer description)
├── ML approach (feature engineering + model selection)
├── How to run (docker compose up + demo commands)
├── Results (AUC, latency, calibration)
└── What I'd do next (honest limitations + v2 ideas)
```

Portfolio submission checklist:
- [ ] All 4 technical documents (Docs 01–04)
- [ ] Simulation code with calibration report
- [ ] EDA notebook (narrated, clean outputs)
- [ ] Benchmarking notebook with 5-model comparison
- [ ] MLflow experiments with logged artifacts
- [ ] FastAPI service with /docs (auto-generated Swagger)
- [ ] Streamlit dashboard running locally
- [ ] Docker Compose full stack
- [ ] 8-minute demo recording

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ML model underperforms (<0.75 AUC) | Low | High | Check for target leakage; try adding engineered interaction features; verify train/val split is truly temporal |
| Streamlit dashboard too slow | Medium | Medium | Cache ML predictions in Redis; pre-compute daily batch at 06:00 |
| n8n workflow too complex to debug | Medium | Low | Skip advanced workflows; implement as simple cron + Python script if needed |
| SHAP computation too slow (<200ms) | Medium | Medium | Use `TreeExplainer` (fast for tree models), cache SHAP values for batch |
| Docker compose memory overrun | Low | Medium | Reduce Ollama model size; disable Ollama if RAM <16GB, use OpenRouter instead |
| Portfolio looks like tutorial, not product | Medium | High | Never show raw notebooks as portfolio; always show the product UX (dashboard + explanation) first |

---

## Definition of Done

The portfolio project is complete when a technical reviewer can:

1. Clone the repository and run `docker compose up` without configuration
2. See the Streamlit dashboard with live delay predictions
3. POST an order to `/api/v1/predictions/delay` and receive a SHAP explanation
4. Open MLflow and see all 5 model comparison runs with logged artifacts
5. Read the README and understand the complete technical rationale in 10 minutes
6. Find the simulation, EDA, and tuning notebooks in `ml/notebooks/`

---

*Document 04 of 04 — Manufacturing Process Copilot Technical Series*
