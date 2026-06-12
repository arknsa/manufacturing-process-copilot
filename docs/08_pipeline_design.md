# Manufacturing Process Copilot
## Design Specification: `ml/src/mpc_ml/features/pipeline.py`

**Status:** Authoritative — implement exactly as specified  
**Phase:** 1, Day 3  
**Depends on:** `constants.py` (Doc 06), `transformers.py` (Doc 07)  
**Consumed by:** `ml/scripts/train.py`, `ml/scripts/evaluate.py`, `backend/app/services/ml/`

---

## 1. Module Purpose

`pipeline.py` has one job: assemble the preprocessing pipeline and expose it as `build_pipeline()`. It wires together the transformers from `transformers.py` and the preprocessing groups from `constants.py` into a configured, validated sklearn `Pipeline`.

This module does **not** contain business logic, training loops, model definitions, or evaluation code. It is a factory and a configuration. All logic lives in the classes it assembles.

**The critical design decision:** `build_pipeline()` returns a 3-step preprocessing pipeline **without a model**. The model is added by the training script via a nested `Pipeline`. This separation is mandatory for SHAP compatibility and multi-task training. Both patterns are explicitly specified in this document.

---

## 2. Dependency Diagram

```
constants.py
├── FEATURE_COLS ─────────────────────────► build_pipeline() validation assertion
├── INTERACTION_FEATURE_NAMES ────────────► build_pipeline() validation assertion
├── LOG_FEATURES ─────────────────────────► ColumnTransformer log_scale branch
├── SCALE_FEATURES ───────────────────────► ColumnTransformer scale_only branch
├── BINARY_FEATURES ──────────────────────► ColumnTransformer binary branch
├── ORDINAL_FEATURES ─────────────────────► ColumnTransformer ordinal branch
├── PASSTHROUGH_FEATURES ─────────────────► ColumnTransformer passthrough_counts branch
│                                            (minus ZERO_VARIANCE_FEATURES)
└── ZERO_VARIANCE_FEATURES ───────────────► ColumnTransformer zero_variance branch

transformers.py
├── ColumnSelector ───────────────────────► Pipeline step 1
└── InteractionFeatureAdder ──────────────► Pipeline step 2

sklearn.pipeline.Pipeline ───────────────► outermost container
sklearn.compose.ColumnTransformer ───────► Pipeline step 3
sklearn.pipeline.Pipeline (nested) ──────► log_scale sub-pipeline
sklearn.preprocessing.StandardScaler ────► log_scale sub-pipeline step 2,
│                                          scale_only transformer
sklearn.preprocessing.FunctionTransformer► log_scale sub-pipeline step 1
numpy.log1p ─────────────────────────────► FunctionTransformer function argument

pipeline.py
└── exports: build_pipeline() → sklearn.pipeline.Pipeline (3 steps, no model)
```

---

## 3. Public API

### `build_pipeline() → Pipeline`

The sole public function of this module. Returns an unfitted sklearn `Pipeline` containing exactly 3 named steps. Always unfitted — calling `fit()` is the responsibility of the training script, never of this factory.

**Parameters:** None. This function has no arguments. All configuration comes from `constants.py`. If preprocessing behaviour needs to change (e.g., a feature moves from SCALE to LOG), the change is made in `constants.py` and `build_pipeline()` automatically reflects it.

**Returns:** `sklearn.pipeline.Pipeline` with named steps:
- `'column_selector'` → `ColumnSelector()`
- `'interaction_adder'` → `InteractionFeatureAdder()`
- `'column_transformer'` → `ColumnTransformer(...)` (configured with all 6 branches)

**Raises at call time (before any data is seen):**
- `ValueError`: if the union of all feature group lists does not exactly equal `FEATURE_COLS + INTERACTION_FEATURE_NAMES` — meaning a feature is unassigned or assigned twice

**Does not raise:** `build_pipeline()` never raises on data-dependent conditions. All data validation is in `ColumnSelector` and `InteractionFeatureAdder`.

### `get_feature_names() → List[str]`

Returns the 41-element list of feature names in the exact order they appear in the ColumnTransformer output. This is the authoritative source for naming SHAP values after model prediction. Uses the ColumnTransformer's static column assignment order — no fitting required.

---

## 4. Build-Time Validation

The first executable statement inside `build_pipeline()` (before constructing any sklearn object) is the **coverage assertion**. It verifies:

**Condition A: completeness** — every feature in `FEATURE_COLS + INTERACTION_FEATURE_NAMES` appears in at least one group list.

**Condition B: exclusivity** — no feature appears in more than one group list.

**Condition C: total count** — the union of all group lists has exactly 41 members.

If any condition fails, `build_pipeline()` raises `ValueError` with a message that identifies the offending features. This catches configuration errors the instant they are introduced — adding a feature to `FEATURE_COLS` without assigning it to a preprocessing group will fail at the first training attempt, not silently corrupt the model.

The interaction features have a split assignment:
- `lag_as_pct_of_window` → `LOG_FEATURES` branch (included alongside base `LOG_FEATURES`)
- `tightness_x_queue` → `SCALE_FEATURES` branch
- `oee_x_maintenance_ratio` → `SCALE_FEATURES` branch
- `log_experience_x_concurrent` → `ZERO_VARIANCE_FEATURES` branch

These assignments are not in `constants.py` (they are not base feature groups); they are computed inside `build_pipeline()` and are part of the ColumnTransformer configuration. The validation assertion ensures all 4 interaction features are accounted for.

---

## 5. Three-Step Preprocessing Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Pipeline('column_selector', 'interaction_adder',                │
│           'column_transformer')                                  │
│                                                                  │
│  Step 1: ColumnSelector                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Input: arbitrary DataFrame                                 │  │
│  │ → validates 37 FEATURE_COLS, rejects TARGET_COLS           │  │
│  │ → fills rolling NaN with cold_start_defaults_              │  │
│  │ → asserts zero-variance columns                            │  │
│  │ Output: clean 37-column DataFrame                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  Step 2: InteractionFeatureAdder                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Input: 37-column DataFrame                                 │  │
│  │ → computes 4 derived interaction features                  │  │
│  │ → appends to right side of DataFrame                       │  │
│  │ Output: 41-column DataFrame                                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  Step 3: ColumnTransformer (6 named branches)                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Input: 41-column DataFrame                                 │  │
│  │ → applies group-specific transforms per column             │  │
│  │ → concatenates branch outputs in branch order              │  │
│  │ Output: 41-column DataFrame (or array, see §8)             │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. ColumnTransformer Architecture

The `ColumnTransformer` is configured with 6 named transformers and `remainder='drop'`. Each named transformer handles one preprocessing group and receives only the columns assigned to it, by name. The output of all 6 transformers is concatenated horizontally in the order they are listed.

### Full Branch Specification

**Branch 1: `log_scale` — 7 features**

A nested `Pipeline` object (not a direct transformer) applied to the 7 log-transformed features.

Sub-pipeline steps:
1. `'log'`: `FunctionTransformer(np.log1p, validate=False)` — applies log1p element-wise
2. `'scaler'`: `StandardScaler()` — centres and scales the log-transformed values

The `validate=False` parameter on `FunctionTransformer` is mandatory. With `validate=True` (default in some sklearn versions), the transformer attempts to convert input to a 2D numpy array before applying the function, which can interfere with pandas DataFrame input and discard column names.

Columns:
- Base: `planned_lead_time_hours`, `release_lag_hours`, `quantity`, `estimated_total_hours`, `machine_unplanned_downtime_hours_30d`, `operator_experience_months`
- Interaction: `lag_as_pct_of_window`

The `StandardScaler` inside this sub-pipeline fits on the log-transformed values (not the raw values). After training, `pipeline['column_transformer'].named_transformers_['log_scale']['scaler'].mean_` contains the mean of the log-transformed training data — not the mean of the raw data. This is important for the SHAP inverse-transform context (see §8).

Output positions [0, 6].

---

**Branch 2: `scale_only` — 17 features**

`StandardScaler()` — a single transformer, no sub-pipeline.

Columns:
- Base: `schedule_tightness_ratio`, `product_complexity_score`, `machine_utilization_at_release`, `machine_oee_30d`, `days_since_last_planned_maintenance`, `changeover_complexity_score`, `hours_into_shift_at_start`, `product_delay_rate_90d`, `machine_delay_rate_90d`, `operator_delay_rate_90d`, `product_x_machine_delay_rate_90d`, `product_first_pass_yield_90d`, `machine_setup_overrun_rate_90d`, `shift_delay_rate_30d`, `planned_start_hour`
- Interactions: `tightness_x_queue`, `oee_x_maintenance_ratio`

The `StandardScaler` for this branch fits on the raw (non-log) values. The fitted `mean_` and `scale_` arrays have 17 elements corresponding to the 17 columns in this branch, in the order listed.

Output positions [7, 23].

---

**Branch 3: `binary` — 8 features**

`'passthrough'` — the sklearn string shorthand for the passthrough transformer. No transformation applied.

Columns: `schedule_revision_count`, `is_expedited`, `is_month_end`, `is_quarter_end`, `work_center_queue_depth_at_release`, `maintenance_due_within_order_window`, `changeover_required`, `material_availability_at_release`

**Why not use `FunctionTransformer(lambda x: x)` instead of `'passthrough'`?** The `'passthrough'` string is sklearn's canonical passthrough signal. It is recognised by `ColumnTransformer.get_feature_names_out()` and produces the correct feature names in the output without any additional configuration.

Output positions [24, 31].

---

**Branch 4: `ordinal` — 4 features**

`'passthrough'` — same rationale as binary.

Columns: `priority_encoded`, `operator_skill_tier_encoded`, `shift_type_encoded`, `planned_start_day_of_week`

Output positions [32, 35].

---

**Branch 5: `passthrough_counts` — 3 features**

`'passthrough'` — same rationale.

Columns: `operation_count`, `material_bom_complexity`, `component_shortage_count`

Note: `operator_concurrent_order_count` is excluded from this branch even though it is in `PASSTHROUGH_FEATURES`. It goes to the `zero_variance` branch to be explicitly managed.

Output positions [36, 38].

---

**Branch 6: `zero_variance` — 2 features**

`'passthrough'` — same rationale.

Columns: `operator_concurrent_order_count`, `log_experience_x_concurrent`

These two columns must NEVER enter a `StandardScaler`. They are currently constant (all zeros) — `StandardScaler` would divide by a standard deviation of zero and produce `NaN` silently. This branch explicitly routes them to passthrough, preventing this failure mode.

When `operator_concurrent_order_count` and `log_experience_x_concurrent` become non-zero (future simulation change), these features should be moved to `SCALE_FEATURES` and this branch updated. The zero-variance warning in `ColumnSelector` (see Doc 07) is the notification mechanism for this change.

Output positions [39, 40].

---

### ColumnTransformer Global Settings

- `remainder='drop'`: columns not assigned to any named transformer are discarded. The build-time validation assertion ensures this never silently drops a needed feature.
- `verbose_feature_names_out=False`: output feature names are the original column names (`planned_lead_time_hours`, `release_lag_hours`, ...) without transformer-name prefixes (`log_scale__planned_lead_time_hours`). This is essential for SHAP feature name mapping.
- `n_jobs=1`: single-threaded. Parallel fitting of branches is a premature optimisation — the dataset is small and parallelism overhead exceeds the benefit.

---

## 7. Complete Output Feature Map

The ColumnTransformer's output has exactly 41 columns in this fixed order. This order is the model's feature space. It is preserved identically between training and serving because the model is always called via the fitted ColumnTransformer, which references columns by name.

```
Position  Feature Name                              Transform Applied
────────  ────────────────────────────────────────  ────────────────────────────
[  0]     planned_lead_time_hours                   log1p → StandardScaler
[  1]     release_lag_hours                         log1p → StandardScaler
[  2]     quantity                                  log1p → StandardScaler
[  3]     estimated_total_hours                     log1p → StandardScaler
[  4]     machine_unplanned_downtime_hours_30d      log1p → StandardScaler
[  5]     operator_experience_months                log1p → StandardScaler
[  6]     lag_as_pct_of_window       [INTERACTION]  log1p → StandardScaler
[  7]     schedule_tightness_ratio                  StandardScaler
[  8]     product_complexity_score                  StandardScaler
[  9]     machine_utilization_at_release            StandardScaler
[ 10]     machine_oee_30d                           StandardScaler
[ 11]     days_since_last_planned_maintenance       StandardScaler
[ 12]     changeover_complexity_score               StandardScaler
[ 13]     hours_into_shift_at_start                 StandardScaler
[ 14]     product_delay_rate_90d                    StandardScaler
[ 15]     machine_delay_rate_90d                    StandardScaler
[ 16]     operator_delay_rate_90d                   StandardScaler
[ 17]     product_x_machine_delay_rate_90d          StandardScaler
[ 18]     product_first_pass_yield_90d              StandardScaler
[ 19]     machine_setup_overrun_rate_90d            StandardScaler
[ 20]     shift_delay_rate_30d                      StandardScaler
[ 21]     planned_start_hour                        StandardScaler
[ 22]     tightness_x_queue          [INTERACTION]  StandardScaler
[ 23]     oee_x_maintenance_ratio    [INTERACTION]  StandardScaler
[ 24]     schedule_revision_count                   passthrough
[ 25]     is_expedited                              passthrough
[ 26]     is_month_end                              passthrough
[ 27]     is_quarter_end                            passthrough
[ 28]     work_center_queue_depth_at_release        passthrough
[ 29]     maintenance_due_within_order_window       passthrough
[ 30]     changeover_required                       passthrough
[ 31]     material_availability_at_release          passthrough
[ 32]     priority_encoded                          passthrough
[ 33]     operator_skill_tier_encoded               passthrough
[ 34]     shift_type_encoded                        passthrough
[ 35]     planned_start_day_of_week                 passthrough
[ 36]     operation_count                           passthrough
[ 37]     material_bom_complexity                   passthrough
[ 38]     component_shortage_count                  passthrough
[ 39]     operator_concurrent_order_count           passthrough (ZERO_VAR)
[ 40]     log_experience_x_concurrent [INTERACTION] passthrough (ZERO_VAR)
```

---

## 8. Training Workflow

The training workflow is **not** part of `pipeline.py`. It is implemented in `ml/scripts/train.py` and `ml/notebooks/02_baseline.ipynb`. However, `pipeline.py` must be designed to support it correctly. This section specifies the expected usage contract.

### Pattern: Nested Pipeline

The preprocessing pipeline from `build_pipeline()` is wrapped in an outer `Pipeline` alongside the model. This is the recommended sklearn pattern when:
1. The preprocessing and model are separately logged to MLflow
2. The SHAP service needs the preprocessing steps extracted independently
3. Multiple models (one per prediction task) share the same preprocessing

```
full_pipeline = Pipeline([
    ('preprocessor', build_pipeline()),    ← from pipeline.py
    ('model', XGBClassifier(...))          ← added by train.py
])
```

### Training Sequence

```
Step 1: Load data
  train_df = pd.read_csv('ml/data/processed/train.csv')
  X_train = train_df.drop(TARGET_COLS, axis=1)
  y_train = train_df['is_delayed']   ← (or other target depending on task)

Step 2: Build pipeline
  preprocessor = build_pipeline()         ← unfitted, from pipeline.py

Step 3: Configure model
  model = XGBClassifier(**best_params)

Step 4: Assemble full pipeline
  full_pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('model', model)
  ])

Step 5: Fit
  full_pipeline.fit(X_train, y_train)

  Internal execution during fit():
    a. preprocessor.fit_transform(X_train)
       ├─ ColumnSelector.fit(X_train) → learns cold_start_defaults_
       ├─ ColumnSelector.transform(X_train) → 37 clean columns
       ├─ InteractionFeatureAdder.fit(X_37) → no-op
       ├─ InteractionFeatureAdder.transform(X_37) → 41 columns
       ├─ ColumnTransformer.fit_transform(X_41)
       │   ├─ log_scale.fit_transform(X_log_cols)
       │   │   ├─ FunctionTransformer.fit_transform → log1p applied
       │   │   └─ StandardScaler.fit_transform → mean_/scale_ learned
       │   ├─ scale_only.fit_transform(X_scale_cols) → mean_/scale_ learned
       │   ├─ binary passthrough
       │   ├─ ordinal passthrough
       │   ├─ passthrough_counts passthrough
       │   └─ zero_variance passthrough
       └─ returns X_model (n_train × 41)
    b. model.fit(X_model, y_train)

Step 6: Evaluate on validation set
  X_val = val_df.drop(TARGET_COLS, axis=1)
  y_val_pred = full_pipeline.predict_proba(X_val)[:, 1]
  val_auc = roc_auc_score(val_df['is_delayed'], y_val_pred)
```

### Fit-Time State Created

After `full_pipeline.fit(X_train, y_train)`, the following fitted state is created and stored inside the pipeline object:

| Component | Fitted state | Purpose |
|---|---|---|
| `ColumnSelector` | `cold_start_defaults_` (7 means) | Fill NaN at inference time |
| `ColumnSelector` | `zero_variance_observed_values_` | Assert at inference time |
| `ColumnSelector` | `feature_names_in_`, `n_features_in_` | sklearn standard |
| `InteractionFeatureAdder` | `is_fitted_` = True | Fitted check |
| ColumnTransformer `log_scale.scaler` | `mean_` (7 values), `scale_` (7 values) | Log-scale normalisation |
| ColumnTransformer `scale_only` | `mean_` (17 values), `scale_` (17 values) | Scale normalisation |

This state travels with the serialised pipeline object. When loaded from MLflow, these values are identical to what was computed during training.

---

## 9. Inference Workflow

Inference uses the fitted `full_pipeline.predict_proba()`. The pipeline calls `preprocessor.transform()` (not `fit_transform()`) and then `model.predict_proba()`.

```
Input: X_new (37 columns, 1 or N rows)

full_pipeline.predict_proba(X_new)

Internal execution:
  a. preprocessor.transform(X_new)
     ├─ ColumnSelector.transform(X_new)
     │   ├─ Check is_fitted_
     │   ├─ Reject TARGET_COLS
     │   ├─ Validate FEATURE_COLS exist
     │   ├─ Coerce dtypes (API JSON coercion case)
     │   ├─ Fill rolling-feature NaN with cold_start_defaults_
     │   ├─ Assert zero NaN remaining
     │   ├─ Assert zero-variance columns
     │   └─ Return 37-column DataFrame
     ├─ InteractionFeatureAdder.transform(X_37)
     │   ├─ Check is_fitted_
     │   ├─ Compute lag_as_pct_of_window (with guards)
     │   ├─ Compute tightness_x_queue
     │   ├─ Compute log_experience_x_concurrent
     │   ├─ Compute oee_x_maintenance_ratio (with guards)
     │   ├─ Assert no NaN or inf in output
     │   └─ Return 41-column DataFrame
     └─ ColumnTransformer.transform(X_41)
         ├─ log_scale: log1p → scale using stored mean_/scale_
         ├─ scale_only: scale using stored mean_/scale_
         ├─ binary: passthrough
         ├─ ordinal: passthrough
         ├─ passthrough_counts: passthrough
         ├─ zero_variance: passthrough
         └─ Return X_model (N × 41)
  b. model.predict_proba(X_model) → [[p_on_time, p_delayed], ...]
  c. Return probabilities
```

**Critical invariant:** The ColumnTransformer calls `transform()` (not `fit_transform()`) at inference time. It uses the `mean_` and `scale_` values learned during training. These are never recomputed from the inference data. This is the primary training-serving consistency guarantee provided by sklearn's Pipeline contract.

---

## 10. SHAP Compatibility Strategy

SHAP values require a model and the same feature representation the model was trained on. The `TreeExplainer` for XGBoost/LightGBM computes exact SHAP values in polynomial time.

### The SHAP Access Pattern

```
Serving:
  full_pipeline = mlflow.sklearn.load_model("models:/delay_classifier/Production")
  preprocessor = full_pipeline.named_steps['preprocessor']

Single order explanation:
  X_raw = order_features_dict_to_dataframe(order)      ← 37 cols
  X_model = preprocessor.transform(X_raw)              ← 41 cols, transformed
  feature_names = get_feature_names()                  ← 41 names in output order
  
  explainer = shap.TreeExplainer(full_pipeline.named_steps['model'])
  shap_values = explainer(X_model)                     ← shap.Explanation object
  shap_dict = dict(zip(feature_names, shap_values.values[0]))
```

### Feature Name Preservation

`get_feature_names()` (the module-level function in `pipeline.py`) returns the 41 feature names in their ColumnTransformer output order. This is the mapping between SHAP value positions [0..40] and human-readable feature names.

This mapping is deterministic from the constants — it does not require a fitted pipeline. It is computed once at module load time and returned by `get_feature_names()`.

### Why SHAP Values Are in Transformed Space

SHAP values from `TreeExplainer` reflect the model's feature space — the space after preprocessing. Features [0..6] (LOG group) have SHAP values in the log-standardised space, not the original hour/count space. Features [24..31] (BINARY group, passthrough) have SHAP values in the original {0, 1} space.

**This is correct and intended.** The `explainability.py` service in the backend (Doc 05) maps SHAP values to human-readable explanations using the feature labels from the Feature Dictionary (Doc 03). The SHAP *magnitude* (abs value) determines importance; the feature *value* from the original data provides context. The explanation reads: "material_availability_at_release = 0 (shortage) contributed +0.34 to delay probability" — this uses the original feature value (0 or 1) alongside the SHAP value, not the transformed value.

### SHAP Background Dataset

`shap.TreeExplainer(model, data=background_sample)` accepts a background dataset for computing expected SHAP values. The background should be a sample of the **transformed** training data:

```
background = X_train_transformed_sample  ← N×41 array, already transformed
explainer = shap.TreeExplainer(model, data=background)
```

Store the background sample (100–200 rows of transformed training data) as an MLflow artifact alongside the model. The `registry.py` in the backend loads this background sample when initialising the `TreeExplainer`.

---

## 11. MLflow Serialization Strategy

When a training run completes, `mlflow_utils.py` logs the following artifacts. The design is explicit about what is logged, why, and how it is loaded at serve time.

### Artifact: `pipeline` (the full pipeline)

- **What:** `full_pipeline` = `Pipeline([('preprocessor', ...), ('model', ...)])`
- **How logged:** `mlflow.sklearn.log_model(full_pipeline, artifact_path='pipeline')`
- **Input signature:** logged with `mlflow.models.infer_signature(X_train, full_pipeline.predict_proba(X_train))`
- **Purpose:** Single artifact for batch prediction, A/B testing, and any use case that needs a single `.predict()` call
- **Loaded by:** `backend/app/services/ml/registry.py` as the primary serving artifact

### Artifact: `preprocessing_pipeline` (preprocessor only)

- **What:** `preprocessor` = `build_pipeline()` after fitting = `full_pipeline.named_steps['preprocessor']`
- **How logged:** `mlflow.sklearn.log_model(preprocessor, artifact_path='preprocessing_pipeline')`
- **Purpose:** SHAP computation requires the preprocessing pipeline separately from the model
- **Loaded by:** `backend/app/services/ml/explainability.py` for the SHAP service

### Artifact: `feature_names.json`

- **What:** `get_feature_names()` output — the 41-element list in ColumnTransformer output order
- **How logged:** `mlflow.log_dict({'feature_names': get_feature_names()}, 'feature_names.json')`
- **Purpose:** Associates SHAP value positions [0..40] with human-readable names; serves as a schema contract between training and serving
- **Loaded by:** `backend/app/services/ml/explainability.py`

### Artifact: `cold_start_defaults.json`

- **What:** `preprocessor.named_steps['column_selector'].cold_start_defaults_`
- **How logged:** `mlflow.log_dict(cold_start_defaults, 'cold_start_defaults.json')`
- **Purpose:** Documents the population means used for cold-start filling; enables debugging of cold-start predictions
- **Loaded by:** Not loaded at serve time (already embedded in the serialised `preprocessing_pipeline`); used for audit and debugging

### Artifact: `shap_background_sample.npy`

- **What:** 200 rows of transformed training data (X_model shape: 200 × 41)
- **How logged:** `mlflow.log_artifact('shap_background_sample.npy')`
- **Purpose:** Background dataset for `shap.TreeExplainer(model, data=background)`. Provides stable expected SHAP values (E[f(X)] baseline) that match the training distribution
- **Loaded by:** `backend/app/services/ml/explainability.py` at startup

### Artifact: `metrics.json`

- **What:** Train AUC, val AUC, precision at 80% recall, F1, calibration ECE
- **How logged:** `mlflow.log_metrics(metrics_dict)` (MLflow's native metric logging, not `log_artifact`)
- **Purpose:** Queryable for champion selection
- **Loaded by:** `backend/app/db/models/prediction.py` → `ml_model_registry` table

### Serialization Format Choice

`mlflow.sklearn.log_model()` serialises using `pickle` by default (or `cloudpickle` for more complex objects). The full pipeline serialises to approximately 5–20 MB depending on the XGBoost/LightGBM model size. This is the correct format for sklearn pipelines — no alternatives needed.

**Version pinning requirement:** The `ml/pyproject.toml` must pin exact versions of `scikit-learn`, `xgboost`, `lightgbm`, and `numpy`. Deserialized pipelines must be loaded by the same library versions used during training. `mlflow.sklearn.log_model()` records these as `requirements.txt` in the artifact — the backend's `pyproject.toml` must match.

---

## 12. Training-Serving Consistency Guarantees

### Guarantee 1: Fitted State Travels with the Artifact

All fit-time state (StandardScaler `mean_`/`scale_`, ColumnSelector `cold_start_defaults_`, InteractionFeatureAdder `is_fitted_`) is serialised into the `preprocessing_pipeline` artifact. The serving container loads this artifact, which includes the same state that was computed on the training data. **Nothing is recomputed from production data at serve time.**

### Guarantee 2: Column Selection by Name, Not Position

The `ColumnTransformer` selects features from the input DataFrame **by column name**, not by position. An inference request with columns in a different order than training will still produce the correct output. This is guaranteed by pandas column selection: `X[['feature_a', 'feature_b']]` always returns feature_a then feature_b regardless of how they appeared in X.

### Guarantee 3: Cold-Start Fill Values Are Fixed at Train Time

`ColumnSelector.fit()` learns `cold_start_defaults_` from the training set and stores them. Every subsequent `transform()` call (validation, test, production) uses these stored values. The population means in `constants.COLD_START_DEFAULTS` are documentation only — they are never used at inference time once the pipeline is fitted.

### Guarantee 4: Interaction Feature Formulas Are Stateless Constants

`InteractionFeatureAdder` is stateless — its formulas are hard-coded in the class body and do not depend on any training data. The same formula executed at training time executes identically at serving time. The only configurable parameters (`lag_clip_upper`, `oee_maintenance_scale`) are constructor arguments stored in the fitted object.

### Guarantee 5: Output Column Order Is Fixed

`get_feature_names()` returns the same 41-element list every time it is called, regardless of fitting state. This list determines the position-to-name mapping for SHAP values. Since the ColumnTransformer outputs branches in the order they are declared, and the branch order is fixed in `build_pipeline()`, the position mapping is stable across all pipeline versions that use the same branch order.

### Guarantee 6: The Model Input Space Never Changes Without Detection

If a new feature is added to `FEATURE_COLS` (requiring a simulation rerun) or an interaction feature formula changes, `build_pipeline()` must be called again. The new `build_pipeline()` produces a pipeline with a different input space (different `FEATURE_COLS`, different interactions), which requires a new `full_pipeline.fit()`. MLflow tags the run with the pipeline version. Loading a pipeline for a model trained on 37 features and calling it on a 38-feature DataFrame will either fail (if the new feature is in `FEATURE_COLS`) or silently use cold-start defaults (if it is a rolling feature). The former is caught by `ColumnSelector`. The latter is the expected cold-start behaviour.

---

## 13. Execution Diagram

### Training Execution

```
train.py
    │
    ├─► build_pipeline() ──► Pipeline([column_selector, interaction_adder, column_transformer])
    │
    ├─► full_pipeline = Pipeline([('preprocessor', build_pipeline()), ('model', model)])
    │
    └─► full_pipeline.fit(X_train, y_train)
          │
          ├─ ColumnSelector.fit(X_train)
          │     learn: cold_start_defaults_ = {rolling_feature: train_mean, ...}
          │     learn: zero_variance_observed_values_ = {zero_var_col: 0.0, ...}
          │     learn: feature_names_in_ = FEATURE_COLS
          │     CHECK: no target columns
          │     CHECK: all feature columns present
          │
          ├─ ColumnSelector.transform(X_train)
          │     FILL:   rolling NaN with cold_start_defaults_
          │     ASSERT: zero NaN remaining
          │     SELECT: X[FEATURE_COLS] in canonical order
          │     RETURN: 37-column DataFrame
          │
          ├─ InteractionFeatureAdder.fit(X_37) → no-op
          │
          ├─ InteractionFeatureAdder.transform(X_37)
          │     COMPUTE: 4 interaction features with guards
          │     CONCAT:  append 4 columns to right
          │     ASSERT:  shape (n_train, 41), zero NaN
          │     RETURN:  41-column DataFrame
          │
          ├─ ColumnTransformer.fit_transform(X_41)
          │     log_scale branch:
          │       fit: FunctionTransformer (stateless)
          │       fit: StandardScaler → learn mean_, scale_ for 7 features
          │       transform: log1p → standardise
          │     scale_only branch:
          │       fit: StandardScaler → learn mean_, scale_ for 17 features
          │       transform: standardise
          │     binary, ordinal, passthrough_counts, zero_variance:
          │       fit: no-op (passthrough has no fit state)
          │       transform: identity
          │     CONCAT: 6 branch outputs horizontally
          │     RETURN: (n_train, 41) matrix in output order [0..40]
          │
          └─ model.fit(X_model, y_train)
                  learn: XGBoost tree structure, weights
```

### Inference Execution

```
full_pipeline.predict_proba(X_new)
    │
    ├─ ColumnSelector.transform(X_new)
    │     [uses stored cold_start_defaults_, feature_names_in_]
    │     [does NOT refit, does NOT recalculate means]
    │
    ├─ InteractionFeatureAdder.transform(X_37)
    │     [same stateless formulas as training]
    │
    ├─ ColumnTransformer.transform(X_41)
    │     [uses stored mean_/scale_ from training]
    │     [does NOT recalculate from X_new]
    │
    └─ model.predict_proba(X_model)
          [return [[p_class_0, p_class_1], ...]]
```

### SHAP Execution

```
preprocessor = full_pipeline.named_steps['preprocessor']
model = full_pipeline.named_steps['model']
feature_names = get_feature_names()               ← 41 names from pipeline.py

X_model = preprocessor.transform(X_raw)          ← (1, 41) matrix, same as inference
explainer = shap.TreeExplainer(model, data=background_sample)
shap_result = explainer(X_model)                 ← shap.Explanation

# shap_result.values shape: (1, 41) — one SHAP value per feature
# Map positions to names:
shap_dict = {name: val for name, val in zip(feature_names, shap_result.values[0])}

# Top risk factors:
top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
```

---

## 14. Unit Testing Strategy

### Test Organisation

Tests for `pipeline.py` live in `ml/tests/test_pipeline.py`. They use fixtures from `ml/tests/conftest.py` (sample DataFrame, random seed). All tests are deterministic — no stochastic behaviour in the pipeline itself (StandardScaler is deterministic given the same input).

---

### Build-Time Tests (`test_build_pipeline.py`)

**PIPE-BUILD-01: `build_pipeline()` returns a Pipeline with 3 steps**
```
pipeline = build_pipeline()
Assert: isinstance(pipeline, sklearn.pipeline.Pipeline)
Assert: list(pipeline.named_steps.keys()) == ['column_selector', 'interaction_adder', 'column_transformer']
Assert: isinstance(pipeline.named_steps['column_selector'], ColumnSelector)
Assert: isinstance(pipeline.named_steps['interaction_adder'], InteractionFeatureAdder)
Assert: isinstance(pipeline.named_steps['column_transformer'], ColumnTransformer)
```

**PIPE-BUILD-02: ColumnTransformer has 6 named transformers**
```
ct = build_pipeline().named_steps['column_transformer']
Assert: set(t[0] for t in ct.transformers) == {'log_scale', 'scale_only', 'binary', 'ordinal', 'passthrough_counts', 'zero_variance'}
Assert: ct.remainder == 'drop'
Assert: ct.verbose_feature_names_out == False
```

**PIPE-BUILD-03: `log_scale` branch is a sub-pipeline with 2 steps**
```
ct = build_pipeline().named_steps['column_transformer']
log_scale_transformer = dict(ct.transformers)['log_scale'][1]  # [name, transformer, columns] tuple
Assert: isinstance(log_scale_transformer, sklearn.pipeline.Pipeline)
Assert: list(log_scale_transformer.named_steps.keys()) == ['log', 'scaler']
Assert: isinstance(log_scale_transformer.named_steps['log'], FunctionTransformer)
Assert: isinstance(log_scale_transformer.named_steps['scaler'], StandardScaler)
Assert: log_scale_transformer.named_steps['log'].func == np.log1p
```

**PIPE-BUILD-04: Column coverage is complete with no duplicates**
```
ct = build_pipeline().named_steps['column_transformer']
all_ct_cols = []
for name, transformer, cols in ct.transformers:
    all_ct_cols.extend(cols)

all_expected = list(FEATURE_COLS) + list(INTERACTION_FEATURE_NAMES)
Assert: set(all_ct_cols) == set(all_expected), "Feature coverage mismatch"
Assert: len(all_ct_cols) == len(set(all_ct_cols)), "Duplicate feature assignment"
Assert: len(all_ct_cols) == 41
```

**PIPE-BUILD-05: `get_feature_names()` returns 41 names in correct order**
```
names = get_feature_names()
Assert: len(names) == 41
Assert: names[0] == 'planned_lead_time_hours'    ← first LOG feature
Assert: names[6] == 'lag_as_pct_of_window'       ← last LOG feature (interaction)
Assert: names[24] == 'schedule_revision_count'   ← first BINARY feature
Assert: names[39] == 'operator_concurrent_order_count'  ← first ZERO_VAR
Assert: names[40] == 'log_experience_x_concurrent'      ← second ZERO_VAR
```

**PIPE-BUILD-06: build_pipeline() raises on coverage mismatch**
```
Setup: temporarily modify a constant group list to omit one feature
Assert: build_pipeline() raises ValueError
Assert: error message names the missing feature
```

---

### Shape and Schema Tests

**PIPE-SHAPE-01: fit_transform produces correct output shape**
```
pipeline = build_pipeline()
X_out = pipeline.fit_transform(X_train_37col)
Assert: X_out.shape == (n_train, 41)
```

**PIPE-SHAPE-02: transform produces correct output shape**
```
pipeline = build_pipeline()
pipeline.fit(X_train_37col)
X_val_out = pipeline.transform(X_val_37col)
Assert: X_val_out.shape == (n_val, 41)
```

**PIPE-SHAPE-03: output has zero NaN**
```
X_out = pipeline.fit_transform(X_train)
Assert: (pd.isna(X_out)).sum() == 0    (or np.isnan(X_out).sum() for arrays)
```

**PIPE-SHAPE-04: output has no infinite values**
```
X_out = pipeline.fit_transform(X_train)
Assert: np.isfinite(X_out).all()
```

**PIPE-SHAPE-05: single-row inference produces correct shape**
```
X_single = X_val.iloc[[0]]
X_out = pipeline.transform(X_single)
Assert: X_out.shape == (1, 41)
```

---

### Feature Transformation Tests

**PIPE-TRANSFORM-01: log1p is applied to LOG_FEATURES**
```
# Verify that log_scale branch actually applies log1p
# Create input with a known value in planned_lead_time_hours
X_test = X_sample.copy()
X_test.loc[0, 'planned_lead_time_hours'] = np.e - 1   # log1p(e-1) = 1.0 (before StandardScaler)
X_out = pipeline.transform(X_test)

# The StandardScaler then centres: (1.0 - mean_of_log_values) / std_of_log_values
# We can verify by checking the log_scale scaler's parameters
scaler = pipeline.named_steps['column_transformer'].named_transformers_['log_scale']['scaler']
Assert: scaler.mean_ is not None   # confirms fitting occurred
Assert: scaler.scale_ is not None
Assert: scaler.mean_.shape == (7,)  # 7 LOG features
```

**PIPE-TRANSFORM-02: BINARY_FEATURES pass through unchanged**
```
X_test = X_sample.copy()
X_test['is_expedited'] = 1
X_out = pipeline.transform(X_test)

# is_expedited is at position 25 in output order
is_expedited_pos = get_feature_names().index('is_expedited')
Assert: X_out[:, is_expedited_pos].mean() == 1.0  (if converted to array)
# Or for DataFrame output:
Assert: X_out['is_expedited'].mean() == 1.0
```

**PIPE-TRANSFORM-03: SCALE_FEATURES are mean-centred after StandardScaler**
```
# When applied to training data, StandardScaler output should be mean ≈ 0, std ≈ 1
X_out = pipeline.fit_transform(X_train)
tightness_pos = get_feature_names().index('schedule_tightness_ratio')

Assert: abs(X_out[:, tightness_pos].mean()) < 0.01   # approximately mean-centred
Assert: abs(X_out[:, tightness_pos].std() - 1.0) < 0.01  # approximately unit variance
```

**PIPE-TRANSFORM-04: ZERO_VAR columns pass through as zeros**
```
X_out = pipeline.fit_transform(X_train)
conc_pos = get_feature_names().index('operator_concurrent_order_count')
lec_pos = get_feature_names().index('log_experience_x_concurrent')

Assert: (X_out[:, conc_pos] == 0.0).all()
Assert: (X_out[:, lec_pos] == 0.0).all()
```

**PIPE-TRANSFORM-05: lag_as_pct_of_window appears in LOG output**
```
# Verify interaction feature is log-transformed (has approximately zero mean on train set)
X_out = pipeline.fit_transform(X_train)
lag_pos = get_feature_names().index('lag_as_pct_of_window')
Assert: abs(X_out[:, lag_pos].mean()) < 0.1  # approximately centered (log-standardised)
```

---

### Training-Serving Consistency Tests

**PIPE-CONSISTENCY-01: StandardScaler uses training statistics on validation data**
```
# Fit on train, transform on val with different distribution
pipeline = build_pipeline()
pipeline.fit(X_train)

# Verify scaler uses training mean, not validation mean
scaler = pipeline.named_steps['column_transformer'].named_transformers_['scale_only']
train_mean_tightness = scaler.mean_[SCALE_FEATURES.index('schedule_tightness_ratio')]

# Check manually
val_col = X_val['schedule_tightness_ratio']
scale = scaler.scale_[SCALE_FEATURES.index('schedule_tightness_ratio')]
expected_output = (val_col - train_mean_tightness) / scale

X_val_out = pipeline.transform(X_val)
tightness_pos = get_feature_names().index('schedule_tightness_ratio')
Assert: np.allclose(X_val_out[:, tightness_pos], expected_output)
```

**PIPE-CONSISTENCY-02: Serialisation round-trip produces identical output**
```
pipeline = build_pipeline()
pipeline.fit(X_train)
X_out_before = pipeline.transform(X_val)

# Serialise to bytes and deserialise
import pickle
pipeline_bytes = pickle.dumps(pipeline)
pipeline_loaded = pickle.loads(pipeline_bytes)
X_out_after = pipeline_loaded.transform(X_val)

Assert: np.allclose(X_out_before, X_out_after)
```

**PIPE-CONSISTENCY-03: MLflow log_model → load_model round-trip**
```
# This is an integration test requiring MLflow tracking server (or mlflow.set_tracking_uri('sqlite:///'))
with mlflow.start_run():
    pipeline.fit(X_train)
    mlflow.sklearn.log_model(pipeline, 'preprocessing_pipeline')
    run_id = mlflow.active_run().info.run_id

loaded = mlflow.sklearn.load_model(f'runs:/{run_id}/preprocessing_pipeline')
X_loaded_out = loaded.transform(X_val)
Assert: np.allclose(X_original_out, X_loaded_out)
```

**PIPE-CONSISTENCY-04: Cold-start defaults identical after serialisation**
```
pipeline.fit(X_train)
original_defaults = pipeline.named_steps['column_selector'].cold_start_defaults_

pipeline_bytes = pickle.dumps(pipeline)
pipeline_loaded = pickle.loads(pipeline_bytes)
loaded_defaults = pipeline_loaded.named_steps['column_selector'].cold_start_defaults_

Assert: original_defaults == loaded_defaults
```

---

### Full Pipeline Integration Tests (with model)

**PIPE-INTEG-01: Full pipeline fit → predict_proba produces valid probabilities**
```
full_pipeline = Pipeline([('preprocessor', build_pipeline()), ('model', XGBClassifier())])
full_pipeline.fit(X_train, y_train)
proba = full_pipeline.predict_proba(X_val)

Assert: proba.shape == (n_val, 2)
Assert: np.allclose(proba.sum(axis=1), 1.0)  # probabilities sum to 1
Assert: (proba >= 0).all() and (proba <= 1).all()
```

**PIPE-INTEG-02: SHAP extraction pattern produces 41 SHAP values**
```
full_pipeline.fit(X_train, y_train)
preprocessor = full_pipeline.named_steps['preprocessor']
model = full_pipeline.named_steps['model']

X_model = preprocessor.transform(X_val.iloc[[0]])
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_model)

Assert: shap_values.values.shape == (1, 41)
Assert: len(get_feature_names()) == shap_values.values.shape[1]
```

**PIPE-INTEG-03: Feature names match SHAP output dimensions**
```
feature_names = get_feature_names()
X_out = preprocessor.transform(X_sample)
Assert: len(feature_names) == X_out.shape[1]
# Verify each name is unique
Assert: len(feature_names) == len(set(feature_names))
```

---

## 15. Known Limitations and Upgrade Paths

### L-01: Zero-Variance Features
`operator_concurrent_order_count` and `log_experience_x_concurrent` are in the `zero_variance` branch because they are currently constant. When the multi-machine operator model is implemented:
1. Remove both from `ZERO_VARIANCE_FEATURES`
2. Add `operator_concurrent_order_count` to `SCALE_FEATURES`
3. `log_experience_x_concurrent` will automatically move to `SCALE_FEATURES` (added to scale branch in `build_pipeline()`)
4. Retrain the full pipeline — existing serialised pipelines are incompatible with the new feature distribution

### L-02: `oee_x_maintenance_ratio` Near-Zero Signal
This interaction has |r| = 0.023 with `is_delayed` in the 120-day run. If Optuna determines it hurts validation AUC, move it to `CANDIDATE_REMOVAL_FEATURES` and add a build-time conditional in `build_pipeline()` that excludes it when `include_weak_interactions=False`.

### L-03: Output Format — Array vs DataFrame
`ColumnTransformer` returns a numpy array by default. Setting `pipeline.set_output(transform='pandas')` (sklearn ≥ 1.2) makes all steps return DataFrames. The SHAP service benefits from DataFrame output (named columns for free). However, XGBoost/LightGBM work correctly with both. The design does not mandate one format — `pipeline.py` should configure `set_output(transform='pandas')` if the sklearn version supports it, with a fallback to numpy array for compatibility.

### L-04: Model-Specific Pipeline Variants
The current design has one preprocessing pipeline shared by all 4 prediction tasks (binary, regression, ordinal, multi-class). If a future task requires different feature engineering (e.g., the root cause classifier benefits from additional interaction features not needed for binary classification), `build_pipeline()` should accept an optional `task` parameter that selects a task-specific variant. For now, one pipeline serves all tasks.

---

*Design Specification — `ml/src/mpc_ml/features/pipeline.py`*  
*Manufacturing Process Copilot Technical Series*
