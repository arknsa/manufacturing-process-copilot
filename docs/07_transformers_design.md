# Manufacturing Process Copilot
## Design Specification: `ml/src/mpc_ml/features/transformers.py`

**Status:** Authoritative — implement exactly as specified  
**Phase:** 1, Day 3  
**Depends on:** `constants.py` (fully specified in Doc 06)  
**Consumed by:** `pipeline.py` (Pipeline steps 1 and 2)

---

## 1. Architecture Overview

`transformers.py` defines exactly two classes. They occupy the first two steps of the sklearn `Pipeline` and together establish the invariant that every downstream step can trust: the input matrix has exactly 41 named columns, zero NaN values, correct dtypes, no target column contamination, and numerically stable interaction features.

```
raw DataFrame (arbitrary columns)
     │
     ▼  Step 1
 ColumnSelector
  • rejects target columns
  • validates 37 FEATURE_COLS exist
  • coerces dtypes
  • fills rolling-feature NaN via cold-start defaults
  • asserts no remaining NaN
  • guards zero-variance columns
  • returns ordered 37-column DataFrame
     │
     ▼  Step 2
 InteractionFeatureAdder
  • appends 4 derived columns
  • applies numerical guards on each formula
  • validates output shape and zero-NaN
  • returns 41-column DataFrame
     │
     ▼  Step 3  (in pipeline.py, not here)
 ColumnTransformer
  • applies LOG / SCALE / passthrough per group
  • returns numpy array or pandas DataFrame
```

Both classes inherit from `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin`. This provides:
- `fit_transform()` default implementation
- `get_params()` / `set_params()` for Optuna and cross-validation compatibility
- `__repr__` generation
- Compatibility with `sklearn.utils.check_is_fitted()`

Both classes must return **pandas DataFrames from `transform()`**, not numpy arrays. The downstream `ColumnTransformer` selects columns by name — a numpy array would silently break all column references.

---

## 2. Class Dependency Diagram

```
constants.py
│
├─ FEATURE_COLS ─────────────────────────────────► ColumnSelector
├─ TARGET_COLS ──────────────────────────────────► ColumnSelector
├─ COLUMN_DTYPE_CONTRACT ────────────────────────► ColumnSelector
├─ COLD_START_DEFAULTS ──────────────────────────► ColumnSelector (as seed)
├─ COLD_START_FEATURE_NAMES ─────────────────────► ColumnSelector
├─ ZERO_VARIANCE_FEATURES ───────────────────────► ColumnSelector
│
└─ INTERACTION_FEATURE_NAMES ────────────────────► InteractionFeatureAdder
   (4 names: lag_as_pct_of_window, tightness_x_queue,
    log_experience_x_concurrent, oee_x_maintenance_ratio)

transformers.py
│
├─ ColumnSelector
│   ├─ fit(X, y=None) → self
│   ├─ transform(X) → DataFrame[FEATURE_COLS]
│   └─ get_feature_names_out() → List[str]
│
└─ InteractionFeatureAdder
    ├─ fit(X, y=None) → self
    ├─ transform(X) → DataFrame[FEATURE_COLS + INTERACTION_FEATURE_NAMES]
    └─ get_feature_names_out() → List[str]

pipeline.py
├─ imports ColumnSelector, InteractionFeatureAdder from transformers.py
└─ assembles Pipeline with these as first two steps
```

---

## 3. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: arbitrary pandas DataFrame                              │
│  Columns: may include FEATURE_COLS + TARGET_COLS + extra        │
│  NaN: possible in rolling features (cold-start)                 │
│  Dtypes: may differ from COLUMN_DTYPE_CONTRACT (API coercion)   │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼  ColumnSelector.transform()
          ╔════════════════════╗
          ║  GUARD PHASE       ║  [raises on violation]
          ╠════════════════════╣
          ║  1. Reject TARGET_COLS          ─→ ValueError if found
          ║  2. Validate FEATURE_COLS exist ─→ ValueError if missing
          ║  3. Coerce dtypes               ─→ Warning + attempt
          ║     (cannot coerce)             ─→ ValueError
          ╚════════════════════╝
                   │
                   ▼  FILL PHASE
          ╔════════════════════╗
          ║  4. Fill NaN in rolling features with cold_start_defaults_
          ║  5. Assert no remaining NaN in FEATURE_COLS  ─→ ValueError
          ║  6. Assert ZERO_VARIANCE_FEATURES values
          ╚════════════════════╝
                   │
                   ▼  SELECT PHASE
          ╔════════════════════╗
          ║  7. Select and reorder to FEATURE_COLS (drops extra cols)
          ║  8. Return DataFrame[37 cols, no NaN, correct dtypes]
          ╚════════════════════╝
                   │
                   ▼  InteractionFeatureAdder.transform()
          ╔════════════════════╗
          ║  COMPUTE PHASE     ║
          ╠════════════════════╣
          ║  9.  lag_as_pct_of_window    (LOG target)
          ║  10. tightness_x_queue       (SCALE target)
          ║  11. log_experience_x_concurrent  (ZERO_VARIANCE)
          ║  12. oee_x_maintenance_ratio (SCALE target)
          ╚════════════════════╝
                   │
                   ▼  VALIDATE PHASE
          ╔════════════════════╗
          ║  13. Assert no name collisions (4 new names not in existing 37)
          ║  14. Assert output shape = (n_rows, 41)
          ║  15. Assert zero NaN in all 41 columns
          ║  16. Return DataFrame[41 cols]
          ╚════════════════════╝
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: pandas DataFrame, n rows × 41 columns                  │
│  Columns: FEATURE_COLS (37) + INTERACTION_FEATURE_NAMES (4)     │
│  NaN: none (guaranteed)                                         │
│  Dtypes: float64 throughout                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. `ColumnSelector` — Full Specification

### 4.1 Purpose

`ColumnSelector` is the pipeline's gatekeeper. Its job is not to transform data — it is to **enforce the contract** that everything downstream depends on. It answers one question per invocation: "Is this input safe to process?"

The class has three distinct behavioral responsibilities:
1. **Schema enforcement** — correct columns, correct dtypes, no targets
2. **Cold-start filling** — replace NaN in rolling features with learned population means
3. **Invariant assertion** — guarantee zero NaN and expected zero-variance column values on output

### 4.2 Constructor Parameters

```
ColumnSelector(
    strict_dtypes: bool = True
)
```

- `strict_dtypes=True` (default): dtype mismatches raise `ValueError` after failed coercion
- `strict_dtypes=False`: dtype mismatches log a warning and attempt coercion; only raise if coercion produces NaN

**Why not a `strict_mode` parameter that relaxes all validations?** Silent failures are worse than loud failures. The only intentional leniency is cold-start NaN filling — all other violations are programming errors that must surface as exceptions in both training and production.

### 4.3 Fit-Time Behaviour

`fit(X, y=None) → self`

Fit is called exactly once during `Pipeline.fit()`. It receives the **training DataFrame** (before any splitting, on the full training set).

**What `fit()` computes and stores:**

**`self.cold_start_defaults_`**: dict mapping each rolling feature name to its mean computed from the training set. Computed only from non-NaN values (`.mean(skipna=True)`). These are the values that will be used to fill NaN in `transform()`.

Critically: these values are learned from training data, NOT taken from `constants.COLD_START_DEFAULTS`. The constants file provides a **seed/documentation reference**; the pipeline learns the true values from the data it sees. This is the mechanism that prevents training-serving skew: the same population mean that was implicitly used during simulation is explicitly re-learned and stored.

**`self.feature_names_in_`**: stored as `list(FEATURE_COLS)` — confirms the expected schema the transformer was fitted on.

**`self.n_features_in_`**: `len(FEATURE_COLS)` = 37.

**`self.zero_variance_observed_values_`**: dict mapping each `ZERO_VARIANCE_FEATURES` column to its unique observed value from training. Currently `{'operator_concurrent_order_count': 0.0, ...}`. Used at transform-time to detect when these columns are no longer zero-variance (signalling that the zero-variance assumption has broken and the pipeline needs retraining).

**`self.is_fitted_`**: boolean flag set to `True` at end of fit. Checked by `sklearn.utils.check_is_fitted()` and by every `transform()` call.

**Fit-time validations (same as transform-time):**
All schema validations run during `fit()`. A training DataFrame that fails schema validation is a configuration error that must surface immediately, not during the first production prediction.

### 4.4 Transform-Time Behaviour

`transform(X) → pd.DataFrame`

Called during both `Pipeline.fit_transform()` (training) and `Pipeline.predict()` (inference). Must be idempotent — calling it twice on the same input produces identical output.

**Step-by-step execution:**

**Step 1: Check fitted state**
Call `sklearn.utils.check_is_fitted(self, 'is_fitted_')`. Raises `sklearn.exceptions.NotFittedError` with message: `"ColumnSelector has not been fitted. Call fit() before transform()."` This is the first check — no other validation runs if not fitted.

**Step 2: Reject target columns**
Check if any column in `TARGET_COLS` appears in `X.columns`. If found, raise `ValueError`:
```
ValueError: "Target columns found in input DataFrame: {found_targets}. 
These columns must be excluded before calling pipeline.transform(). 
Did you forget to call X.drop(TARGET_COLS, axis=1)?"
```
This check is strict and has no lenient mode. Target columns in inference input indicate a bug in the calling code.

**Step 3: Validate FEATURE_COLS existence**
Compute `missing = set(FEATURE_COLS) - set(X.columns)`. If non-empty, raise `ValueError`:
```
ValueError: "Missing feature columns: {missing}. 
Expected all 37 features from FEATURE_COLS. 
Add these columns or check the simulation output schema."
```
Extra columns beyond `FEATURE_COLS` are **silently dropped** in Step 7. This is intentional — callers should not be penalised for providing enriched DataFrames.

**Step 4: Dtype coercion**
For each column in `FEATURE_COLS`:
- Check dtype against `COLUMN_DTYPE_CONTRACT[col]`
- If mismatch: attempt `pd.to_numeric()` coercion
- If coercion produces NaN where none existed before: raise `ValueError` (or warning if `strict_dtypes=False`)
- If coercion succeeds without new NaN: emit `logging.WARNING` regardless of `strict_dtypes`

**Why coerce rather than always reject?** JSON APIs round-trip integers as floats. A column defined as `int64` will arrive as `float64` from virtually every REST API. Rejecting this in production would make the service brittle to a completely benign and universal serialisation artefact.

**Step 5: Cold-start NaN filling**
For each column in `COLD_START_FEATURE_NAMES` (the 7 rolling features):
- If any NaN values exist: fill with `self.cold_start_defaults_[col]`
- Log at `DEBUG` level: `"Filled {n_filled} NaN values in {col} with cold-start default {value:.4f}"`

No warning is emitted — this is the **expected production path** for new product-machine-operator combinations. Logging at DEBUG allows observability without noise.

**Step 6: Assert no remaining NaN**
After cold-start filling, check all 37 `FEATURE_COLS` for NaN. If any remain:
```
ValueError: "Unexpected NaN values in non-rolling feature(s): {cols_with_nan}. 
NaN is only expected in rolling historical features and will be auto-filled.
Check upstream data pipeline for missing values in {cols_with_nan}."
```

**Step 7: Zero-variance assertion**
For each column in `ZERO_VARIANCE_FEATURES`:
- Check current values against `self.zero_variance_observed_values_[col]`
- If any row has a value different from the expected zero-variance value:
  - Emit `logging.WARNING`: `"Column {col} was zero-variance at fit time (value: {expected}) but now contains non-zero values {unique_values}. Consider retraining the pipeline."`
  - Do NOT raise — the pipeline should not crash in production when a simulation assumption changes. But the warning must be observable.

**Step 8: Select and reorder**
Return `X[FEATURE_COLS].copy()` — select exactly the 37 required columns in the canonical order, with a copy to prevent downstream modifications affecting the input.

### 4.5 `get_feature_names_out()`

Returns `list(FEATURE_COLS)` — the 37 feature names in canonical order. This is called by `Pipeline.get_feature_names_out()` and by downstream `ColumnTransformer` to build its feature name registry. Must work whether or not the transformer has been fitted (returns `FEATURE_COLS` regardless).

### 4.6 Training vs. Serving Skew Prevention

The skew prevention mechanism is: `fit()` learns `cold_start_defaults_` from training data; `transform()` uses stored `cold_start_defaults_`, never the constants file values. This creates a sealed contract:

- Training: `fit_transform(train_df)` → computes cold-start defaults from `train_df` → stores them → applies them
- Validation/Test: `transform(val_df)` → applies stored training defaults (not recomputed from val/test)
- Production: `transform(new_order)` → applies stored training defaults (not recomputed from production data)

The pipeline artifact in MLflow (the serialised `pipeline.pkl`) contains the fitted `ColumnSelector` with its stored `cold_start_defaults_`. Loading this artifact and calling `transform()` guarantees the same defaults at serve time as at train time.

---

## 5. `InteractionFeatureAdder` — Full Specification

### 5.1 Purpose

`InteractionFeatureAdder` computes the 4 derived features defined in `INTERACTION_FEATURE_NAMES`. It receives the clean 37-column DataFrame from `ColumnSelector` and appends 4 columns, producing a 41-column DataFrame.

This class is almost entirely **stateless** — `fit()` performs input schema validation and sets `is_fitted_`, but learns no parameters from data. All formulas use only the input values and domain-knowledge constants.

The exception: if in the future an interaction feature requires fit-time computation (e.g., learning a clipping threshold from training data), the class structure already supports it without interface changes.

### 5.2 Constructor Parameters

```
InteractionFeatureAdder(
    lag_clip_upper: float = 10.0,
    oee_maintenance_scale: float = 30.0
)
```

- `lag_clip_upper = 10.0`: upper bound for `lag_as_pct_of_window`. Values above this are capped. An order released 10× after its planned window is already at maximum risk; the exact magnitude beyond that carries no additional predictive signal and would create extreme outliers.
- `oee_maintenance_scale = 30.0`: the number of days per "unit" in the OEE maintenance ratio denominator. Controls how aggressively recent PM is penalised vs. not.

Both parameters are exposed as constructor arguments (not module-level constants) so Optuna can tune them if needed. The defaults are domain-validated choices, not arbitrary.

### 5.3 Fit-Time Behaviour

`fit(X, y=None) → self`

**What `fit()` does:**
1. Calls `sklearn.utils.check_is_fitted(self, 'is_fitted_')` — fails gracefully if called twice
2. Validates `X` is a pandas DataFrame
3. Validates all columns in `FEATURE_COLS` are present (should always pass after `ColumnSelector`)
4. Validates no column in `INTERACTION_FEATURE_NAMES` already exists in `X` (would indicate double-fitting or incorrect pipeline assembly)
5. Sets `self.is_fitted_ = True`
6. Returns `self`

`fit()` does NOT compute any statistics from the data. If you need a fit-time statistic for an interaction formula, it belongs in `ColumnSelector` or a new transformer, not here.

### 5.4 Transform-Time Behaviour

`transform(X) → pd.DataFrame`

**Step 1: Check fitted state**
Same check as `ColumnSelector`. Raises `NotFittedError` if not fitted.

**Step 2: Validate input**
- Assert `X` is a pandas DataFrame
- Assert all `FEATURE_COLS` present (defensive; should always pass after ColumnSelector)
- Assert no `INTERACTION_FEATURE_NAMES` already in `X.columns` — collision would silently overwrite a base feature

**Step 3: Compute all 4 interaction features**
Each formula is computed independently on the full DataFrame (vectorised pandas operations). See §6 for full formula specifications.

**Step 4: Concatenate**
Build a new DataFrame by concatenating the original `X` with the 4 computed Series. Use `pd.concat([X, interactions_df], axis=1)`. Do not mutate the input `X` — always work on a copy.

**Step 5: Post-computation validation**
- Assert output shape is `(len(X), 41)` — `len(FEATURE_COLS) + len(INTERACTION_FEATURE_NAMES)`
- Assert zero NaN in all 41 columns
- Assert all interaction feature values are finite (no `np.inf`, no `-np.inf`)

**Step 6: Return**
Return the 41-column DataFrame. All columns are `float64`.

### 5.5 `get_feature_names_out()`

Returns `list(FEATURE_COLS) + list(INTERACTION_FEATURE_NAMES)` — the 41 feature names in canonical order. The 4 interaction features are appended after the 37 base features, maintaining their original relative order.

---

## 6. Interaction Feature Formulas — Complete Specification

### 6.1 `lag_as_pct_of_window`

**Business meaning:**  
The fraction of the total planning window consumed by the release lag before manufacturing starts. When this value equals 0.5, half the available lead time is already spent waiting for materials before the first tool touches the part. When it exceeds 1.0 — which occurs for 13.5% of orders in the simulation — the order was released after its own planned completion date. Such orders are virtually guaranteed to be late regardless of execution quality.

**Why this interaction outperforms its components:**  
`release_lag_hours` alone (|r|=0.478) tells you a lag is long in absolute terms, but a 40-hour lag on a 200-hour order is fine; a 40-hour lag on a 50-hour order is catastrophic. `planned_lead_time_hours` alone (|r|=0.257) tells you the window is short, but a short window is only a problem if the lag has eaten into it. The ratio captures the joint condition: the lag is large relative to the window. Empirical confirmation: |r|=0.521, the strongest predictor in the 41-feature set.

**Formula:**
```
lag_as_pct_of_window = release_lag_hours / max(planned_lead_time_hours, ε)

where ε = 0.1  (minimum denominator clamp, in hours = 6 minutes)
```

**Implementation as vectorised pandas:**
```
denominator = planned_lead_time_hours.clip(lower=0.1)
lag_as_pct_of_window = (release_lag_hours / denominator).clip(upper=lag_clip_upper)
```

**Edge cases and numerical stability:**

| Scenario | Input values | Without guard | With guard | Correct? |
|---|---|---|---|---|
| Normal order | lag=12h, lead=30h | 0.40 | 0.40 | ✓ |
| Released after deadline | lag=50h, lead=30h | 1.67 | 1.67 | ✓ value > 1 is valid signal |
| Extreme late release | lag=140h, lead=8.3h | 16.87 | 10.0 (clipped) | ✓ clip at `lag_clip_upper` |
| Very tight window | lag=5h, lead=0.5h | 10.0 | 10.0 (clipped) | ✓ |
| Zero lag | lag=0, lead=30h | 0.0 | 0.0 | ✓ |
| Zero lead time | lag=5h, lead=0.0 | ÷0 = ∞ | 50.0 → clipped 10.0 | ✓ clip prevents ∞ |
| Both near zero | lag=0, lead=0.0 | 0/ε = 0 | 0.0 | ✓ |

**Why `lag_clip_upper = 10.0`:**  
Empirical maximum in simulation is 7.40. A cap at 10.0 provides ~35% headroom above observed maxima while preventing division-result explosion on pathological production inputs. At 10.0, the log1p transform (this feature is in `LOG_FEATURES`) produces log1p(10) = 2.40 — a reasonable upper bound on the standardised scale.

**Output range:** [0.0, `lag_clip_upper`] = [0.0, 10.0] in theory; [0.016, 7.40] in practice.  
**Preprocessing group:** `LOG_FEATURES` (skew +3.76 confirmed on validation data).

---

### 6.2 `tightness_x_queue`

**Business meaning:**  
A congestion risk multiplier. A tight schedule alone is dangerous but manageable — there's no buffer, but if execution is smooth, the order completes on time. An existing machine queue alone is a delay risk — but for a loosely scheduled order, the queue wait may be absorbed. Only when BOTH conditions are true simultaneously (tight schedule + non-empty queue) does the order face compounding pressure: it needs every minute of its buffer but must also wait behind other orders.

In the current simulation `work_center_queue_depth_at_release` is binary {0, 1}. The interaction therefore acts as a **gate**: when queue=0, `tightness_x_queue = 0` regardless of tightness; when queue=1, `tightness_x_queue = schedule_tightness_ratio`. This is the correct semantic — the tightness signal only matters when there is congestion to absorb it.

**Note on correlation direction:** Empirical correlation with `is_delayed` is -0.399. This appears counterintuitive (tighter+queue → less delay?) but reflects a simulation calibration artifact: high-utilization machines (which generate the queue signal) have lower observed delay rates in the current simulation due to selection effects in machine assignment. The magnitude (|r|=0.40) is genuine signal the model can use; the negative direction is a known property of this dataset.

**Formula:**
```
tightness_x_queue = schedule_tightness_ratio × work_center_queue_depth_at_release
```

**Implementation:**
```
tightness_x_queue = schedule_tightness_ratio * work_center_queue_depth_at_release
```

No guards needed. Both inputs are bounded [0, 1.02] and [0, 1] respectively. The product is bounded [0, ~1.02]. No division. No log.

**Edge cases and numerical stability:**

| Scenario | Tightness | Queue | Result | Meaning |
|---|---|---|---|---|
| No queue | 0.80 | 0.0 | 0.0 | Queue not present; tightness irrelevant |
| Queue, tight | 0.90 | 1.0 | 0.90 | High-risk compound condition |
| Queue, loose | 0.40 | 1.0 | 0.40 | Queue absorbed by slack |
| No queue, tight | 0.90 | 0.0 | 0.0 | Tight but no congestion |
| Max values | 1.02 | 1.0 | 1.02 | Theoretical maximum |

**Why not cap at 1.0?** `schedule_tightness_ratio` has an observed maximum of 1.023 (the order is already scheduled with zero slack — any overrun = delay). This value slightly above 1.0 carries real signal. Capping at 1.0 would destroy a meaningful edge case.

**Output range:** [0.0, 1.016] observed; [0.0, ~1.03] theoretical.  
**Preprocessing group:** `SCALE_FEATURES`.

---

### 6.3 `log_experience_x_concurrent`

**Business meaning:**  
Concurrent workload weighted by operator seniority. A junior operator (2 months experience) managing 3 simultaneous orders faces qualitatively higher risk than a senior operator (120 months) managing the same 3. Log-transforming experience captures the diminishing returns of experience: the difference between 1 and 12 months is huge; between 120 and 132 months is marginal.

**Current state: zero-variance placeholder.** Because `operator_concurrent_order_count = 0` for all orders in the current simulation (single-machine assignment model), this feature is identically 0.0. The formula is correct; the input is degenerate.

**Formula:**
```
log_experience_x_concurrent = log1p(operator_experience_months) × operator_concurrent_order_count
```

**Implementation:**
```
log_experience_x_concurrent = np.log1p(operator_experience_months) * operator_concurrent_order_count
```

No guards needed. `log1p` handles `operator_experience_months = 0` gracefully: `log1p(0) = 0`. Multiplication by 0 always produces 0.

**Edge cases when non-zero concurrent count is implemented:**

| Scenario | Experience | Concurrent | Result | Meaning |
|---|---|---|---|---|
| Current (always) | any | 0 | 0.0 | No concurrent workload |
| Junior, high load | 6 months | 3 | log1p(6) × 3 = 5.77 | High risk |
| Senior, high load | 120 months | 3 | log1p(120) × 3 = 14.4 | Lower risk |
| Zero experience | 0 months | 3 | log1p(0) × 3 = 0.0 | Edge case; treated as no modifier |
| Max values | 176 months | 5 | log1p(176) × 5 = 25.9 | Would need scaling |

**Zero-variance handling in the pipeline:**  
This feature is in `ZERO_VARIANCE_FEATURES`. The `ColumnTransformer` in `pipeline.py` must include it in a passthrough branch, NOT in `StandardScaler`. When `operator_concurrent_order_count` becomes non-zero in a future simulation run, this feature should be moved from `ZERO_VARIANCE_FEATURES` to `SCALE_FEATURES` and the pipeline refitted.

**Output range:** [0.0, 0.0] currently; [0.0, ~26.0] theoretical.  
**Preprocessing group:** `PASSTHROUGH` (zero-variance).

---

### 6.4 `oee_x_maintenance_ratio`

**Business meaning:**  
OEE adjusted for maintenance recency. OEE measured over the last 30 days captures average machine performance, but a machine that was well-maintained last week is at lower breakdown risk than a machine with the same 30-day OEE that's 3 months overdue for its planned maintenance. This feature combines the two signals: recent PM boosts the effective reliability above the 30-day OEE baseline; overdue PM reduces it.

**Formula correction notice:** The initial implementation (v1 in calibration runs) used `.clip(lower=0.5)` on the denominator expression `(days_since_PM / 30).clip(lower=1)`. This caused machines serviced in the last 15 days to receive an OEE **boost to 2× their measured OEE** (OEE/0.5 = 2×OEE), producing values up to 1.569 — physically meaningless for a metric bounded [0, 1]. The corrected formula uses `max(days_since_PM / 30, 1.0)`, which enforces: recent PM → no penalty (ratio = OEE); overdue PM → reduced ratio (< OEE). The corrected version also shows slightly better correlation with is_delayed (0.023 vs 0.003) and a semantically clean output range [0.19, 0.79].

**Formula (corrected v2):**
```
oee_x_maintenance_ratio = machine_oee_30d / max(days_since_last_planned_maintenance / oee_maintenance_scale, 1.0)

where oee_maintenance_scale = 30.0  (constructor parameter; days per "unit")
```

**Implementation as vectorised pandas:**
```
denominator = (days_since_last_planned_maintenance / oee_maintenance_scale).clip(lower=1.0)
oee_x_maintenance_ratio = machine_oee_30d / denominator
```

**Why `clip(lower=1.0)` and not `clip(lower=0.1)` or similar?**  
The `clip(lower=1.0)` means: "if the machine was serviced within the last 30 days (`days_since_PM / 30 < 1`), treat it as if it was serviced exactly 30 days ago." This removes the recent-PM boost (which was physically meaningless) while not penalising freshly serviced machines. The formula only penalises machines that are OVERDUE relative to the 30-day baseline.

**Semantic behaviour with `oee_maintenance_scale = 30.0`:**

| Days since PM | Denominator | Result (OEE=0.65) | Interpretation |
|---|---|---|---|
| 0 days | max(0/30, 1) = 1.0 | 0.65 | Freshly serviced — baseline OEE |
| 15 days | max(0.5, 1) = 1.0 | 0.65 | Still within 30-day window — baseline |
| 30 days | max(1.0, 1) = 1.0 | 0.65 | At exactly the scale — baseline |
| 60 days | max(2.0, 1) = 2.0 | 0.325 | 2× overdue — half OEE |
| 89 days | max(2.97, 1) = 2.97 | 0.219 | 3× overdue — severe penalty |

**Edge cases and numerical stability:**

| Scenario | days_since_PM | OEE | Denominator | Result |
|---|---|---|---|---|
| Just serviced | 0 | 0.65 | 1.0 | 0.650 |
| At PM boundary | 30 | 0.65 | 1.0 | 0.650 |
| Overdue 1 cycle | 60 | 0.65 | 2.0 | 0.325 |
| Very overdue | 89 | 0.65 | 2.97 | 0.219 |
| Zero OEE | 30 | 0.0 | 1.0 | 0.0 (valid) |
| OEE = 1.0 | 30 | 1.0 | 1.0 | 1.0 (valid) |
| days_since_PM < 0 | −1 | 0.65 | max(−0.033, 1) = 1.0 | 0.650 (impossible but safe) |

No division by zero is possible because `denominator >= 1.0` always.

**Practical note on signal strength:** Correlation with `is_delayed` = 0.023 (v2 corrected formula). This is weak but not zero. The feature is in `CANDIDATE_REMOVAL_FEATURES` in `constants.py`. Whether it improves model performance will be determined empirically during Optuna tuning. The formula is correct regardless of signal strength.

**Output range:** [min_OEE / max_denominator, max_OEE / 1.0] = [0.568/2.97, 0.785] = [0.191, 0.785] in simulation.  
**Preprocessing group:** `SCALE_FEATURES`.

---

## 7. Failure Scenarios

Each failure is described with: trigger, class that raises, error type, message contract, and resolution.

### F-01: Target Column Present
- **Trigger:** `X` contains `is_delayed`, `delay_minutes`, `delay_category`, or `delay_root_cause`
- **Class:** `ColumnSelector.transform()`
- **Type:** `ValueError`
- **Message contract:** Must list the specific offending columns
- **When occurs:** Most commonly when a notebook passes the full dataset (features + targets) directly to the pipeline without splitting
- **Resolution:** Caller must drop target columns before pipeline

### F-02: Missing Feature Column
- **Trigger:** One or more columns from `FEATURE_COLS` are absent from `X`
- **Class:** `ColumnSelector.transform()`
- **Type:** `ValueError`
- **Message contract:** Must list all missing column names, not just the first
- **When occurs:** Schema drift between simulation output and pipeline expectation; new simulation version drops a feature
- **Resolution:** Check simulation output schema; update `FEATURE_COLS` if intentional change

### F-03: Uncoercible Dtype
- **Trigger:** Column has unexpected dtype AND `pd.to_numeric()` produces NaN where no NaN existed
- **Class:** `ColumnSelector.transform()`
- **Type:** `ValueError` (if `strict_dtypes=True`)
- **Message contract:** Must specify column name, expected dtype, actual dtype
- **When occurs:** A column arrives as non-numeric string from a malformed API payload
- **Resolution:** Fix upstream data serialisation

### F-04: Unexpected NaN in Required Feature
- **Trigger:** After cold-start filling, NaN remains in a non-rolling feature column
- **Class:** `ColumnSelector.transform()`
- **Type:** `ValueError`
- **Message contract:** Must list columns with remaining NaN, confirm which are rolling vs. non-rolling
- **When occurs:** Upstream data pipeline has a bug producing missing values in operational features (e.g., machine OEE not computed)
- **Resolution:** Fix upstream; identify root cause of NaN in operational feature

### F-05: transform() Before fit()
- **Trigger:** `ColumnSelector.transform()` or `InteractionFeatureAdder.transform()` called without prior `fit()`
- **Class:** Either transformer
- **Type:** `sklearn.exceptions.NotFittedError`
- **Message contract:** sklearn standard: `"This {class_name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."`
- **When occurs:** Pipeline loaded from disk without deserialisation; or pipeline steps assembled incorrectly
- **Resolution:** Ensure `Pipeline.fit()` has been called on the training data before `Pipeline.transform()`

### F-06: Interaction Name Collision
- **Trigger:** `X` already contains a column named `lag_as_pct_of_window`, `tightness_x_queue`, `log_experience_x_concurrent`, or `oee_x_maintenance_ratio`
- **Class:** `InteractionFeatureAdder.transform()`
- **Type:** `ValueError`
- **Message contract:** Must name the colliding column(s) and suggest checking for double-invocation
- **When occurs:** Pipeline assembled with two `InteractionFeatureAdder` steps; or simulation adds a column with the same name as an interaction
- **Resolution:** Remove duplicate pipeline step; rename collision in simulation if it adds a feature with same name

### F-07: NaN in Interaction Output
- **Trigger:** A computed interaction formula produces NaN despite the numerical guards
- **Class:** `InteractionFeatureAdder.transform()`
- **Type:** `ValueError` (post-computation assertion)
- **Message contract:** Must identify which interaction produced NaN and the input values that caused it
- **When occurs:** Edge case not covered by guards (should not occur with complete guard specification)
- **Resolution:** Add specific guard for the identified input pattern; file as a bug

### F-08: Infinite Value in Interaction Output
- **Trigger:** An interaction formula produces `np.inf` or `-np.inf`
- **Class:** `InteractionFeatureAdder.transform()`
- **Type:** `ValueError` (post-computation assertion)
- **Message contract:** Must identify which interaction and the input values
- **When occurs:** `lag_as_pct_of_window` if `planned_lead_time_hours = 0` and the `clip(lower=0.1)` guard fails somehow
- **Resolution:** Verify guard is in place; this should never occur given the simulation's minimum `planned_lead_time_hours = 8.3`

### F-09: Zero-Variance Column Becomes Non-Zero
- **Trigger:** `operator_concurrent_order_count` contains non-zero values
- **Class:** `ColumnSelector.transform()`
- **Type:** `logging.WARNING` (not an exception)
- **Message contract:** Must log the column name, expected value, and observed new values
- **When occurs:** Future simulation adds multi-machine operator scheduling
- **Resolution:** Move `operator_concurrent_order_count` (and `log_experience_x_concurrent`) from `ZERO_VARIANCE_FEATURES` to `SCALE_FEATURES`; retrain the pipeline

### F-10: Shape Mismatch on InteractionFeatureAdder Output
- **Trigger:** Output DataFrame has fewer or more than 41 columns
- **Class:** `InteractionFeatureAdder.transform()`
- **Type:** `AssertionError` (internal post-condition)
- **Message contract:** Must report expected 41 columns, actual column count, and list any unexpected columns
- **When occurs:** Programming error in the transform implementation; extra column concatenated accidentally
- **Resolution:** Debug the concatenation step

---

## 8. Production Inference Behaviour

When the serialised `Pipeline` is loaded from MLflow and called on a new production order, this is the expected execution path:

**Scenario A: Normal inference on a known product-machine-operator combination**
1. `ColumnSelector` receives 37-column dict/DataFrame from the API
2. No target columns → passes check
3. All 37 FEATURE_COLS present → passes check
4. Dtypes match (possible minor coercion for int→float) → passes
5. No NaN in any column → no cold-start filling needed
6. All zero-variance columns confirm expected values → WARNING suppressed
7. Returns clean 37-column DataFrame
8. `InteractionFeatureAdder` computes 4 interactions, returns 41-column DataFrame
9. `ColumnTransformer` applies transforms, returns model-ready array
10. Model produces prediction

**Scenario B: Cold-start inference (new product, no history)**
Steps 1–4 same as A. At step 5:
- `product_delay_rate_90d = NaN` (new product, no history)
- `product_x_machine_delay_rate_90d = NaN` (new combination)
- `ColumnSelector` fills both with `self.cold_start_defaults_` values (learned from training set)
- Logs at DEBUG level: "Filled 2 NaN values in product_delay_rate_90d, product_x_machine_delay_rate_90d with cold-start defaults"
- Continues normally

**Scenario C: Inference with extra columns (enriched API payload)**
The API might provide metadata columns alongside features: `order_id`, `product_name`, `machine_code`. `ColumnSelector` silently drops all columns not in `FEATURE_COLS`. This is correct and expected — the class is a SELECTOR.

**Scenario D: Concurrent order count becomes non-zero (future)**
`ColumnSelector` emits a WARNING. The pipeline continues and produces a prediction using `log_experience_x_concurrent = log1p(experience) × concurrent_count`. The prediction is valid (the formula is correct); however, the pipeline was trained with this feature always zero, so its weight in the model is effectively zero. The WARNING alerts the operations team to retrain.

**Scenario E: Inference on batch of orders (100-order batch API)**
All transformations are vectorised pandas operations. Performance scales linearly with batch size. No per-row iteration in any transformer. Target latency: <10ms for 100-order batch through `ColumnSelector` + `InteractionFeatureAdder`.

---

## 9. Unit Test Scenarios

### `ColumnSelector` Tests

**CSEL-01: Happy path — clean DataFrame**
```
Input: valid 37-column DataFrame from simulation, no NaN, correct dtypes
Expected: returns identical 37-column DataFrame in FEATURE_COLS order
Assert: output.shape == (n_rows, 37)
Assert: list(output.columns) == FEATURE_COLS
Assert: output.isna().sum().sum() == 0
```

**CSEL-02: Target columns rejected**
```
Input: DataFrame containing FEATURE_COLS + ['is_delayed', 'delay_minutes']
Expected: raises ValueError
Assert: error message contains 'is_delayed' and 'delay_minutes'
Assert: error message contains the word 'Target' or 'target'
```

**CSEL-03: Missing feature column**
```
Input: DataFrame missing 'release_lag_hours' and 'material_availability_at_release'
Expected: raises ValueError
Assert: error message contains both missing column names
Assert: error message does NOT include columns that ARE present
```

**CSEL-04: Extra columns silently dropped**
```
Input: 37-column DataFrame + 5 extra columns ('order_id', etc.)
Expected: returns 37-column DataFrame with no error, no warning
Assert: output.shape[1] == 37
Assert: set(output.columns) == set(FEATURE_COLS)
Assert: none of the extra 5 columns appear in output
```

**CSEL-05: Cold-start NaN filling for rolling feature**
```
Input: valid DataFrame with NaN in 'product_delay_rate_90d' (rows 5, 10, 15)
Expected: returns DataFrame with NaN filled by fitted cold_start_defaults_
Assert: output['product_delay_rate_90d'].isna().sum() == 0
Assert: output.loc[5, 'product_delay_rate_90d'] == selector.cold_start_defaults_['product_delay_rate_90d']
Assert: output.loc[0, 'product_delay_rate_90d'] == original non-NaN value (unchanged)
```

**CSEL-06: NaN in non-rolling feature raises**
```
Input: valid DataFrame with NaN in 'schedule_tightness_ratio'
Expected: raises ValueError
Assert: error message contains 'schedule_tightness_ratio'
Assert: error message contains 'unexpected NaN' or 'non-rolling'
```

**CSEL-07: cold_start_defaults_ learned from training data, not constants**
```
Setup: train DataFrame where product_delay_rate_90d mean = 0.412 (differs from COLD_START_DEFAULTS constant 0.343)
After fit(): selector.cold_start_defaults_['product_delay_rate_90d'] == 0.412 (not 0.343)
Assert: fit-time mean used, not constants value
```

**CSEL-08: transform() before fit() raises NotFittedError**
```
Input: fresh ColumnSelector (not fitted)
Action: call transform(valid_df)
Expected: raises NotFittedError
```

**CSEL-09: Dtype coercion with warning**
```
Input: DataFrame where 'is_expedited' is float64 (should be int64) with no NaN
Expected: no exception, WARNING emitted
Assert: output['is_expedited'].notna().all()  (no NaN introduced by coercion)
```

**CSEL-10: Dtype coercion failure raises ValueError**
```
Input: DataFrame where 'quantity' contains string values 'N/A'
Expected: raises ValueError (strict_dtypes=True)
Assert: error message contains 'quantity' and 'coerce'
```

**CSEL-11: Zero-variance column non-zero emits warning, does not raise**
```
Input: DataFrame with 'operator_concurrent_order_count' = 2 (not 0)
Expected: no exception, WARNING logged
Assert: output['operator_concurrent_order_count'] == 2 (value preserved unchanged)
```

**CSEL-12: Empty DataFrame (0 rows) passes without error**
```
Input: valid 37-column DataFrame with 0 rows
Expected: returns 0-row DataFrame with correct columns
Assert: output.shape == (0, 37)
Assert: no ValueError or assertion error
```

**CSEL-13: Single-row DataFrame**
```
Input: valid 37-column DataFrame with 1 row
Expected: returns 1-row DataFrame with correct columns
Assert: output.shape == (1, 37)
```

**CSEL-14: Column order preserved as FEATURE_COLS**
```
Input: valid 37-column DataFrame with columns in reverse order
Expected: returns DataFrame with columns in FEATURE_COLS canonical order
Assert: list(output.columns) == list(FEATURE_COLS)
```

**CSEL-15: fit_transform idempotency**
```
Action: selector.fit_transform(df) then selector.transform(df)
Assert: both outputs are identical (same values, same shape, same dtypes)
```

---

### `InteractionFeatureAdder` Tests

**IADDER-01: Happy path — correct output shape**
```
Input: valid 37-column DataFrame from ColumnSelector
Expected: returns 41-column DataFrame
Assert: output.shape == (n_rows, 41)
Assert: list(output.columns[:37]) == FEATURE_COLS
Assert: list(output.columns[37:]) == INTERACTION_FEATURE_NAMES
```

**IADDER-02: No NaN in output**
```
Input: valid 37-column DataFrame (no NaN guaranteed by ColumnSelector)
Expected: output has zero NaN in all 41 columns
Assert: output.isna().sum().sum() == 0
```

**IADDER-03: No infinity in output**
```
Input: valid 37-column DataFrame
Expected: output has no inf or -inf
Assert: np.isfinite(output[INTERACTION_FEATURE_NAMES].values).all()
```

**IADDER-04: `lag_as_pct_of_window` normal computation**
```
Input: row with release_lag_hours=12, planned_lead_time_hours=30
Expected: lag_as_pct_of_window = 12/30 = 0.4
Assert: abs(output.loc[0, 'lag_as_pct_of_window'] - 0.4) < 1e-9
```

**IADDER-05: `lag_as_pct_of_window` clipped at upper bound**
```
Input: row with release_lag_hours=200, planned_lead_time_hours=10
  raw = 200/10 = 20.0, should be clipped to lag_clip_upper=10.0
Expected: output.loc[0, 'lag_as_pct_of_window'] == 10.0
```

**IADDER-06: `lag_as_pct_of_window` near-zero denominator handled**
```
Input: row with release_lag_hours=5, planned_lead_time_hours=0.0
  denominator clipped to 0.1, raw = 5/0.1 = 50.0, then clipped to 10.0
Expected: output.loc[0, 'lag_as_pct_of_window'] == 10.0 (no exception)
```

**IADDER-07: `lag_as_pct_of_window` > 1.0 is valid (not capped at 1.0)**
```
Input: row with release_lag_hours=50, planned_lead_time_hours=30
  ratio = 50/30 = 1.667 < lag_clip_upper=10.0
Expected: output.loc[0, 'lag_as_pct_of_window'] ≈ 1.667 (not capped at 1.0)
Assert: output.loc[0, 'lag_as_pct_of_window'] > 1.0
```

**IADDER-08: `tightness_x_queue` when queue=0**
```
Input: row with schedule_tightness_ratio=0.80, work_center_queue_depth=0.0
Expected: tightness_x_queue = 0.0
Assert: output.loc[0, 'tightness_x_queue'] == 0.0
```

**IADDER-09: `tightness_x_queue` when queue=1**
```
Input: row with schedule_tightness_ratio=0.75, work_center_queue_depth=1.0
Expected: tightness_x_queue = 0.75
Assert: abs(output.loc[0, 'tightness_x_queue'] - 0.75) < 1e-9
```

**IADDER-10: `log_experience_x_concurrent` always zero in current simulation**
```
Input: any valid DataFrame (operator_concurrent_order_count is all 0)
Expected: log_experience_x_concurrent is all 0.0
Assert: (output['log_experience_x_concurrent'] == 0.0).all()
```

**IADDER-11: `log_experience_x_concurrent` formula when concurrent > 0**
```
Input: row with operator_experience_months=6, operator_concurrent_order_count=3
  expected = log1p(6) × 3 ≈ 1.946 × 3 = 5.838
Expected: output.loc[0, 'log_experience_x_concurrent'] ≈ 5.838
(Note: this tests future behavior when zero-variance constraint is relaxed)
```

**IADDER-12: `oee_x_maintenance_ratio` at PM boundary (30 days)**
```
Input: row with machine_oee_30d=0.65, days_since_last_planned_maintenance=30
  denominator = max(30/30, 1.0) = max(1.0, 1.0) = 1.0
  result = 0.65 / 1.0 = 0.65
Expected: output.loc[0, 'oee_x_maintenance_ratio'] ≈ 0.65
```

**IADDER-13: `oee_x_maintenance_ratio` overdue PM penalty**
```
Input: row with machine_oee_30d=0.65, days_since_last_planned_maintenance=60
  denominator = max(60/30, 1.0) = max(2.0, 1.0) = 2.0
  result = 0.65 / 2.0 = 0.325
Expected: output.loc[0, 'oee_x_maintenance_ratio'] ≈ 0.325
```

**IADDER-14: `oee_x_maintenance_ratio` freshly serviced (no boost above OEE)**
```
Input: row with machine_oee_30d=0.78, days_since_last_planned_maintenance=0
  denominator = max(0/30, 1.0) = max(0.0, 1.0) = 1.0
  result = 0.78 / 1.0 = 0.78  (not boosted above 0.78)
Expected: output.loc[0, 'oee_x_maintenance_ratio'] == 0.78
Assert: output.loc[0, 'oee_x_maintenance_ratio'] <= machine_oee_30d  (no artificial boost)
```

**IADDER-15: No mutation of input DataFrame**
```
Input: clean 37-column DataFrame X
Action: adder.transform(X)
Assert: X.shape[1] == 37 after transform (input not modified)
Assert: INTERACTION_FEATURE_NAMES not in X.columns
```

**IADDER-16: transform() before fit() raises NotFittedError**
```
Input: fresh InteractionFeatureAdder (not fitted)
Action: call transform(valid_df)
Expected: raises NotFittedError
```

**IADDER-17: Column name collision raises ValueError**
```
Input: 37-column DataFrame that already has a column 'lag_as_pct_of_window'
  (would happen if run through InteractionFeatureAdder twice)
Expected: raises ValueError
Assert: error message contains 'lag_as_pct_of_window' and 'collision' or 'already exists'
```

**IADDER-18: get_feature_names_out() returns 41 names**
```
Input: fitted InteractionFeatureAdder
Expected: get_feature_names_out() returns list of 41 strings
Assert: len(adder.get_feature_names_out()) == 41
Assert: adder.get_feature_names_out()[:37] == FEATURE_COLS
Assert: adder.get_feature_names_out()[37:] == INTERACTION_FEATURE_NAMES
```

---

### Integration Tests

**INTEG-01: Full pipeline chain (ColumnSelector → InteractionFeatureAdder)**
```
Input: simulation output DataFrame with target columns included
Setup: pipe = Pipeline([('selector', ColumnSelector()), ('adder', InteractionFeatureAdder())])
       pipe.fit(train_df_with_targets.drop(TARGET_COLS, axis=1))
Action: result = pipe.transform(test_df_without_targets)
Assert: result.shape == (len(test_df), 41)
Assert: result.isna().sum().sum() == 0
Assert: set(result.columns) == set(FEATURE_COLS + INTERACTION_FEATURE_NAMES)
```

**INTEG-02: Serialisation round-trip**
```
Setup: fit the ColumnSelector → InteractionFeatureAdder chain
Action: pickle.dumps() then pickle.loads()
Assert: deserialized transformer produces identical output to original
Assert: cold_start_defaults_ preserved across serialisation
```

**INTEG-03: fit_transform(train) → transform(val) cold-start consistency**
```
Setup: introduce NaN in rolling features in val set (simulating cold-start)
Assert: val NaN values filled with train-set population means (not val means)
Assert: same fill values used whether val is transformed immediately or after serialisation
```

---

## 10. Constants Used by Each Class

Summary of which constants from `constants.py` each transformer directly imports:

| Constant | `ColumnSelector` | `InteractionFeatureAdder` |
|---|---|---|
| `FEATURE_COLS` | ✓ (select/validate) | ✓ (validate input) |
| `TARGET_COLS` | ✓ (reject) | — |
| `COLUMN_DTYPE_CONTRACT` | ✓ (validate dtypes) | — |
| `COLD_START_DEFAULTS` | ✓ (seed for fit-time default) | — |
| `COLD_START_FEATURE_NAMES` | ✓ (identify fillable columns) | — |
| `ZERO_VARIANCE_FEATURES` | ✓ (assert expected values) | — |
| `INTERACTION_FEATURE_NAMES` | — | ✓ (name outputs + collision check) |

Neither transformer imports `LOG_FEATURES`, `SCALE_FEATURES`, `BINARY_FEATURES`, `ORDINAL_FEATURES`, or `PASSTHROUGH_FEATURES`. Those preprocessing group constants are consumed only by `pipeline.py` when assembling the `ColumnTransformer`.

---

*Design Specification — `ml/src/mpc_ml/features/transformers.py`*  
*Manufacturing Process Copilot Technical Series*
