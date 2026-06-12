# Manufacturing Process Copilot
## Design Specification: `ml/src/mpc_ml/features/constants.py`

**Status:** Authoritative — implement exactly as specified  
**Phase:** 1, Day 3  
**Implements:** Feature engineering contract for all 37 base features + 4 interaction features

---

## Purpose of This File

`constants.py` is the single source of truth for the feature engineering contract. It defines:

- Every constant that `pipeline.py`, `transformers.py`, `service.py`, and the training scripts reference
- The preprocessing assignment for every feature
- The encoding contracts for every categorical variable
- The cold-start defaults for every rolling historical feature

No logic belongs here. This file contains only module-level declarations. It has no imports from `mpc_ml` itself — it is the root of the dependency graph within the package.

**The cost of getting this wrong:** if a feature appears in the wrong preprocessing group, the pipeline will silently produce incorrect transformations. StandardScaler applied to a binary column is not an error — it produces valid output that is simply suboptimal. These bugs do not raise exceptions; they degrade model performance invisibly. This document specifies each assignment with explicit statistical justification.

---

## Design Rule: Preprocessing Groups Are Mutually Exclusive

Every feature appears in exactly one group. The groups, in order of transformation:

```
INTERACTION_FEATURE_NAMES  →  derived FIRST (InteractionFeatureAdder step)
                               each derived feature then appears in one of:
LOG_FEATURES               →  log1p → StandardScaler
SCALE_FEATURES             →  StandardScaler only
BINARY_FEATURES            →  passthrough, no transform (0/1)
ORDINAL_FEATURES           →  passthrough, no transform (ordered int codes)
PASSTHROUGH_FEATURES       →  passthrough, no transform (small counts)
ZERO_VARIANCE_FEATURES     →  passthrough, excluded from all scalers (currently constant)
```

The ColumnTransformer in `pipeline.py` will have one named transformer per group (except ZERO_VARIANCE which is a passthrough). Together they must cover every column in `FEATURE_COLS + INTERACTION_FEATURE_NAMES`. The absence of any column from any group is a configuration error that will surface as a silent drop by the ColumnTransformer.

---

## 1. `FEATURE_COLS` — The 37 Base Features

Ordered exactly as they appear in the simulation output CSV. This order is the authoritative column sequence — the DataFrame passed to the pipeline must have these columns available (order does not matter for ColumnTransformer, but this is the canonical reference).

```
Category A: Order Planning (9)
  planned_lead_time_hours
  release_lag_hours
  schedule_revision_count
  is_expedited
  priority_encoded
  quantity
  operation_count
  estimated_total_hours
  schedule_tightness_ratio

Category B: Product Characteristics (4)
  product_complexity_score
  material_bom_complexity
  [operation_count — shared with A]
  [estimated_total_hours — shared with A]

Category C: Machine State at Release (8)
  machine_utilization_at_release
  work_center_queue_depth_at_release
  machine_oee_30d
  machine_unplanned_downtime_hours_30d
  days_since_last_planned_maintenance
  maintenance_due_within_order_window
  changeover_required
  changeover_complexity_score

Category D: Operator State at Release (5)
  operator_experience_months
  operator_skill_tier_encoded
  operator_concurrent_order_count
  hours_into_shift_at_start
  shift_type_encoded

Category E: Material State at Release (2)
  material_availability_at_release
  component_shortage_count

Category F: Historical Rolling Features (7)
  product_delay_rate_90d
  machine_delay_rate_90d
  operator_delay_rate_90d
  product_x_machine_delay_rate_90d
  product_first_pass_yield_90d
  machine_setup_overrun_rate_90d
  shift_delay_rate_30d

Category G: Temporal Features (2 + 2 flags)
  planned_start_day_of_week
  planned_start_hour
  is_month_end            [placed here in output CSV]
  is_quarter_end          [placed here in output CSV]
```

**Total: 37 base features.** The complete list used in `FEATURE_COLS` is the 37 columns listed above (with the two shared features counted once each, categories B sharing with A).

---

## 2. `TARGET_COLS` — The 4 Prediction Targets

```
is_delayed              primary binary classification  (int64, 0 or 1)
delay_minutes           regression                     (int64, 0 or positive)
delay_category          ordinal classification         (str → int label encoding)
delay_root_cause        multi-class classification     (str → int label encoding)
```

These columns are NEVER input to the feature pipeline. Any function that receives a DataFrame containing these columns must drop them before calling `pipeline.transform()`. A validation check in `ColumnSelector` should assert that no target column appears in the input.

---

## 3. `LOG_FEATURES` — Log1p Then StandardScaler

### Rationale

These are continuous variables with right-skewed distributions where the distance between small values carries more information than the distance between large values. `log1p(x)` compresses the right tail, making the distribution approximately Gaussian, after which StandardScaler centers and scales. The result is a feature where the model can learn linear relationships across the full range.

**Failing to log-transform these features** causes two problems: (1) linear models weight the high end of the range disproportionately; (2) even tree models waste splits trying to recover a linear relationship in log-space that they could split cleanly after transformation.

### Features

---

#### `planned_lead_time_hours`
- **Observed:** float64, range [8.3, 224.8], mean 30.2, median 26.8, **skew +3.57**
- **Why log:** The distribution is generated by three overlapping uniform distributions (rush 1–5 days, normal 5–15, planned 15–45). The mixture produces strong right skew. The 225-hour maximum is a 27× multiple of the 8.3-hour minimum. Without log, the upper 5% of values would dominate the feature's contribution in a linear model.
- **Post-transform:** Approximately Normal in [log(8.3), log(225)] = [2.1, 5.4], then standardized to mean ≈ 0, std ≈ 1.
- **Leakage risk:** None. Computed at order creation from `lead_days × 24`, derived from the DemandGenerator's lead time distribution. Known before any execution begins.

---

#### `release_lag_hours`
- **Observed:** float64, range [0.8, 140.0], mean 12.0, median 7.0, **skew +3.28**
- **Why log:** Generated by lognormal(1.8, 0.7) for same-day releases and lognormal(3.6, 0.5) for delayed releases. The generator is inherently log-normal — this is the only feature where log1p recovers the exact generating distribution. The strongest lagged predictor (|r|=0.478); correctly scaled, this carries maximum predictive signal.
- **Post-transform:** Near-Gaussian after log1p. Note: `release_lag_hours > planned_lead_time_hours` is theoretically possible (order released after due date); log1p handles this without clipping.
- **Leakage risk:** None. Known at release time. The `release_lag_hours` value is computed before `actual_end` is determined.
- **Implementation note:** The `lag_as_pct_of_window` interaction feature (see §8) is derived from this column before the log transform — the raw value is needed for the ratio.

---

#### `quantity`
- **Observed:** int64, range [3, 617], mean 45.8, median 33, **skew +4.08**
- **Why log:** Generated directly from lognormal(3.5, 0.8). The generator is lognormal by design. The 617-unit maximum order is a 206× multiple of the 3-unit minimum. After log1p: range becomes [log(4), log(618)] = [1.4, 6.4]. The resulting spread is approximately Gaussian.
- **Post-transform:** Approximately Normal, standardized.
- **Leakage risk:** None. Order quantity is a planning parameter determined at demand generation.
- **Interaction note:** `estimated_total_hours` is partially derived from `quantity × run_time_per_unit`. These two features carry overlapping information but different signal: `quantity` captures production volume; `estimated_total_hours` captures total processing time. Keep both.

---

#### `estimated_total_hours`
- **Observed:** float64, range [0.70, 115.1], mean 9.6, median 6.5, **skew +3.76**
- **Why log:** Product of `quantity` (lognormal) × `standard_run_time_per_unit` (lognormal) + constant setup, so the result is approximately lognormal. The 115-hour maximum is a 164× multiple of the 0.70-hour minimum. Strong right skew.
- **Post-transform:** Approximately Normal, standardized.
- **Leakage risk:** None. Computed from standard (planned) times, not actual execution times. `estimated_total_hours = (standard_setup + standard_run × quantity) / 60` — all known before execution.
- **Critical distinction:** `estimated_total_hours` uses STANDARD times. The actual execution time (which would be a leaky feature) is never stored as a feature.

---

#### `machine_unplanned_downtime_hours_30d`
- **Observed:** float64, range [0.0, 6.6], mean 1.89, median 1.50, **skew +1.10**
- **Why log:** Zero-inflated count-like variable. Some machines have had no breakdowns (0.0 hours); others have had several. The distribution is right-skewed with a mass of lower values. `log1p(0) = 0` correctly handles the zero-valued cases without requiring special treatment.
- **Post-transform:** Zero-inflated Gaussian. The zero values remain distinct (all map to log1p(0) = 0) before scaling.
- **Leakage risk:** None. Computed from historical breakdown events that completed before `release_time`. The 30-day lookback is applied with `release_time` as the strict cutoff.
- **Note:** Skewness of +1.10 is borderline (just above the +1.0 informal threshold). The zero-inflation is the deciding factor: without log1p, a StandardScaler would assign the same magnitude of importance to the difference between 0h and 1h as to the difference between 5h and 6h, which is not correct.

---

#### `operator_experience_months`
- **Observed:** int64, range [1, 176], mean 59.3, median 34, **skew +1.25**, 13 unique values
- **Why log:** Trimodal distribution with peaks at junior (1–12), mid (13–48), and senior (49–180) tiers. The senior tail is long: 176 months is nearly 15 years. The log transform compresses this tail while preserving the meaningful distinction between a 1-month and a 12-month operator (both junior, but very different skill levels in practice). After log1p: log1p(1)=0.69, log1p(12)=2.56, log1p(48)=3.87, log1p(176)=5.18 — a more even spread.
- **Post-transform:** Compressed range that respects the psychological meaning of experience (doubling from 6 to 12 months is more significant than doubling from 80 to 160 months).
- **Leakage risk:** None. Experience in months is a property of the operator at the time of assignment, known at release.
- **Note:** Only 13 unique values because the simulation initialises 24 operators once at the start with fixed experience values. This low cardinality does not change the preprocessing decision — the log transform is correct for the data-generating process even if the sample shows only 13 values.

---

### `LOG_FEATURES` Summary

| Feature | Skew | Min | Max | Key reason |
|---|---|---|---|---|
| `planned_lead_time_hours` | +3.57 | 8.3 | 224.8 | Lead time mixture distribution |
| `release_lag_hours` | +3.28 | 0.8 | 140.0 | Lognormal generator |
| `quantity` | +4.08 | 3 | 617 | Lognormal generator |
| `estimated_total_hours` | +3.76 | 0.7 | 115.1 | Product of two lognormals |
| `machine_unplanned_downtime_hours_30d` | +1.10 | 0.0 | 6.6 | Zero-inflated, right tail |
| `operator_experience_months` | +1.25 | 1 | 176 | Long senior tail |
| `lag_as_pct_of_window` *(interaction)* | +3.76 | 0.02 | 7.4 | Ratio of two log features |

---

## 4. `SCALE_FEATURES` — StandardScaler Only

### Rationale

These are continuous variables with distributions that are approximately Gaussian, bounded, or have sufficient symmetry that log transformation would provide no benefit or would distort the feature. StandardScaler (subtract mean, divide by std) centers them at zero and normalizes variance — essential for linear models and beneficial for distance-based models. Tree-based models (XGBoost, LightGBM) are invariant to monotonic transformations, so StandardScaler is harmless and makes the feature importances comparable across features with different natural scales.

### Features

---

#### `schedule_tightness_ratio`
- **Observed:** float64, range [0.176, 1.023], mean 0.60, median 0.59, **skew +0.32**
- **Why scale:** Already a dimensionless ratio approximately bounded [0, 1]. Near-symmetric distribution (skew barely above zero). `estimated_total_hours / planned_window_hours` — the ratio itself is already the natural scale. StandardScaler centers it at zero and normalizes the variance; no transformation of the relationship is needed.
- **Causal significance:** The monotonic delay-rate relationship across quartiles (30.5% → 26.1% → 39.9% → 49.8%) confirms this feature is highly informative after scaling.
- **Leakage risk:** None. Both numerator (estimated from standard times) and denominator (planned window from DemandGenerator) are known at order creation.

---

#### `product_complexity_score`
- **Observed:** float64, only 3 unique values: {0.25, 0.55, 0.85}, **skew -0.23**
- **Why scale (not ordinal):** The values are continuous proxies of an underlying complexity scale, not arbitrary labels. The spacing is equal (LOW=0.25, MED=0.55, HIGH=0.85, ∆=0.30), which encodes a meaningful quantitative distance. StandardScaler normalises this to approximately {-1.36, 0.0, +1.36}. The symmetric near-zero skew confirms no transformation is needed.
- **Alternative considered:** ORDINAL_FEATURES (passthrough). For pure tree models this makes no difference. SCALE_FEATURES is chosen to maintain a consistent preprocessing contract that also works for linear models.
- **Leakage risk:** None. Complexity is a product property, fixed at product initialisation.

---

#### `machine_utilization_at_release`
- **Observed:** float64, range [0.0, 1.0], mean 0.48, **skew +0.09**, bimodal (~50% near 0.0, ~43% near 1.0)
- **Why scale:** Bounded [0, 1]. Despite the bimodal distribution, StandardScaler is appropriate — it does not require unimodality. The model will learn from the two modes; forcing them into a log or other non-linear transform would be incorrect.
- **Causal note:** In the current simulation, the causal direction of this feature is REVERSED from the intended design (high utilisation → lower delay rate in the data, due to selection effects in machine assignment). The correlation with `is_delayed` is near zero. **Do not treat this as a strong predictor.** Keep in the pipeline because: (1) it contributes to the `tightness_x_queue` interaction, (2) future simulation iterations may correct the causal direction.
- **Leakage risk:** None. Computed from scheduled windows that existed before `release_time`.

---

#### `machine_oee_30d`
- **Observed:** float64, range [0.568, 0.785], mean 0.642, **skew +0.73**, only 8 unique values
- **Why scale:** Bounded continuous [0.35, 0.92] by design. The 8 unique values reflect 8 machines with fixed OEE values. StandardScaler normalises to approximately [-1.1, +2.0] range. The mild positive skew does not warrant log transformation.
- **Important note:** With only 8 unique values, this feature is effectively an ordinal machine identifier. In a production deployment with a real factory, this feature would have continuous variation. The current low cardinality is a simulation artifact. Treat as continuous.
- **Leakage risk:** None. OEE is computed from completed orders strictly before `release_time`.

---

#### `days_since_last_planned_maintenance`
- **Observed:** float64, range [0, 89], mean 24.1, median 22.0, **skew +0.92**
- **Why scale:** Bounded [0, 90] by the PM interval design (30–90 days). The distribution is roughly uniform within the PM interval, with a mild right skew from machines that are overdue. StandardScaler normalises the range. Log would not help — the risk relationship with delay is near-linear (each additional day of wear increases failure probability proportionally in the Weibull model).
- **Interaction note:** This feature participates in the `oee_x_maintenance_ratio` interaction feature. Both the raw value and the interaction term are kept.
- **Leakage risk:** None. Computed from `release_time - machine.last_maintenance_date`.

---

#### `changeover_complexity_score`
- **Observed:** float64, range [1.0, 2.999], mean 1.646, **skew +0.46**, bimodal (1.0 when no changeover, Uniform[1.5, 3.0] when changeover)
- **Why scale:** Bounded [1.0, 3.0]. The bimodal distribution (1.0 for 51% of orders with no changeover, continuous [1.5, 3.0] for 49% with changeover) is handled correctly by StandardScaler. Tree models will find the threshold at ~1.25 naturally.
- **Do not use alone:** This feature must always be used alongside `changeover_required`. A complexity score of 1.0 could mean "no changeover" or "trivial changeover" — the binary flag disambiguates.
- **Leakage risk:** None. Changeover is detected from the last completed product on the machine at release time.

---

#### `hours_into_shift_at_start`
- **Observed:** float64, range [0.0, 7.0], mean 3.4, **skew +0.03**, 8 unique integer values
- **Why scale:** Bounded [0, 7.5] (one shift). Near-symmetric (skew ≈ 0). StandardScaler normalises the range. The 8 integer values {0, 1, 2, 3, 4, 5, 6, 7} represent hours-into-shift — a continuous concept with integer granularity.
- **Leakage risk:** None. `planned_start` hour relative to shift start is known at order creation.

---

#### `product_delay_rate_90d`
- **Observed:** float64, range [0.0, 1.0], mean 0.343, **skew +0.66**
- **Why scale:** Bounded rate [0, 1]. Skew +0.66 is within the scale-only threshold. StandardScaler centres it around 0.
- **Leakage risk: COLD-START.** For new products with fewer than 3 completed orders in the 90-day window, this feature is set to a population default (≈0.35). **This is not data leakage** — it is a known approximation. However, the feature distribution shifts over the simulation: early orders (days 0–90) will cluster at the default value, while later orders have true historical rates. The train/val/test temporal split ensures the model is evaluated on orders with richer historical data, matching the production deployment scenario.
- **Multicollinearity note:** Pearson |r| ≈ 0.94 with `product_x_machine_delay_rate_90d`. Both are kept for tree models (which handle correlated features via feature importance averaging across trees). For linear models or ElasticNet, drop one.

---

#### `machine_delay_rate_90d`
- **Observed:** float64, range [0.083, 0.70], mean 0.347, **skew +0.12**
- **Why scale:** Bounded rate, symmetric (skew near zero). StandardScaler appropriate.
- **Leakage risk: COLD-START** (same as above). The cold-start window is shorter for machines than for products — each machine processes more orders per day.
- **Note:** The lower bound 0.083 (not 0.0) exists because by the time any order is scored, each machine has at least 1 completed order. This is a feature of the simulation's warm-up period, not a problem.

---

#### `operator_delay_rate_90d`
- **Observed:** float64, range [0.0, 0.667], mean 0.354, **skew -0.91**
- **Why scale:** Bounded rate, mild negative skew (some very good operators at rate near 0). StandardScaler appropriate.
- **Leakage risk: COLD-START + HIGH VARIANCE.** Operators process fewer orders than machines. A single operator may have only 3–10 completed orders in 90 days, making this rate high-variance and noisy. The model should weight this less than `machine_delay_rate_90d` — tree models will learn this from feature importance.

---

#### `product_x_machine_delay_rate_90d`
- **Observed:** float64, range [0.0, 1.0], mean 0.342, **skew +0.72**
- **Why scale:** Bounded rate. Similar distribution to `product_delay_rate_90d`.
- **Leakage risk: HIGH COLD-START RISK.** The product × machine combination is sparse — most pairs appear fewer than 5 times in 90 days. Cold-start default is used frequently. This means the feature provides genuine signal only for common product-machine pairs.
- **Multicollinearity:** |r| ≈ 0.94 with `product_delay_rate_90d`. Retain for tree models; drop for linear models.

---

#### `product_first_pass_yield_90d`
- **Observed:** float64, range [0.50, 1.0], mean 0.916, median 0.929, **skew -1.22**
- **Why scale:** Bounded rate in [0.5, 1.0]. Mild negative skew — most products have near-perfect FPY. StandardScaler will amplify variation in the low end (0.50–0.85), which is exactly where the signal is (poor FPY → quality failure → rework → delay).
- **Note:** The narrow range means small absolute differences carry large signals. An FPY of 0.80 vs 0.93 is a 16% difference in failure rate.
- **Leakage risk: COLD-START** (same mechanism as other historical features).

---

#### `machine_setup_overrun_rate_90d`
- **Observed:** float64, range [0.0, 0.833], mean 0.521, **skew -0.91**
- **Why scale:** Bounded rate with mild negative skew. StandardScaler appropriate.
- **Note:** The threshold used in this feature (1.5× standard) differs from the threshold used in the simulation for root cause attribution (1.7× standard). This is intentional — a softer threshold for the feature captures the leading edge of overrun risk; the harder threshold for the cause ensures only significant overruns are attributed.
- **Leakage risk: COLD-START** (same mechanism).

---

#### `shift_delay_rate_30d`
- **Observed:** float64, range [0.20, 0.43], mean 0.358, median 0.357, **skew -1.12**
- **Why scale:** Bounded rate in narrow range [0.20, 0.43]. Mild negative skew. StandardScaler will amplify the variation across the 0.23-point range — appropriate since small differences in shift-level delay rates represent meaningful operational changes.
- **Low predictive value note:** This is a factory-wide aggregate with low standalone signal. However, it provides a useful baseline shift effect for orders where individual-level features are sparse (early in simulation). Keep.
- **Leakage risk: MILD COLD-START.** The 30-day window is shorter, so stabilises faster. Fully populated after day 30.

---

#### `planned_start_hour`
- **Observed:** int64, range [4, 23], mean 13.7, **skew +0.04**, 20 unique values
- **Why scale:** Nearly uniform distribution across operating hours (4:00 to 23:00). Skew ≈ 0. StandardScaler normalises the [4, 23] range to approximately [-1.7, +1.7]. This is appropriate for linear models. Tree models split on the raw integer values equally well without scaling, but the overhead is negligible.
- **Alternative considered:** Cyclical encoding (sin/cos). For strict tree models this provides no benefit; for the linear baseline model it would correctly encode the 23→4 wrap-around (end of night shift → start of morning shift). If linear models are used in production, add a `cyclical_hour` feature derived here.
- **Leakage risk:** None. `planned_start.hour` is a property of the order's planned execution time.

---

#### `tightness_x_queue` *(interaction)*
- **Observed:** float64, range [0.0, 1.016], mean 0.41, **skew -0.42**, corr with is_delayed **-0.399**
- **Why scale:** Bounded product of two [0, 1] features. Near-symmetric skew.
- **Directionality note:** Negative correlation with is_delayed is EXPECTED given that `work_center_queue_depth_at_release` has a negative correlation with delay in the current simulation (a known calibration artifact). The signal is strong (|r|=0.40) even if the direction is the reverse of the causal theory. The model will learn the correct direction.

---

#### `oee_x_maintenance_ratio` *(interaction)*
- **Observed:** float64, range [0.207, 1.569], mean ~0.8, **skew -0.01**, corr with is_delayed **+0.003**
- **Why scale:** Bounded continuous product. Symmetric.
- **Critical note:** Correlation with is_delayed is near zero (+0.003). **This interaction feature carries essentially no predictive signal in the current simulation.** The near-zero correlation arises because `machine_oee_30d` itself has only 8 unique values and near-zero correlation with is_delayed, and the maintenance recency factor amplifies a signal that doesn't exist. Include in the pipeline for completeness, but flag for elimination in feature selection. The `run_study()` Optuna function should include a boolean hyperparameter `use_maintenance_interaction` that the study can set to False.
- **Why include despite near-zero signal:** The correlation may improve in the 540-day full run where machine state variation accumulates more. Include as a placeholder; Optuna's feature importance will eliminate it if it remains uninformative.

---

## 5. `BINARY_FEATURES` — Passthrough (No Transform)

### Rationale

Binary features (0/1) carry boolean information. StandardScaler on a binary column would produce {−skew_left, +skew_right} float values that do not improve model performance. Log1p(0) = 0 and log1p(1) = 0.693 — this would incorrectly treat 0 as the absence of the feature and 1 as log-scale distance from zero, which has no semantic meaning for a flag.

Tree models split on binary features by checking `value <= 0.5`. This is already the optimal split. No transformation needed or beneficial.

For linear/logistic regression models: binary features are numerically equivalent to dummy-encoded features, so they are correctly handled without one-hot encoding.

### Features

---

#### `schedule_revision_count`
- **Observed:** float64, {0.0, 1.0}, mean 0.031, **skew +5.38** (extreme skew because 96.9% are 0)
- **Why binary (not log or passthrough):** Currently only two values: 0 (order not revised) and 1 (order revised). The extreme skew is a property of the sparse flag, not a distribution to be transformed.
- **Semantic note:** This is a count that could be >1 in a real production system. In the current simulation it is always 0 or 1. The BINARY_FEATURES group is correct for now; if future simulation versions produce values >1, move to PASSTHROUGH_FEATURES.
- **Leakage risk:** None. Revision count is determined at order creation time.

---

#### `is_expedited`
- **Observed:** int64, {0, 1}, mean 0.031, **skew +5.38**
- **Why binary:** Binary flag by definition. Set for ~35% of critical-priority orders (3.1% overall).
- **Note:** Perfectly correlated with `schedule_revision_count` in the current simulation (both fire for the same expedited orders). In `pipeline.py`, the ColumnTransformer should process both. During model evaluation, if these two features are perfectly correlated, the model should either drop one or the feature selection step will do it automatically.
- **Leakage risk:** None. Expedite status is determined at order creation.

---

#### `is_month_end`
- **Observed:** int64, {0, 1}, mean 0.227, **skew +1.30**
- **Why binary:** Binary flag for the last 4 days of each calendar month. Skew +1.30 is the natural skew of a Bernoulli(0.23) variable.
- **Semantic meaning:** Month-end demand surge (1.35× multiplier in DemandGenerator) creates congestion. The feature is a proxy for this congestion — it does not directly cause delays but correlates with the higher-demand periods.
- **Leakage risk:** None. Calendar position is known at order creation.

---

#### `is_quarter_end`
- **Observed:** int64, {0, 1}, mean 0.096, **skew +2.74**
- **Why binary:** Binary flag for the last 4 days of each quarter. Rarer than month-end.
- **Note:** A subset of `is_month_end` (quarter-end days are always month-end). These are not mutually exclusive but both are kept because quarter-end carries a stronger demand surge signal (1.70× vs 1.35× multiplier). The model will learn that `is_quarter_end=1 AND is_month_end=1` carries the strongest congestion risk.
- **Leakage risk:** None.

---

#### `work_center_queue_depth_at_release`
- **Observed:** float64, **confirmed binary {0.0, 1.0}**, mean 0.688, **skew -0.81**
- **Why binary:** Despite being named "depth" (implying a count), the current implementation produces exactly two values: 0.0 (no orders waiting) and 1.0 (orders are waiting). The statistics confirm this: std = 0.464 matches exactly with `sqrt(0.688 × 0.312) = 0.463`.
- **Causal note:** This is the **strongest predictor in the dataset** (|r|=0.502 with `is_delayed`). However, the sign is counterintuitive: queue=1 is associated with LOWER delay rates in the current simulation (the utilisation causal check fails). This is a simulation calibration artifact — in the real system, a non-empty queue should increase delay probability. The negative correlation is a real signal the model can use; the direction simply reflects how machine assignment and scheduling interact in the simulation.
- **Future consideration:** When the queue depth model is enhanced to return a continuous depth value (0, 1, 2, ...), move from BINARY_FEATURES to SCALE_FEATURES (with possible log1p if skewed).
- **Leakage risk:** None. Queue state is computed from historical scheduled windows at `release_time`.

---

#### `maintenance_due_within_order_window`
- **Observed:** int64, {0, 1}, mean 0.014, **skew +8.40** (extremely rare: 1.4% positive)
- **Why binary:** Binary flag by definition. The extreme skew reflects its rarity.
- **Rarity caveat:** With only ~14 positive examples in 954 orders (120-day run), this feature is statistically noisy for any model that requires sufficient positive examples per split. In the 540-day run (~4,293 orders), expect ~60 positive examples — marginally sufficient. Include in the pipeline; weight by `class_weight` parameters in tree models to handle class imbalance.
- **Leakage risk:** None. PM due date is computed from machine maintenance history at release time.

---

#### `changeover_required`
- **Observed:** int64, {0, 1}, mean 0.510, **skew -0.04** (approximately balanced)
- **Why binary:** Binary flag by definition. Nearly balanced (51% positive) — no rarity concern.
- **Critical coupling:** Always use with `changeover_complexity_score`. The complexity score is 1.0 when `changeover_required = 0` — models must learn this joint structure. The tree model will discover it via splits. Linear models may benefit from the explicit interaction term `changeover_required × changeover_complexity_score`, but this interaction is not in `INTERACTION_FEATURE_NAMES` because the product would be identical to `changeover_complexity_score` when changeover=1 and 1.0 when changeover=0 — redundant with the original feature.
- **Leakage risk:** None. Changeover is detected from the last completed product on the machine at `release_time`.

---

#### `material_availability_at_release`
- **Observed:** int64, {0, 1}, mean 0.881, **skew -2.35** (88.1% positive = materials available)
- **Why binary:** Binary flag by definition. Encoding: 0 = shortage, 1 = available (note: 0 is the **dangerous** state despite being the lower value — tree models handle this correctly regardless of encoding direction).
- **Highest operational value:** Delay rate 69.3% (shortage) vs 32.1% (available) — the **largest absolute gap** of any feature. This single binary feature is the most actionable predictor: a proactive material check before release can reduce delays by preventing the 69.3% case.
- **Causal path confirmed:** Material shortage → extra_hours accumulation in simulation → `actual_end > planned_end` = delay. Direct, verified causal relationship.
- **Leakage risk:** None. Material availability is determined at order release time before any execution begins.

---

### `BINARY_FEATURES` Summary

| Feature | Positive rate | |r| with is_delayed | Notes |
|---|---|---|---|
| `schedule_revision_count` | 3.1% | ~0.09 | Currently same as is_expedited |
| `is_expedited` | 3.1% | 0.088 | Correlated with schedule_revision_count |
| `is_month_end` | 22.7% | ~0.02 | Calendar demand proxy |
| `is_quarter_end` | 9.6% | ~0.01 | Stronger demand proxy |
| `work_center_queue_depth_at_release` | 68.8% | 0.502 | Strongest predictor, reversed direction |
| `maintenance_due_within_order_window` | 1.4% | ~0.04 | Very rare; needs 540-day run for signal |
| `changeover_required` | 51.0% | ~0.02 | Use with changeover_complexity_score |
| `material_availability_at_release` | 88.1% | 0.250 | Highest operational value |

---

## 6. `ORDINAL_FEATURES` — Passthrough (Ordered Integer Codes, Tree Models)

### Rationale

These are low-cardinality variables with a meaningful natural ordering. For tree-based models (XGBoost, LightGBM — the primary production models), passing through ordinal integers is optimal: the model can find threshold splits at any point along the ordinal scale and implicitly discovers the monotonic relationship without any encoding overhead.

**Warning for linear models:** For logistic regression or linear regression baselines, ordinal features with 3+ levels should be one-hot encoded (losing the ordinal structure) or target-encoded. The constants file should note this; the pipeline implementation must handle the model-type flag.

### Features

---

#### `priority_encoded`
- **Observed:** int64, {1, 2, 3}, mean 1.525, **skew +0.90**, 3 values
- **Encoding:** 1=normal, 2=high, 3=critical
- **Why ordinal:** The order is meaningful and causal. Priority directly determines the planning buffer (1.80× for normal, 1.50× for high, 1.15× for critical), which determines `schedule_tightness_ratio`. Delay rates: normal=31%, high=37%, critical=67% — monotonically increasing with priority value. The spacing between levels is NOT equal (the critical jump is much larger than the normal→high jump), but ordinal encoding lets the model discover this asymmetry.
- **Distribution note:** ~57% normal, ~33% high, ~10% critical (120-day run).
- **Leakage risk:** None. Priority is a planning parameter determined at demand generation.
- **Note:** The value 0 (low priority) exists in the encoding but appears rarely (not shown in 120-day run). The constant `PRIORITY_ENCODING` (see §10) must include the `low: 0` mapping for completeness.

---

#### `operator_skill_tier_encoded`
- **Observed:** float64, {0.0, 1.0, 2.0}, mean 1.055, **skew -0.03**, 3 values
- **Encoding:** 0=junior, 1=mid, 2=senior
- **Why ordinal:** The order is meaningful (junior < mid < senior) and correlated with `setup_speed_multiplier` (1.12 < 1.05 < 0.95 — inverse, but monotone). For tree models, splitting at `operator_skill_tier_encoded <= 0.5` isolates junior operators; splitting at `<= 1.5` separates mid from senior.
- **Causal nuance:** Delay rates by tier are 33.6% (junior), 38.3% (mid), 33.8% (senior). The MID tier has the HIGHEST delay rate, not the junior tier. This is a simulation artifact: mid-tier operators are assigned to more orders (they are the plurality at 62.7%) and encounter more varied conditions, including more overloaded machines. The ordinal encoding remains correct — the model will discover the non-monotonic delay rate pattern through tree splits.
- **Leakage risk:** None. Skill tier is a property of the operator at the time of assignment.

---

#### `shift_type_encoded`
- **Observed:** int64, {0, 1, 2}, mean 0.789, **skew +0.37**, 3 values
- **Encoding:** 0=morning (06:00–14:00), 1=afternoon (14:00–22:00), 2=night (22:00–06:00)
- **Why ordinal:** The ordering morning→afternoon→night has a weak causal meaning (decreasing supervision, increasing absenteeism for night shifts). The model can discover any non-linear pattern via tree splits regardless of ordinal encoding.
- **Alternative considered:** One-hot encoding. For strict ordinal models (proportional odds), one-hot is more correct. For tree models, ordinal passthrough is equivalent.
- **Leakage risk:** None. Shift is a property of the operator's assignment, known at release.

---

#### `planned_start_day_of_week`
- **Observed:** float64, {0.0, 1.0, 2.0, 3.0, 4.0}, mean 1.855, **skew +0.13**, 5 values
- **Encoding:** 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
- **Why ordinal:** The ordering Mon→Fri has a weak but meaningful cadence (Monday orders have the most remaining work-week buffer; Friday orders face weekend disruption). The model can find non-linear patterns (e.g., Friday threshold) through tree splits.
- **Alternative considered:** Cyclical encoding. The week has no wrap-around continuity (Monday is NOT adjacent to Friday in the factory calendar — weekends break the cycle). Ordinal passthrough is correct.
- **Leakage risk:** None. Day of week is determined at order creation.

---

### `ORDINAL_FEATURES` Summary

| Feature | Values | Encoding | Natural ordering |
|---|---|---|---|
| `priority_encoded` | {1, 2, 3} | low→critical | Strongly monotonic with delay rate |
| `operator_skill_tier_encoded` | {0, 1, 2} | junior→senior | Weak (mid tier anomaly) |
| `shift_type_encoded` | {0, 1, 2} | morning→night | Weak |
| `planned_start_day_of_week` | {0, 1, 2, 3, 4} | Mon→Fri | Weak |

---

## 7. `PASSTHROUGH_FEATURES` — No Transform (Small Counts and Zero-Variance Placeholders)

### Rationale

These are small-range integer counts where applying StandardScaler would normalize already-small values into a range indistinguishable from noise, or where the feature is currently constant (zero variance) and StandardScaler would fail with NaN.

Tree models split on exact integer values natively: a split at `operation_count <= 3` has direct semantic meaning. Scaling to `operation_count_scaled <= 0.23` does not change the model structure but reduces interpretability.

### Features

---

#### `operation_count`
- **Observed:** int64, range [2, 6], mean 3.595, **skew +0.54**, 5 unique values
- **Why passthrough:** Small count in [2, 6]. Tree models split at integer thresholds; StandardScaler would give approximate values [-1.5, -0.6, 0.3, 1.2, 2.1] — no advantage over the raw integers {2, 3, 4, 5, 6}.
- **Semantic meaning:** Direct count of routing operations. The threshold `operation_count >= 4` cleanly separates HIGH complexity (4–7 ops) from LOW/MEDIUM (1–5 ops).
- **Leakage risk:** None. Routing length is a product property.

---

#### `material_bom_complexity`
- **Observed:** int64, range [2, 9], mean 5.108, **skew +0.14**, 8 unique values
- **Why passthrough:** Integer count in [2, 9] with near-symmetric distribution. The raw integer values have direct interpretability (number of BOM line items). Tree splits at `material_bom_complexity <= 4` have semantic meaning.
- **Note:** This feature has low standalone predictive value. Its main contribution is as a proxy for the probability of a material shortage — more BOM lines means more opportunities for any single component to be unavailable. Most signal flows through `material_availability_at_release` and `component_shortage_count`.
- **Leakage risk:** None. BOM complexity is a product property.

---

#### `component_shortage_count`
- **Observed:** float64, {0.0, 1.0, 2.0}, mean 0.161, **skew +2.95**, 3 unique values
- **Why passthrough:** Three-value count {0, 1, 2}. Despite the high skew (89% zeros), this is NOT a binary feature — the values 1 and 2 represent meaningfully different shortage severities. Applying log1p gives {0, 0.693, 1.099} — the spacing changes but both orderings and tree splits remain equivalent. Passthrough maintains interpretability.
- **Note:** Mean 0.161 reflects ~11.8% of orders having a shortage, with an average of 1.4 components missing when a shortage occurs.
- **Coupling note:** Always zero when `material_availability_at_release = 1`. The joint structure is:
  - `material_availability = 1` → `component_shortage_count = 0` (always)
  - `material_availability = 0` → `component_shortage_count` ∈ {1, 2}
  The tree model will discover this constraint naturally.
- **Leakage risk:** None.

---

#### `operator_concurrent_order_count` ⚠️ ZERO VARIANCE
- **Observed:** float64, {0.0} — **single value, zero variance**
- **Why passthrough:** This feature is ALL ZEROS in the current simulation. StandardScaler on a zero-variance column produces NaN (division by zero in `(x - mean) / std = (0 - 0) / 0`). It MUST be excluded from any StandardScaler step.
- **Critical implementation requirement:** This feature must be in `ZERO_VARIANCE_FEATURES` (a subset of PASSTHROUGH) and the pipeline's `ColumnTransformer` must include it in the passthrough branch using `remainder='passthrough'` or an explicit passthrough transformer. **Excluding it from the DataFrame would break the serving contract** — the feature must be present and zero in production inference.
- **Semantic meaning:** Count of other orders the operator is running in parallel. Currently zero because the simulation assigns one operator per order. In future model versions (multi-machine operators), this becomes non-zero and should be moved to SCALE_FEATURES.
- **Validation requirement:** The pipeline's `transform()` method must include a runtime assertion: `assert (X['operator_concurrent_order_count'] == 0).all(), "Expected zero-variance column"`. This assertion should be configurable for the future multi-machine scenario.
- **Leakage risk:** None.

---

### `PASSTHROUGH_FEATURES` Summary

| Feature | Type | Range | Values | Notes |
|---|---|---|---|---|
| `operation_count` | int64 | [2, 6] | 5 unique | Semantic integer thresholds |
| `material_bom_complexity` | int64 | [2, 9] | 8 unique | Low standalone signal |
| `component_shortage_count` | float64 | {0, 1, 2} | 3 unique | Coupled with material_availability |
| `operator_concurrent_order_count` | float64 | {0} | 1 unique | ZERO VARIANCE — special handling required |

---

## 8. `INTERACTION_FEATURE_NAMES` — Derived Before ColumnTransformer

### Design

Interaction features are computed by `InteractionFeatureAdder` (the first step in the `Pipeline`, before the `ColumnTransformer`). They are derived from the base 37 features and must be named so they do not collide with any existing column name. After derivation, each goes into one of the preprocessing groups above.

### Features

---

#### `lag_as_pct_of_window`
- **Formula:** `release_lag_hours / max(0.1, planned_lead_time_hours)`
- **Semantic meaning:** What fraction of the total lead time has already been consumed by the release lag before manufacturing even starts. Values > 1.0 mean the order was released AFTER its planned due date — virtually guaranteeing a delay.
- **Observed on validation data:** range [0.016, 7.4], mean ~0.43, **skew +3.76**, **corr with is_delayed +0.521**
- **This is the strongest predictor in the dataset** — higher correlation than any base feature. The reason: it compresses the two most predictive base features (`release_lag_hours` and `planned_lead_time_hours`) into a single ratio that directly quantifies the proportion of lead time wasted before work begins.
- **Preprocessing group:** `LOG_FEATURES` — strong right skew (+3.76), values span [0.016, 7.4]
- **Implementation note:** Denominator clipped to minimum 0.1 to prevent division by zero, though in practice `planned_lead_time_hours >= 8.3` by construction.

---

#### `tightness_x_queue`
- **Formula:** `schedule_tightness_ratio × work_center_queue_depth_at_release`
- **Semantic meaning:** Orders are tight AND the queue is non-empty — compounding congestion risk. When both `schedule_tightness_ratio` is high (tight schedule) and `work_center_queue_depth_at_release = 1` (queue exists), the multiplicative effect should identify orders at highest congestion risk.
- **Observed on validation data:** range [0.0, 1.016], **skew -0.42**, **corr with is_delayed -0.399**
- **Directionality note:** Negative correlation reflects the reversed direction of `work_center_queue_depth_at_release` in the current simulation. The signal strength (|r|=0.40) is valid regardless of sign — the model learns the correct direction from data.
- **Preprocessing group:** `SCALE_FEATURES` — bounded [0, 1], symmetric distribution

---

#### `log_experience_x_concurrent`
- **Formula:** `log1p(operator_experience_months) × operator_concurrent_order_count`
- **Semantic meaning:** Concurrent workload weighted by operator seniority — a senior operator managing 3 orders simultaneously is less at risk than a junior operator doing the same. Currently ALL ZEROS because `operator_concurrent_order_count = 0`.
- **Observed on validation data:** {0.0} only — **ZERO VARIANCE**, NaN correlation
- **Preprocessing group:** ZERO_VARIANCE_FEATURES (passthrough with assertion)
- **Future readiness:** When multi-machine operator scheduling is implemented, this feature becomes `log1p(experience) × concurrent_count` — a meaningful interaction. The constant must be retained so the feature exists in the pipeline and the serving schema.

---

#### `oee_x_maintenance_ratio`
- **Formula:** `machine_oee_30d × (1.0 / max(1.0, days_since_last_planned_maintenance / 30.0))`
- **Semantic meaning:** OEE adjusted for maintenance recency — a machine with OEE 0.65 that is overdue for PM (120 days since last maintenance, interval 90 days) poses more breakdown risk than a machine with OEE 0.65 that was serviced yesterday.
- **Observed on validation data:** range [0.207, 1.569], **skew -0.01**, **corr with is_delayed +0.003**
- **Near-zero signal:** The correlation is essentially zero in the 120-day run. The likely cause is that `machine_oee_30d` has only 8 unique values (one per machine) and very low marginal correlation with delay. The maintenance recency factor alone also has near-zero correlation. The combination does not improve signal.
- **Decision:** Include in pipeline but add to `CANDIDATE_REMOVAL_FEATURES` set. Optuna's feature selection should test with and without this interaction. If the 540-day run shows persistent near-zero importance, drop from the pipeline before production.
- **Preprocessing group:** `SCALE_FEATURES` — bounded [0.2, 1.6], near-symmetric

---

## 9. `TARGET_COLS` — Complete Specification

### `is_delayed`
- **Type:** int64, {0, 1}
- **Task:** Binary classification
- **Base rate:** 36.6% positive (120-day run)
- **Encoding:** Already numeric. No label encoding needed.
- **Class weight:** `{0: 1.0, 1: 1.73}` (balanced) or `scale_pos_weight = 1.73` for XGBoost/LightGBM.

### `delay_minutes`
- **Type:** int64, [0, 7590]
- **Task:** Regression (zero-inflated: 63.4% are zero)
- **Transform for training:** `log1p(delay_minutes)` — apply to the target before regression; reverse with `expm1` after prediction
- **Framing decision:** Train regression on delayed orders only (`delay_minutes > 0`), or use zero-inflated regression. The two-stage approach (first predict `is_delayed`, then predict `delay_minutes | delayed`) avoids the zero mass problem.

### `delay_category`
- **Type:** str → ordinal int
- **Task:** Ordinal classification
- **Label encoding order:** on_time=0, minor_delay=1, moderate_delay=2, major_delay=3, critical_delay=4
- **Class weights:** Balanced (critical_delay is rare: 4.8% of orders).

### `delay_root_cause`
- **Type:** str → multi-class int
- **Task:** Multi-class classification
- **Classes:** 7 (see `ROOT_CAUSE_CLASSES` below)
- **Strategy:** Train only on delayed orders where `delay_root_cause != 'none'`. The 'none' class represents orders delayed by pure variance — the root cause classification is less meaningful for this group.

---

## 10. Additional Constants Required in This File

Beyond the preprocessing groups, `constants.py` must export these constants to complete the module's contract.

### `DELAY_CATEGORY_ORDER`
The ordered list of delay category labels for ordinal encoding. Derived from confirmed simulation output (120-day run).

```
['on_time', 'minor_delay', 'moderate_delay', 'major_delay', 'critical_delay']
```

Threshold mapping confirmed:
- on_time: delay_minutes = 0
- minor_delay: 1–60 minutes
- moderate_delay: 61–480 minutes
- major_delay: 481–1,440 minutes
- critical_delay: > 1,440 minutes

### `ROOT_CAUSE_CLASSES`
The 7 class labels for the root cause classifier. Alphabetical ordering is the recommended label encoding baseline.

```
['machine_breakdown', 'material_unavailability', 'multiple_causes',
 'none', 'planning_schedule_conflict', 'quality_failure_rework', 'setup_overrun']
```

### `PRIORITY_ENCODING`
```
{'low': 0, 'normal': 1, 'high': 2, 'critical': 3}
```
Note: `low` is rare in the simulation but must be present in the encoding to handle production inference on any real order.

### `SKILL_TIER_ENCODING`
```
{'junior': 0, 'mid': 1, 'senior': 2}
```

### `SHIFT_ENCODING`
```
{'morning': 0, 'afternoon': 1, 'night': 2}
```

### `COLD_START_DEFAULTS`
Default values injected for rolling historical features when fewer than 3 historical orders exist in the lookback window. These are the population means observed in the 120-day validation run.

```python
{
    'product_delay_rate_90d':          0.343,
    'machine_delay_rate_90d':          0.347,
    'operator_delay_rate_90d':         0.354,
    'product_x_machine_delay_rate_90d': 0.342,
    'product_first_pass_yield_90d':    0.916,
    'machine_setup_overrun_rate_90d':  0.521,
    'shift_delay_rate_30d':            0.358,
}
```

**Implementation note:** These defaults must be computed from the TRAINING set only during `pipeline.fit()`. The values above are derived from the 120-day validation run; the 540-day full run may produce slightly different population means. The pipeline should compute these as a fit-time step, not hardcode the values from this document.

### `ZERO_VARIANCE_FEATURES`
Features that are currently constant and must be excluded from any variance-based transformation.

```
['operator_concurrent_order_count', 'log_experience_x_concurrent']
```

### `CANDIDATE_REMOVAL_FEATURES`
Features with near-zero empirical predictive value that should be tested for removal during Optuna tuning.

```
['oee_x_maintenance_ratio', 'shift_delay_rate_30d', 'is_quarter_end',
 'material_bom_complexity', 'operation_count']
```

These are candidates, not confirmed removals. Optuna will determine whether each improves or hurts validation AUC.

### `COLUMN_DTYPE_CONTRACT`
Expected dtypes for validation in `ColumnSelector.transform()`. Any column arriving with the wrong dtype is a sign of a pipeline bug or schema drift.

```python
{
    'planned_lead_time_hours':              'float64',
    'release_lag_hours':                    'float64',
    'schedule_revision_count':              'float64',
    'is_expedited':                         'int64',
    'priority_encoded':                     'int64',
    'quantity':                             'int64',
    'operation_count':                      'int64',
    'estimated_total_hours':                'float64',
    'schedule_tightness_ratio':             'float64',
    'product_complexity_score':             'float64',
    'material_bom_complexity':              'int64',
    'is_month_end':                         'int64',
    'is_quarter_end':                       'int64',
    'machine_utilization_at_release':       'float64',
    'work_center_queue_depth_at_release':   'float64',
    'machine_oee_30d':                      'float64',
    'machine_unplanned_downtime_hours_30d': 'float64',
    'days_since_last_planned_maintenance':  'float64',
    'maintenance_due_within_order_window':  'int64',
    'changeover_required':                  'int64',
    'changeover_complexity_score':          'float64',
    'operator_experience_months':           'int64',
    'operator_skill_tier_encoded':          'float64',
    'operator_concurrent_order_count':      'float64',
    'hours_into_shift_at_start':            'float64',
    'shift_type_encoded':                   'int64',
    'material_availability_at_release':     'int64',
    'component_shortage_count':             'float64',
    'product_delay_rate_90d':               'float64',
    'machine_delay_rate_90d':               'float64',
    'operator_delay_rate_90d':              'float64',
    'product_x_machine_delay_rate_90d':     'float64',
    'product_first_pass_yield_90d':         'float64',
    'machine_setup_overrun_rate_90d':       'float64',
    'shift_delay_rate_30d':                 'float64',
    'planned_start_day_of_week':            'float64',
    'planned_start_hour':                   'int64',
}
```

---

## 11. Complete Allocation Table

Every feature, its group, and the statistical basis for the assignment.

| Feature | Group | Basis |
|---|---|---|
| `planned_lead_time_hours` | LOG | skew=+3.57, 27× range |
| `release_lag_hours` | LOG | skew=+3.28, lognormal generator |
| `schedule_revision_count` | BINARY | {0,1}, skew=+5.38 |
| `is_expedited` | BINARY | {0,1}, definition |
| `priority_encoded` | ORDINAL | {1,2,3}, causal ordering |
| `quantity` | LOG | skew=+4.08, lognormal generator |
| `operation_count` | PASSTHROUGH | {2..6}, 5 integer values |
| `estimated_total_hours` | LOG | skew=+3.76, product of two lognormals |
| `schedule_tightness_ratio` | SCALE | skew=+0.32, bounded [0.18, 1.02] |
| `product_complexity_score` | SCALE | skew=-0.23, 3 discrete values |
| `material_bom_complexity` | PASSTHROUGH | {2..9}, 8 integer values |
| `is_month_end` | BINARY | {0,1}, definition |
| `is_quarter_end` | BINARY | {0,1}, definition |
| `machine_utilization_at_release` | SCALE | skew=+0.09, bounded [0,1] |
| `work_center_queue_depth_at_release` | BINARY | {0.0,1.0}, confirmed binary |
| `machine_oee_30d` | SCALE | skew=+0.73, bounded [0.57, 0.79] |
| `machine_unplanned_downtime_hours_30d` | LOG | skew=+1.10, zero-inflated |
| `days_since_last_planned_maintenance` | SCALE | skew=+0.92, bounded [0, 89] |
| `maintenance_due_within_order_window` | BINARY | {0,1}, definition |
| `changeover_required` | BINARY | {0,1}, definition |
| `changeover_complexity_score` | SCALE | skew=+0.46, bounded [1, 3] |
| `operator_experience_months` | LOG | skew=+1.25, trimodal right tail |
| `operator_skill_tier_encoded` | ORDINAL | {0,1,2}, causal ordering |
| `operator_concurrent_order_count` | PASSTHROUGH | {0} ZERO VARIANCE |
| `hours_into_shift_at_start` | SCALE | skew=+0.03, bounded [0, 7] |
| `shift_type_encoded` | ORDINAL | {0,1,2}, ordered shifts |
| `material_availability_at_release` | BINARY | {0,1}, strongest causal signal |
| `component_shortage_count` | PASSTHROUGH | {0,1,2}, 3 values |
| `product_delay_rate_90d` | SCALE | skew=+0.66, bounded [0,1] |
| `machine_delay_rate_90d` | SCALE | skew=+0.12, bounded [0.08, 0.70] |
| `operator_delay_rate_90d` | SCALE | skew=-0.91, bounded [0, 0.67] |
| `product_x_machine_delay_rate_90d` | SCALE | skew=+0.72, bounded [0,1] |
| `product_first_pass_yield_90d` | SCALE | skew=-1.22, bounded [0.50, 1.0] |
| `machine_setup_overrun_rate_90d` | SCALE | skew=-0.91, bounded [0, 0.83] |
| `shift_delay_rate_30d` | SCALE | skew=-1.12, bounded [0.20, 0.43] |
| `planned_start_day_of_week` | ORDINAL | {0..4}, Mon–Fri ordering |
| `planned_start_hour` | SCALE | skew=+0.04, bounded [4, 23] |
| `lag_as_pct_of_window` *(interaction)* | LOG | skew=+3.76, highest corr (+0.521) |
| `tightness_x_queue` *(interaction)* | SCALE | skew=-0.42, bounded [0, 1] |
| `log_experience_x_concurrent` *(interaction)* | PASSTHROUGH | ZERO VARIANCE placeholder |
| `oee_x_maintenance_ratio` *(interaction)* | SCALE | skew=-0.01, near-zero signal |

**Totals:** 6 LOG + 15 SCALE + 8 BINARY + 4 ORDINAL + 4 PASSTHROUGH = 37 base + 4 interaction = **41 total pipeline features**

---

## 12. Leakage Certification

The feature engineering design is leakage-free **by construction** because all 37 features are captured by `FeatureCollector.snapshot()` at order release, before any execution events occur.

### No direct leakage
All 37 base features reflect state that is knowable before `actual_end` is determined. The following leaky features are confirmed absent from `FEATURE_COLS`:
- `actual_end` — excluded ✓
- `actual_setup_time_minutes` — excluded ✓
- `actual_run_time_minutes` — excluded ✓
- `delay_minutes`, `is_delayed`, `delay_category`, `delay_root_cause` — all in TARGET_COLS only ✓

### Cold-start approximation (not leakage)
The 7 historical rolling features (`*_delay_rate_90d`, `*_overrun_rate_90d`, `*_yield_90d`) use population mean defaults when fewer than 3 historical orders exist. This is an approximation that degrades feature quality for early-simulation orders, not data leakage.

The time-aware train/val/test split means the training set contains the cold-start period. The model learns to discount rolling features when they cluster at default values. The validation and test sets have richer historical features — consistent with the production deployment scenario where the system has been running for weeks or months before predicting.

### Boundary check for rolling features
All rolling feature computations use `release_time` as the strict upper boundary:
```
SELECT ... FROM completed_orders WHERE completed_at < release_time AND ...
```
No future orders contaminate any historical feature.

---

*Design Specification — `ml/src/mpc_ml/features/constants.py`*  
*Manufacturing Process Copilot Technical Series*
