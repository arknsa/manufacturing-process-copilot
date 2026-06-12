"""
ml/src/mpc_ml/features/constants.py
=====================================
Single source of truth for all feature engineering constants in the
Manufacturing Process Copilot (MPC) ML pipeline.

This module is the root of the dependency graph within ``mpc_ml.features``:
it imports nothing from the package itself and is imported by every other
module (``transformers.py``, ``pipeline.py``, training scripts, the backend
serving layer).

Design contract
---------------
* **No business logic.** Every object defined here is a module-level
  declaration.  All logic — pipeline assembly, transformer behaviour,
  cold-start filling — lives in the modules that consume these constants.
* **Immutable exports.** All sequences are ``tuple``; mappings are ``dict``
  literals assigned to ``Final`` names.  Callers that need a mutable copy
  must call ``list(FEATURE_COLS)``, ``dict(COLUMN_DTYPE_CONTRACT)``, etc.
* **Validated at import time.** ``_validate_constants()`` runs once when the
  module is first imported.  A misconfiguration (e.g., adding a feature to
  two preprocessing groups) raises ``ValueError`` immediately rather than
  producing a silently corrupted pipeline.

Architecture references
-----------------------
* Doc 06 — ``constants.py`` design specification
* Doc 07 — ``transformers.py`` design (consumer of this module)
* Doc 08 — ``pipeline.py`` design (consumer of this module)

Constants exported (in declaration order)
------------------------------------------
Feature identification
    FEATURE_COLS, TARGET_COLS

Input schema contract
    COLUMN_DTYPE_CONTRACT

Cold-start handling
    COLD_START_FEATURE_NAMES, COLD_START_DEFAULTS

Zero-variance sentinel
    ZERO_VARIANCE_FEATURES

Preprocessing groups (base features — consumed by pipeline.py)
    LOG_FEATURES, SCALE_FEATURES, BINARY_FEATURES,
    ORDINAL_FEATURES, PASSTHROUGH_FEATURES

Interaction features
    INTERACTION_FEATURE_NAMES

Encoding contracts (consumed by training scripts and the backend)
    PRIORITY_ENCODING, SKILL_TIER_ENCODING, SHIFT_ENCODING,
    DELAY_CATEGORY_ORDER, ROOT_CAUSE_CLASSES

ML configuration
    CANDIDATE_REMOVAL_FEATURES
"""

from __future__ import annotations

import logging
from typing import Dict, Final, FrozenSet, Tuple

logger = logging.getLogger(__name__)

# ===========================================================================
# §1  Feature identification
# ===========================================================================

#: Ordered tuple of the 38 base feature names produced by the simulation and
#: consumed by the preprocessing pipeline.  This is the authoritative schema
#: contract between the data generator and the ML pipeline.
#:
#: Every column listed here must be present (with the dtype declared in
#: ``COLUMN_DTYPE_CONTRACT``) in any DataFrame passed to
#: ``ColumnSelector.transform()``.  Extra columns are silently dropped;
#: missing columns raise ``ValueError``.
#:
#: Groups (for human reference — preprocessing assignments are in §5 below):
#:   [0–8]   Order planning          (9 features)
#:   [9–10]  Product characteristics (2 features; 2 shared with order planning)
#:   [11–12] Temporal flags          (2 features)
#:   [13–20] Machine state           (8 features)
#:   [21–25] Operator state          (5 features)
#:   [26–27] Material state          (2 features)
#:   [28–35] Historical rolling      (8 features; +1 Phase B3 severity feature)
#:   [36–37] Additional temporal     (2 features)
#:
#: Count: 38.
FEATURE_COLS: Final[Tuple[str, ...]] = (
    # ── Order planning ──────────────────────────────────────────────────────
    "planned_lead_time_hours",           # hours from creation to planned completion
    "release_lag_hours",                  # hours from creation to material release
    "schedule_revision_count",            # number of schedule revisions (currently 0 or 1)
    "is_expedited",                       # 1 if order is expedited (critical priority subset)
    "priority_encoded",                   # 1=normal, 2=high, 3=critical (0=low reserved)
    "quantity",                           # units to produce
    "operation_count",                    # routing steps in product BOM
    "estimated_total_hours",              # (std_setup + std_run × qty) / 60 — planner's estimate
    "schedule_tightness_ratio",           # estimated_total_hours / planned_window_hours
    # ── Product characteristics ─────────────────────────────────────────────
    "product_complexity_score",           # 0.25=LOW, 0.55=MEDIUM, 0.85=HIGH complexity
    "material_bom_complexity",            # BOM line-item count [2, 9]
    # ── Temporal flags ───────────────────────────────────────────────────────
    "is_month_end",                       # 1 if order falls in final 4 days of month
    "is_quarter_end",                     # 1 if order falls in final 4 days of quarter
    # ── Machine state at release ─────────────────────────────────────────────
    "machine_utilization_at_release",     # fraction of scheduled time in past 24 h
    "work_center_queue_depth_at_release", # 0 = empty queue, 1 = orders waiting (binary in sim)
    "machine_oee_30d",                    # 30-day Overall Equipment Effectiveness [0, 1]
    "machine_unplanned_downtime_hours_30d", # unplanned downtime hours in trailing 30 days
    "days_since_last_planned_maintenance",  # calendar days since last PM event
    "maintenance_due_within_order_window",  # 1 if PM falls within the order's planned window
    "changeover_required",                # 1 if product differs from last job on machine
    "changeover_complexity_score",        # 1.0 (no changeover) or Uniform[1.5, 3.0] (changeover)
    # ── Operator state at release ────────────────────────────────────────────
    "operator_experience_months",         # months of experience [1, 176]
    "operator_skill_tier_encoded",        # 0=junior, 1=mid, 2=senior
    "operator_concurrent_order_count",    # parallel orders (all-zero in current sim — ZERO_VAR)
    "hours_into_shift_at_start",          # hours elapsed in operator's shift at order start
    "shift_type_encoded",                 # 0=morning, 1=afternoon, 2=night
    # ── Material state at release ────────────────────────────────────────────
    "material_availability_at_release",   # 1=all components available, 0=shortage
    "component_shortage_count",           # missing component count {0, 1, 2}
    # ── Historical rolling features ──────────────────────────────────────────
    "product_delay_rate_90d",             # fraction of delayed orders for this product (90 d)
    "machine_delay_rate_90d",             # fraction of delayed orders on this machine (90 d)
    "operator_delay_rate_90d",            # fraction of delayed orders for this operator (90 d)
    "product_x_machine_delay_rate_90d",   # delay rate for this product × machine pair (90 d)
    "product_first_pass_yield_90d",       # first-pass yield for this product (90 d)
    "machine_setup_overrun_rate_90d",     # fraction of setup overruns on this machine (90 d)
    "shift_delay_rate_30d",               # factory-wide delay rate for this shift (30 d)
    "machine_avg_delay_minutes_90d",      # Phase B3: per-machine expanding-mean delay_minutes (strictly preceding orders)
    # ── Additional temporal ──────────────────────────────────────────────────
    "planned_start_day_of_week",          # 0=Monday … 4=Friday
    "planned_start_hour",                 # hour of day [4, 23]
)

#: The 4 prediction targets produced by the simulation.  These columns are
#: present in the raw simulation CSV but must be excluded from any DataFrame
#: passed to the preprocessing pipeline.  ``ColumnSelector.transform()``
#: raises ``ValueError`` if any of these appear in the input.
TARGET_COLS: Final[Tuple[str, ...]] = (
    "is_delayed",        # primary: binary classification (0=on-time, 1=delayed)
    "delay_minutes",     # regression: minutes late (0 for on-time orders)
    "delay_category",    # ordinal: on_time / minor / moderate / major / critical
    "delay_root_cause",  # multi-class: 7 root-cause labels
)

# ===========================================================================
# §2  Input schema contract
# ===========================================================================

#: Expected pandas dtype string for every column in ``FEATURE_COLS``.
#:
#: Used by ``ColumnSelector._select_and_coerce()`` to validate incoming
#: DataFrames and attempt numeric coercion on dtype mismatches.  The most
#: common benign mismatch is a JSON API round-tripping an ``int64`` column as
#: ``float64``; coercion handles this transparently with a WARNING.
#:
#: Rules applied per column:
#:   * dtype matches contract → no action
#:   * dtype differs → ``pd.to_numeric(errors='coerce')`` attempted
#:   * coercion introduces NaN → ``ValueError`` (strict_dtypes=True) or WARNING
#:
#: Count: 37 entries (one per ``FEATURE_COLS`` member).
COLUMN_DTYPE_CONTRACT: Final[Dict[str, str]] = {
    # ── Order planning ───────────────────────────────────────────────────
    "planned_lead_time_hours":              "float64",
    "release_lag_hours":                    "float64",
    "schedule_revision_count":              "float64",  # {0.0, 1.0} in sim; float by design
    "is_expedited":                         "int64",
    "priority_encoded":                     "int64",
    "quantity":                             "int64",
    "operation_count":                      "int64",
    "estimated_total_hours":                "float64",
    "schedule_tightness_ratio":             "float64",
    # ── Product characteristics ──────────────────────────────────────────
    "product_complexity_score":             "float64",
    "material_bom_complexity":              "int64",
    # ── Temporal flags ───────────────────────────────────────────────────
    "is_month_end":                         "int64",
    "is_quarter_end":                       "int64",
    # ── Machine state ────────────────────────────────────────────────────
    "machine_utilization_at_release":       "float64",
    "work_center_queue_depth_at_release":   "float64",  # {0.0, 1.0}; float by design
    "machine_oee_30d":                      "float64",
    "machine_unplanned_downtime_hours_30d": "float64",
    "days_since_last_planned_maintenance":  "float64",
    "maintenance_due_within_order_window":  "int64",
    "changeover_required":                  "int64",
    "changeover_complexity_score":          "float64",
    # ── Operator state ───────────────────────────────────────────────────
    "operator_experience_months":           "int64",
    "operator_skill_tier_encoded":          "float64",  # {0.0, 1.0, 2.0}
    "operator_concurrent_order_count":      "float64",  # {0.0} — zero-variance
    "hours_into_shift_at_start":            "float64",
    "shift_type_encoded":                   "int64",
    # ── Material state ───────────────────────────────────────────────────
    "material_availability_at_release":     "int64",
    "component_shortage_count":             "float64",  # {0.0, 1.0, 2.0}
    # ── Historical rolling features ──────────────────────────────────────
    "product_delay_rate_90d":               "float64",
    "machine_delay_rate_90d":               "float64",
    "operator_delay_rate_90d":              "float64",
    "product_x_machine_delay_rate_90d":     "float64",
    "product_first_pass_yield_90d":         "float64",
    "machine_setup_overrun_rate_90d":       "float64",
    "shift_delay_rate_30d":                 "float64",
    "machine_avg_delay_minutes_90d":        "float64",
    # ── Additional temporal ──────────────────────────────────────────────
    "planned_start_day_of_week":            "float64",  # {0.0–4.0}
    "planned_start_hour":                   "int64",
}

# ===========================================================================
# §3  Cold-start handling
# ===========================================================================

#: The 7 rolling historical features that may legitimately contain ``NaN``
#: in production inference (when an order involves a product, machine, or
#: operator with fewer than 3 completed orders in the lookback window).
#:
#: ``ColumnSelector.fit()`` computes the training-set population mean for
#: each of these columns and stores it in ``self.cold_start_defaults_``.
#: ``ColumnSelector.transform()`` fills any NaN in these columns with the
#: stored training means — never with the seed values in ``COLD_START_DEFAULTS``.
#:
#: All other ``FEATURE_COLS`` are expected to be non-null; any NaN remaining
#: after cold-start filling raises ``ValueError``.
#:
#: Count: 8.
COLD_START_FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "product_delay_rate_90d",
    "machine_delay_rate_90d",
    "operator_delay_rate_90d",
    "product_x_machine_delay_rate_90d",
    "product_first_pass_yield_90d",
    "machine_setup_overrun_rate_90d",
    "shift_delay_rate_30d",
    "machine_avg_delay_minutes_90d",      # Phase B3: no history for new machines
)

#: Seed population means derived from the 120-day validation simulation run.
#:
#: **These values are used only as a fallback of last resort** inside
#: ``ColumnSelector.fit()``: if a rolling feature is *entirely* NaN in the
#: training set (a degenerate edge case), the seed value is used instead of
#: the data-derived mean.  Under normal training conditions the 540-day
#: production dataset provides non-NaN values for all rolling features.
#:
#: The pipeline always uses ``self.cold_start_defaults_`` (learned from the
#: training data at ``fit()`` time) for inference — never these module-level
#: seed values.  The 540-day training run will produce slightly different
#: population means; these constants are documentation, not serving logic.
#:
#: Keys must match ``COLD_START_FEATURE_NAMES`` exactly.
COLD_START_DEFAULTS: Final[Dict[str, float]] = {
    "product_delay_rate_90d":          0.343,
    "machine_delay_rate_90d":          0.347,
    "operator_delay_rate_90d":         0.354,
    "product_x_machine_delay_rate_90d": 0.342,
    "product_first_pass_yield_90d":    0.916,
    "machine_setup_overrun_rate_90d":  0.521,
    "shift_delay_rate_30d":            0.358,
    "machine_avg_delay_minutes_90d":   100.0,   # Phase B3 seed; actual mean learned from training data
}

# ===========================================================================
# §4  Zero-variance sentinel
# ===========================================================================

#: Features that are currently constant (zero-variance) across all simulation
#: orders.  They must NEVER enter a ``StandardScaler``: fitting a scaler on a
#: constant column computes ``std = 0`` and produces ``NaN`` on transform.
#:
#: Routing in ``pipeline.py``:
#:   * ``operator_concurrent_order_count`` is also in ``PASSTHROUGH_FEATURES``
#:     (declared there for semantic correctness as a count feature), but
#:     ``build_pipeline()`` excludes it from ``_PASSTHROUGH_COLS`` via the
#:     expression ``[f for f in PASSTHROUGH_FEATURES if f not in
#:     set(ZERO_VARIANCE_FEATURES)]``, routing it here instead.
#:   * ``log_experience_x_concurrent`` is an interaction feature
#:     (in ``INTERACTION_FEATURE_NAMES``).  It is all-zeros because its
#:     ``operator_concurrent_order_count`` multiplicand is all-zeros.
#:     When multi-machine operator scheduling is added to the simulation both
#:     features become non-zero; they should then be moved to ``SCALE_FEATURES``
#:     and the pipeline retrained.
#:
#: ``ColumnSelector._check_zero_variance_cols()`` emits a ``WARNING`` (never
#: raises) if these columns contain unexpected values at inference time.
#:
#: Count: 2 (1 base feature + 1 interaction feature).
ZERO_VARIANCE_FEATURES: Final[Tuple[str, ...]] = (
    "operator_concurrent_order_count",  # base feature — always 0.0 in current sim
    "log_experience_x_concurrent",      # interaction feature — always 0.0 in current sim
)

# ===========================================================================
# §5  Preprocessing groups (base features)
# ===========================================================================
#
# These tuples configure the six branches of the ``ColumnTransformer`` inside
# ``build_pipeline()``.  Each base feature appears in exactly one group.
# Interaction features are routed by ``pipeline.py`` alongside the matching
# group (e.g., ``lag_as_pct_of_window`` → ``_LOG_COLS``).
#
# Coverage rule enforced at build-time by ``pipeline._validate_feature_coverage()``:
#   set(FEATURE_COLS) ∪ set(INTERACTION_FEATURE_NAMES)
#       == set(all branch columns)          [completeness]
#   len(all branch columns)                               == 41  [total count]
#   Counter(all branch columns).most_common(1)[0][1]      == 1   [no duplicates]

#: 6 base features with right-skewed distributions (skew > 1.0 or zero-inflated).
#: Transform: ``np.log1p`` → ``StandardScaler`` (sub-pipeline in branch 1).
#:
#: Routing note: ``lag_as_pct_of_window`` (interaction, skew +3.76) is also
#: log-scaled.  ``build_pipeline()`` appends it to ``_LOG_COLS``; it is not
#: listed here since it is not a base feature.
#:
#: Count: 7.
LOG_FEATURES: Final[Tuple[str, ...]] = (
    "planned_lead_time_hours",            # skew +3.57 — lead-time mixture distribution
    "release_lag_hours",                  # skew +3.28 — lognormal generator
    "quantity",                           # skew +4.08 — lognormal generator
    "estimated_total_hours",              # skew +3.76 — product of two lognormals
    "machine_unplanned_downtime_hours_30d", # skew +1.10 — zero-inflated right tail
    "operator_experience_months",         # skew +1.25 — long senior tail
    "machine_avg_delay_minutes_90d",      # Phase B3: zero-inflated right tail (0 for all-on-time machines)
)

#: 15 base features that are bounded, approximately symmetric, or rate-valued
#: ([0, 1]).  Transform: ``StandardScaler`` only (branch 2).
#:
#: Routing note: ``tightness_x_queue`` and ``oee_x_maintenance_ratio``
#: (interactions) are also scale-only.  ``build_pipeline()`` appends them to
#: ``_SCALE_COLS``; they are not listed here.
#:
#: Count: 15.
SCALE_FEATURES: Final[Tuple[str, ...]] = (
    # ── Order planning ───────────────────────────────────────────────────
    "schedule_tightness_ratio",           # bounded [0.18, 1.02], skew +0.32
    # ── Product characteristics ──────────────────────────────────────────
    "product_complexity_score",           # 3 discrete values {0.25, 0.55, 0.85}, skew -0.23
    # ── Machine state ────────────────────────────────────────────────────
    "machine_utilization_at_release",     # bounded [0, 1], bimodal, skew +0.09
    "machine_oee_30d",                    # bounded [0.57, 0.79], skew +0.73
    "days_since_last_planned_maintenance", # bounded [0, 89], skew +0.92
    "changeover_complexity_score",        # bounded [1.0, 3.0], bimodal, skew +0.46
    # ── Operator state ───────────────────────────────────────────────────
    "hours_into_shift_at_start",          # bounded [0, 7], skew +0.03
    # ── Historical rolling features ──────────────────────────────────────
    "product_delay_rate_90d",             # bounded [0, 1] rate, skew +0.66
    "machine_delay_rate_90d",             # bounded [0.08, 0.70] rate, skew +0.12
    "operator_delay_rate_90d",            # bounded [0, 0.67] rate, skew -0.91
    "product_x_machine_delay_rate_90d",   # bounded [0, 1] rate, skew +0.72
    "product_first_pass_yield_90d",       # bounded [0.50, 1.0] rate, skew -1.22
    "machine_setup_overrun_rate_90d",     # bounded [0, 0.83] rate, skew -0.91
    "shift_delay_rate_30d",               # bounded [0.20, 0.43] rate, skew -1.12
    # ── Additional temporal ──────────────────────────────────────────────
    "planned_start_hour",                 # bounded [4, 23], skew +0.04
)

#: 8 binary (0/1) flag features.  Transform: passthrough (branch 3).
#:
#: StandardScaler is deliberately NOT applied: binary features carry boolean
#: semantics and tree-based models split optimally on the raw {0, 1} values.
#:
#: Count: 8.
BINARY_FEATURES: Final[Tuple[str, ...]] = (
    # ── Order planning flags ─────────────────────────────────────────────
    "schedule_revision_count",            # {0.0, 1.0} — positive rate 3.1 %
    "is_expedited",                       # {0, 1}     — positive rate 3.1 %
    # ── Temporal demand flags ────────────────────────────────────────────
    "is_month_end",                       # {0, 1}     — positive rate 22.7 %
    "is_quarter_end",                     # {0, 1}     — positive rate 9.6 %
    # ── Machine state flags ──────────────────────────────────────────────
    "work_center_queue_depth_at_release", # {0.0, 1.0} — positive rate 68.8 %
    "maintenance_due_within_order_window", # {0, 1}    — positive rate 1.4 %
    "changeover_required",                # {0, 1}     — positive rate 51.0 %
    # ── Material state flag ──────────────────────────────────────────────
    "material_availability_at_release",   # {0, 1}     — positive rate 88.1 % (1=available)
)

#: 4 low-cardinality ordered-integer features.  Transform: passthrough (branch 4).
#:
#: The natural ordinal ordering is meaningful (higher code = higher tier/priority);
#: tree-based models find threshold splits on raw integers without scaling.
#:
#: Linear model note: one-hot encoding is more appropriate for logistic/linear
#: regression baselines; the ``build_pipeline()`` factory can be extended with
#: an ``is_linear=True`` flag if needed.
#:
#: Count: 4.
ORDINAL_FEATURES: Final[Tuple[str, ...]] = (
    "priority_encoded",            # {1, 2, 3} — 1=normal, 2=high, 3=critical
    "operator_skill_tier_encoded", # {0.0, 1.0, 2.0} — 0=junior, 1=mid, 2=senior
    "shift_type_encoded",          # {0, 1, 2} — 0=morning, 1=afternoon, 2=night
    "planned_start_day_of_week",   # {0.0 … 4.0} — Mon=0, Fri=4
)

#: 4 small-integer count features.  Transform: passthrough (branch 5 or 6).
#:
#: ``operator_concurrent_order_count`` is also in ``ZERO_VARIANCE_FEATURES``
#: and is routed to branch 6 (zero_variance) by ``build_pipeline()`` via:
#:     ``_PASSTHROUGH_COLS = [f for f in PASSTHROUGH_FEATURES
#:                             if f not in set(ZERO_VARIANCE_FEATURES)]``
#: The remaining 3 features go to branch 5 (passthrough_counts).
#:
#: Count: 4.
PASSTHROUGH_FEATURES: Final[Tuple[str, ...]] = (
    "operation_count",                  # {2 … 6}  — routing step count
    "material_bom_complexity",          # {2 … 9}  — BOM line-item count
    "component_shortage_count",         # {0, 1, 2} — missing components at release
    "operator_concurrent_order_count",  # {0.0}    — ZERO_VAR; routed to branch 6
)

# ===========================================================================
# §6  Interaction features
# ===========================================================================

#: 4 derived features computed by ``InteractionFeatureAdder.transform()``.
#: They are appended to the right of the 37-column base DataFrame, producing
#: a 41-column DataFrame consumed by the ``ColumnTransformer``.
#:
#: The ORDER of names here is authoritative: ``InteractionFeatureAdder``
#: constructs a ``pd.DataFrame(dict(zip(INTERACTION_FEATURE_NAMES, series_list)))``
#: where the list order matches this tuple.
#:
#: Branch routing in ``pipeline.py`` (architecture doc §4):
#:   [0] ``lag_as_pct_of_window``        → LOG branch    (skew +3.76, |r|=0.521)
#:   [1] ``tightness_x_queue``           → SCALE branch  (skew -0.42, |r|=0.399)
#:   [2] ``log_experience_x_concurrent`` → ZERO_VAR branch (all-zero in current sim)
#:   [3] ``oee_x_maintenance_ratio``     → SCALE branch  (corrected v2, |r|=0.023)
#:   [4] ``util_x_queue``               → SCALE branch  (F=386, Day 8 addition)
#:   [5] ``util_x_tight``               → SCALE branch  (F=262, Day 8 addition)
#:
#: Count: 6.
INTERACTION_FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "lag_as_pct_of_window",         # release_lag_hours / planned_lead_time_hours (clipped)
    "tightness_x_queue",            # schedule_tightness_ratio × work_center_queue_depth
    "log_experience_x_concurrent",  # log1p(exp_months) × concurrent_count (zero-var placeholder)
    "oee_x_maintenance_ratio",      # machine_oee_30d / max(days_since_pm/30, 1.0)
    "util_x_queue",                 # machine_utilization_at_release × work_center_queue_depth_at_release
    "util_x_tight",                 # machine_utilization_at_release × schedule_tightness_ratio
)

# ===========================================================================
# §7  Encoding contracts
# ===========================================================================

#: Mapping from human-readable priority label to its integer encoding in
#: ``priority_encoded``.  ``low=0`` is reserved for completeness in production
#: inference even though the current simulation rarely generates low-priority
#: orders.
PRIORITY_ENCODING: Final[Dict[str, int]] = {
    "low":      0,
    "normal":   1,
    "high":     2,
    "critical": 3,
}

#: Mapping from skill tier label to its float encoding in
#: ``operator_skill_tier_encoded``.  Float (not int) because the simulation
#: stores this column as ``float64``.
SKILL_TIER_ENCODING: Final[Dict[str, float]] = {
    "junior": 0.0,
    "mid":    1.0,
    "senior": 2.0,
}

#: Mapping from shift name to its integer encoding in ``shift_type_encoded``.
SHIFT_ENCODING: Final[Dict[str, int]] = {
    "morning":   0,
    "afternoon": 1,
    "night":     2,
}

#: Ordered list of delay-category labels for ordinal label encoding.
#:
#: Ordinal encoding contract: on_time=0, minor_delay=1, moderate_delay=2,
#: major_delay=3, critical_delay=4.  Derived from simulation output:
#:   on_time:        delay_minutes = 0
#:   minor_delay:    1 – 60 min
#:   moderate_delay: 61 – 480 min
#:   major_delay:    481 – 1440 min
#:   critical_delay: > 1440 min
DELAY_CATEGORY_ORDER: Final[Tuple[str, ...]] = (
    "on_time",
    "minor_delay",
    "moderate_delay",
    "major_delay",
    "critical_delay",
)

#: Alphabetically sorted tuple of all root-cause class labels for the
#: multi-class classifier.  Alphabetical ordering is the canonical label
#: encoding baseline (label 0 = machine_breakdown, ..., label 6 = setup_overrun).
ROOT_CAUSE_CLASSES: Final[Tuple[str, ...]] = (
    "machine_breakdown",
    "material_unavailability",
    "multiple_causes",
    "none",
    "planning_schedule_conflict",
    "quality_failure_rework",
    "setup_overrun",
)

# ===========================================================================
# §8  ML configuration
# ===========================================================================

#: Features with near-zero empirical predictive value that are candidates for
#: removal during Optuna hyperparameter tuning.
#:
#: These are NOT removed from the pipeline by default — inclusion is tested
#: empirically with permutation importance on the validation split.  Optuna's
#: objective function can gate on a boolean hyperparameter per feature to
#: test with and without each candidate.
#:
#: Basis for inclusion:
#:   oee_x_maintenance_ratio:  |r| = 0.023 (near-zero, corrected v2 formula)
#:   shift_delay_rate_30d:     |r| < 0.05  (aggregate, low standalone signal)
#:   material_bom_complexity:  |r| < 0.06  (signal flows through material_availability)
#:   operation_count:          |r| < 0.05  (correlated with product_complexity_score)
#:
#: Removed:
#:   is_quarter_end: was measured on zero-variance broken data (calendar bug); has real
#:                   causal signal after fix (~5 % positive rate, demand-congestion link)
CANDIDATE_REMOVAL_FEATURES: Final[FrozenSet[str]] = frozenset({
    "oee_x_maintenance_ratio",
    "shift_delay_rate_30d",
    "material_bom_complexity",
    "operation_count",
})

# ===========================================================================
# §9  Module-level validation
# ===========================================================================


def _validate_constants() -> None:
    """Validate internal consistency of all constants in this module.

    Called once at module import time.  Raises ``ValueError`` immediately on
    any misconfiguration rather than allowing a silently corrupted pipeline to
    be assembled.  The comprehensive pipeline-level coverage validation
    (Conditions A, B, C in the architecture spec) is performed by
    ``pipeline._validate_feature_coverage()`` at ``build_pipeline()`` call
    time; this function focuses on the constant-file's own invariants.

    Checks performed
    ----------------
    V-01  ``FEATURE_COLS`` has exactly 37 unique members.
    V-02  ``TARGET_COLS`` has exactly 4 unique members.
    V-03  ``INTERACTION_FEATURE_NAMES`` has exactly 4 unique members.
    V-04  ``FEATURE_COLS`` and ``TARGET_COLS`` are disjoint.
    V-05  ``FEATURE_COLS`` has no duplicate entries.
    V-06  ``TARGET_COLS`` has no duplicate entries.
    V-07  All ``LOG_FEATURES`` members are in ``FEATURE_COLS``.
    V-08  All ``SCALE_FEATURES`` members are in ``FEATURE_COLS``.
    V-09  All ``BINARY_FEATURES`` members are in ``FEATURE_COLS``.
    V-10  All ``ORDINAL_FEATURES`` members are in ``FEATURE_COLS``.
    V-11  All ``PASSTHROUGH_FEATURES`` members are in ``FEATURE_COLS``.
    V-12  ``LOG``, ``SCALE``, ``BINARY``, ``ORDINAL`` are mutually exclusive.
    V-13  Union of all base groups (``LOG + SCALE + BINARY + ORDINAL +
          PASSTHROUGH``) covers ``FEATURE_COLS`` exactly.
    V-14  ``COLD_START_FEATURE_NAMES`` is a subset of ``FEATURE_COLS``.
    V-15  ``COLD_START_DEFAULTS`` keys match ``COLD_START_FEATURE_NAMES``.
    V-16  ``COLD_START_DEFAULTS`` values are valid finite floats.
    V-17  ``ZERO_VARIANCE_FEATURES`` members are in
          ``FEATURE_COLS ∪ INTERACTION_FEATURE_NAMES``.
    V-18  ``COLUMN_DTYPE_CONTRACT`` has exactly one entry per ``FEATURE_COLS``
          member and no extras.
    V-19  ``DELAY_CATEGORY_ORDER`` has exactly 5 unique members.
    V-20  ``ROOT_CAUSE_CLASSES`` has exactly 7 unique members.

    Raises
    ------
    ValueError
        Raised on the first violated check, with a message identifying the
        specific offending constant name(s) and the nature of the violation.
    """
    import math

    feature_set: FrozenSet[str] = frozenset(FEATURE_COLS)
    target_set:  FrozenSet[str] = frozenset(TARGET_COLS)
    interaction_set: FrozenSet[str] = frozenset(INTERACTION_FEATURE_NAMES)

    # ── V-01  FEATURE_COLS count ────────────────────────────────────────────
    if len(FEATURE_COLS) != 38:
        raise ValueError(
            f"V-01: FEATURE_COLS must contain exactly 38 features; "
            f"found {len(FEATURE_COLS)}."
        )

    # ── V-02  TARGET_COLS count ─────────────────────────────────────────────
    if len(TARGET_COLS) != 4:
        raise ValueError(
            f"V-02: TARGET_COLS must contain exactly 4 targets; "
            f"found {len(TARGET_COLS)}."
        )

    # ── V-03  INTERACTION_FEATURE_NAMES count ───────────────────────────────
    if len(INTERACTION_FEATURE_NAMES) != 6:
        raise ValueError(
            f"V-03: INTERACTION_FEATURE_NAMES must contain exactly 6 names; "
            f"found {len(INTERACTION_FEATURE_NAMES)}."
        )

    # ── V-04  FEATURE_COLS and TARGET_COLS are disjoint ─────────────────────
    overlap: FrozenSet[str] = feature_set & target_set
    if overlap:
        raise ValueError(
            f"V-04: FEATURE_COLS and TARGET_COLS must be disjoint. "
            f"Overlapping names: {sorted(overlap)}."
        )

    # ── V-05  FEATURE_COLS has no duplicates ────────────────────────────────
    if len(FEATURE_COLS) != len(feature_set):
        from collections import Counter
        dupes = sorted(k for k, v in Counter(FEATURE_COLS).items() if v > 1)
        raise ValueError(
            f"V-05: FEATURE_COLS contains duplicate entries: {dupes}."
        )

    # ── V-06  TARGET_COLS has no duplicates ─────────────────────────────────
    if len(TARGET_COLS) != len(target_set):
        from collections import Counter
        dupes = sorted(k for k, v in Counter(TARGET_COLS).items() if v > 1)
        raise ValueError(
            f"V-06: TARGET_COLS contains duplicate entries: {dupes}."
        )

    # ── V-07 to V-11  All preprocessing group members are in FEATURE_COLS ───
    _group_checks: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("LOG_FEATURES",         LOG_FEATURES),
        ("SCALE_FEATURES",       SCALE_FEATURES),
        ("BINARY_FEATURES",      BINARY_FEATURES),
        ("ORDINAL_FEATURES",     ORDINAL_FEATURES),
        ("PASSTHROUGH_FEATURES", PASSTHROUGH_FEATURES),
    )
    for check_id, (group_name, group) in zip(
        ("V-07", "V-08", "V-09", "V-10", "V-11"), _group_checks
    ):
        unknown = sorted(set(group) - feature_set)
        if unknown:
            raise ValueError(
                f"{check_id}: {group_name} contains feature(s) not in "
                f"FEATURE_COLS: {unknown}."
            )

    # ── V-12  LOG, SCALE, BINARY, ORDINAL are mutually exclusive ────────────
    #   (PASSTHROUGH is allowed to overlap ZERO_VARIANCE_FEATURES by design)
    from collections import Counter
    base_four: Tuple[str, ...] = (
        LOG_FEATURES + SCALE_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES
    )
    base_four_dupes = sorted(
        k for k, v in Counter(base_four).items() if v > 1
    )
    if base_four_dupes:
        raise ValueError(
            "V-12: LOG_FEATURES, SCALE_FEATURES, BINARY_FEATURES, and "
            "ORDINAL_FEATURES must be mutually exclusive. "
            f"Features assigned to more than one group: {base_four_dupes}."
        )

    # ── V-13  Union of all base groups covers FEATURE_COLS exactly ──────────
    all_base: FrozenSet[str] = frozenset(
        LOG_FEATURES + SCALE_FEATURES + BINARY_FEATURES
        + ORDINAL_FEATURES + PASSTHROUGH_FEATURES
    )
    missing_from_groups = feature_set - all_base
    if missing_from_groups:
        raise ValueError(
            "V-13: The following FEATURE_COLS members are not assigned to "
            f"any preprocessing group: {sorted(missing_from_groups)}. "
            "Add each feature to exactly one of LOG_FEATURES, SCALE_FEATURES, "
            "BINARY_FEATURES, ORDINAL_FEATURES, or PASSTHROUGH_FEATURES."
        )
    extra_in_groups = all_base - feature_set
    if extra_in_groups:
        raise ValueError(
            "V-13: The following feature(s) appear in a preprocessing group "
            f"but are absent from FEATURE_COLS: {sorted(extra_in_groups)}."
        )

    # ── V-14  COLD_START_FEATURE_NAMES ⊆ FEATURE_COLS ──────────────────────
    cold_start_unknown = sorted(set(COLD_START_FEATURE_NAMES) - feature_set)
    if cold_start_unknown:
        raise ValueError(
            f"V-14: COLD_START_FEATURE_NAMES contains feature(s) not in "
            f"FEATURE_COLS: {cold_start_unknown}."
        )

    # ── V-15  COLD_START_DEFAULTS keys == COLD_START_FEATURE_NAMES ──────────
    defaults_keys = frozenset(COLD_START_DEFAULTS.keys())
    cold_start_names = frozenset(COLD_START_FEATURE_NAMES)
    missing_defaults = sorted(cold_start_names - defaults_keys)
    extra_defaults   = sorted(defaults_keys - cold_start_names)
    if missing_defaults or extra_defaults:
        raise ValueError(
            "V-15: COLD_START_DEFAULTS keys must match COLD_START_FEATURE_NAMES "
            f"exactly. Missing: {missing_defaults}. Extra: {extra_defaults}."
        )

    # ── V-16  COLD_START_DEFAULTS values are finite floats ──────────────────
    non_finite = sorted(
        k for k, v in COLD_START_DEFAULTS.items()
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v)
    )
    if non_finite:
        raise ValueError(
            f"V-16: COLD_START_DEFAULTS contains non-finite value(s) for "
            f"key(s): {non_finite}."
        )

    # ── V-17  ZERO_VARIANCE_FEATURES ⊆ FEATURE_COLS ∪ INTERACTION_FEATURE_NAMES
    allowed_zero_var = feature_set | interaction_set
    unknown_zero_var = sorted(set(ZERO_VARIANCE_FEATURES) - allowed_zero_var)
    if unknown_zero_var:
        raise ValueError(
            "V-17: ZERO_VARIANCE_FEATURES contains name(s) not present in "
            "FEATURE_COLS or INTERACTION_FEATURE_NAMES: "
            f"{unknown_zero_var}."
        )

    # ── V-18  COLUMN_DTYPE_CONTRACT covers FEATURE_COLS exactly ─────────────
    contract_keys = frozenset(COLUMN_DTYPE_CONTRACT.keys())
    missing_contract = sorted(feature_set - contract_keys)
    extra_contract   = sorted(contract_keys - feature_set)
    if missing_contract or extra_contract:
        raise ValueError(
            "V-18: COLUMN_DTYPE_CONTRACT keys must match FEATURE_COLS exactly. "
            f"Missing entries: {missing_contract}. "
            f"Extra entries (not in FEATURE_COLS): {extra_contract}."
        )

    # ── V-19  DELAY_CATEGORY_ORDER has 5 unique members ─────────────────────
    if len(DELAY_CATEGORY_ORDER) != 5 or len(set(DELAY_CATEGORY_ORDER)) != 5:
        raise ValueError(
            f"V-19: DELAY_CATEGORY_ORDER must have exactly 5 unique entries; "
            f"found {len(DELAY_CATEGORY_ORDER)} entries "
            f"({len(set(DELAY_CATEGORY_ORDER))} unique)."
        )

    # ── V-20  ROOT_CAUSE_CLASSES has 7 unique members ───────────────────────
    if len(ROOT_CAUSE_CLASSES) != 7 or len(set(ROOT_CAUSE_CLASSES)) != 7:
        raise ValueError(
            f"V-20: ROOT_CAUSE_CLASSES must have exactly 7 unique entries; "
            f"found {len(ROOT_CAUSE_CLASSES)} entries "
            f"({len(set(ROOT_CAUSE_CLASSES))} unique)."
        )

    logger.debug(
        "_validate_constants(): all 20 checks passed. "
        "FEATURE_COLS=%d, TARGET_COLS=%d, INTERACTION_FEATURE_NAMES=%d. "
        "Groups: LOG=%d, SCALE=%d, BINARY=%d, ORDINAL=%d, PASSTHROUGH=%d, "
        "ZERO_VAR=%d, COLD_START=%d.",
        len(FEATURE_COLS),
        len(TARGET_COLS),
        len(INTERACTION_FEATURE_NAMES),
        len(LOG_FEATURES),
        len(SCALE_FEATURES),
        len(BINARY_FEATURES),
        len(ORDINAL_FEATURES),
        len(PASSTHROUGH_FEATURES),
        len(ZERO_VARIANCE_FEATURES),
        len(COLD_START_FEATURE_NAMES),
    )


# Run once at import time.  Any misconfiguration surfaces immediately on
# ``import mpc_ml.features.constants`` rather than silently at training time.
_validate_constants()
