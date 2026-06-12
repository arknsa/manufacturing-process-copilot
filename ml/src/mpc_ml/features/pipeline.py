"""
ml/src/mpc_ml/features/pipeline.py
====================================
Factory module for the Manufacturing Process Copilot (MPC) feature
preprocessing pipeline.

Public API
----------
build_pipeline() -> sklearn.pipeline.Pipeline
    Build and return an *unfitted* 3-step preprocessing pipeline.  No model
    is attached.  Raises ``ValueError`` *at call time* if the preprocessing
    group assignments in ``constants.py`` do not cover exactly 41 features
    without duplication.

get_feature_names() -> List[str]
    Return the 41-element ordered list of feature names in the
    ``ColumnTransformer`` output column order [0..40].  Stateless — no
    fitted pipeline is required.

Design constraints (from architecture doc §1–§4)
-------------------------------------------------
* ``build_pipeline()`` has **no parameters**.  All configuration is driven
  by ``constants.py``.  Moving a feature between preprocessing groups
  requires only a change in ``constants.py``; ``build_pipeline()`` reflects
  it automatically at the next call.

* The returned pipeline contains **no model**.  Callers attach one via::

      full_pipeline = Pipeline([
          ('preprocessor', build_pipeline()),
          ('model', XGBClassifier(**best_params)),
      ])

  This separation is mandatory for SHAP extraction and multi-task training.

* ``set_output(transform='pandas')`` is configured unconditionally.
  ``scikit-learn >= 1.4`` is required by this package, so the API is always
  available.  Named DataFrame columns are required by the SHAP service
  (``explainability.py``) and by MLflow signature inference.

* ``verbose_feature_names_out=False`` on the ``ColumnTransformer`` is
  mandatory so that output column names are plain feature names
  (``planned_lead_time_hours``) rather than branch-prefixed names
  (``log_scale__planned_lead_time_hours``).  This is required for correct
  SHAP position-to-name mapping via ``get_feature_names()``.

Output feature map (ColumnTransformer output order)
---------------------------------------------------
Position   Feature name                               Transform
─────────  ─────────────────────────────────────────  ─────────────────────
[  0]      planned_lead_time_hours                    log1p → StandardScaler
[  1]      release_lag_hours                          log1p → StandardScaler
[  2]      quantity                                   log1p → StandardScaler
[  3]      estimated_total_hours                      log1p → StandardScaler
[  4]      machine_unplanned_downtime_hours_30d       log1p → StandardScaler
[  5]      operator_experience_months                 log1p → StandardScaler
[  6]      machine_avg_delay_minutes_90d  [Phase B3]  log1p → StandardScaler
[  7]      lag_as_pct_of_window        [INTERACTION]  log1p → StandardScaler
[  8]      schedule_tightness_ratio                   StandardScaler
[  9]      product_complexity_score                   StandardScaler
[ 10]      machine_utilization_at_release             StandardScaler
[ 11]      machine_oee_30d                            StandardScaler
[ 12]      days_since_last_planned_maintenance        StandardScaler
[ 13]      changeover_complexity_score                StandardScaler
[ 14]      hours_into_shift_at_start                  StandardScaler
[ 15]      product_delay_rate_90d                     StandardScaler
[ 16]      machine_delay_rate_90d                     StandardScaler
[ 17]      operator_delay_rate_90d                    StandardScaler
[ 18]      product_x_machine_delay_rate_90d           StandardScaler
[ 19]      product_first_pass_yield_90d               StandardScaler
[ 20]      machine_setup_overrun_rate_90d             StandardScaler
[ 21]      shift_delay_rate_30d                       StandardScaler
[ 22]      planned_start_hour                         StandardScaler
[ 23]      tightness_x_queue           [INTERACTION]  StandardScaler
[ 24]      oee_x_maintenance_ratio     [INTERACTION]  StandardScaler
[ 25]      util_x_queue                [INTERACTION]  StandardScaler
[ 26]      util_x_tight                [INTERACTION]  StandardScaler
[ 27]      schedule_revision_count                    passthrough
[ 28]      is_expedited                               passthrough
[ 29]      is_month_end                               passthrough
[ 30]      is_quarter_end                             passthrough
[ 31]      work_center_queue_depth_at_release         passthrough
[ 32]      maintenance_due_within_order_window        passthrough
[ 33]      changeover_required                        passthrough
[ 34]      material_availability_at_release           passthrough
[ 35]      priority_encoded                           passthrough
[ 36]      operator_skill_tier_encoded                passthrough
[ 37]      shift_type_encoded                         passthrough
[ 38]      planned_start_day_of_week                  passthrough
[ 39]      operation_count                            passthrough
[ 40]      material_bom_complexity                    passthrough
[ 41]      component_shortage_count                   passthrough
[ 42]      operator_concurrent_order_count            passthrough (ZERO_VAR)
[ 43]      log_experience_x_concurrent [INTERACTION]  passthrough (ZERO_VAR)
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mpc_ml.features.constants import (
    BINARY_FEATURES,
    FEATURE_COLS,
    INTERACTION_FEATURE_NAMES,
    LOG_FEATURES,
    ORDINAL_FEATURES,
    PASSTHROUGH_FEATURES,
    SCALE_FEATURES,
    ZERO_VARIANCE_FEATURES,
)
from mpc_ml.features.transformers import ColumnSelector, InteractionFeatureAdder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level branch column lists
#
# Computed once at import time from ``constants.py``.  These serve two
# purposes:
#   1. Configure the ``ColumnTransformer`` inside ``build_pipeline()``.
#   2. Back ``get_feature_names()`` without requiring a fitted pipeline.
#
# Interaction feature routing (from architecture doc §4):
#   lag_as_pct_of_window        → _LOG_COLS     skew +3.76; LOG treatment
#   tightness_x_queue           → _SCALE_COLS   bounded, near-symmetric
#   oee_x_maintenance_ratio     → _SCALE_COLS   bounded, near-symmetric
#   log_experience_x_concurrent → _ZERO_VAR_COLS already in ZERO_VARIANCE_FEATURES
#
# PASSTHROUGH_FEATURES intentionally split across two branches:
#   _PASSTHROUGH_COLS excludes operator_concurrent_order_count — it goes to
#   _ZERO_VAR_COLS where it is protected from StandardScaler (std=0 → NaN).
# ---------------------------------------------------------------------------

# Branch 1 — log1p then StandardScaler (8 features: 7 base + 1 interaction)
_LOG_COLS: List[str] = list(LOG_FEATURES) + ["lag_as_pct_of_window"]

# Branch 2 — StandardScaler only (19 features: 15 base + 4 interactions)
_SCALE_COLS: List[str] = list(SCALE_FEATURES) + [
    "tightness_x_queue",
    "oee_x_maintenance_ratio",
    "util_x_queue",
    "util_x_tight",
]

# Branch 3 — passthrough, boolean flags (8 features)
_BINARY_COLS: List[str] = list(BINARY_FEATURES)

# Branch 4 — passthrough, ordered integer codes (4 features)
_ORDINAL_COLS: List[str] = list(ORDINAL_FEATURES)

# Branch 5 — passthrough, small integer counts (3 features)
#   operator_concurrent_order_count is intentionally excluded: zero-variance
#   at fit time means StandardScaler would produce NaN (0 / 0).  It is
#   routed to _ZERO_VAR_COLS below.
_PASSTHROUGH_COLS: List[str] = [
    f for f in PASSTHROUGH_FEATURES if f not in set(ZERO_VARIANCE_FEATURES)
]

# Branch 6 — passthrough, zero-variance columns (2 features)
#   operator_concurrent_order_count  — base feature, currently all-zeros
#   log_experience_x_concurrent      — interaction feature, currently all-zeros
#   Both are in ZERO_VARIANCE_FEATURES by design.  Neither must ever enter
#   a StandardScaler until the zero-variance assumption is lifted and the
#   pipeline is retrained.
_ZERO_VAR_COLS: List[str] = list(ZERO_VARIANCE_FEATURES)

# Canonical 44-element output order — ColumnTransformer branch concat order.
# Positions: [0–7] log_scale | [8–26] scale_only | [27–34] binary |
#            [35–38] ordinal | [39–41] passthrough_counts | [42–43] zero_variance
_ORDERED_FEATURE_NAMES: List[str] = (
    _LOG_COLS
    + _SCALE_COLS
    + _BINARY_COLS
    + _ORDINAL_COLS
    + _PASSTHROUGH_COLS
    + _ZERO_VAR_COLS
)


# ---------------------------------------------------------------------------
# Build-time validation
# ---------------------------------------------------------------------------

def _validate_feature_coverage() -> None:
    """
    Assert that the branch column assignments cover ``FEATURE_COLS +
    INTERACTION_FEATURE_NAMES`` exactly once, with no missing features, no
    duplicates, and a total count of exactly 41.

    This is called as the **first executable statement** inside
    ``build_pipeline()`` — before any sklearn object is constructed — so
    that a misconfiguration in ``constants.py`` surfaces immediately at
    training time rather than silently corrupting model output.

    Three conditions are checked independently to produce actionable error
    messages:

    Condition A — completeness
        Every feature in the expected set appears in at least one branch.

    Condition B — exclusivity
        No feature appears in more than one branch.

    Condition C — total count
        The combined branch list has exactly 41 members.

    Raises
    ------
    ValueError
        Raised on any violation of Conditions A, B, or C.  The error
        message names the specific offending feature(s).
    """
    expected: List[str] = list(FEATURE_COLS) + list(INTERACTION_FEATURE_NAMES)
    assigned: List[str] = _ORDERED_FEATURE_NAMES

    expected_set = set(expected)
    assigned_set = set(assigned)

    # ------------------------------------------------------------------
    # Condition A — completeness: every expected feature is assigned
    # ------------------------------------------------------------------
    missing = expected_set - assigned_set
    if missing:
        raise ValueError(
            "build_pipeline() coverage error [Condition A — completeness].\n"
            "The following features are declared in FEATURE_COLS or "
            "INTERACTION_FEATURE_NAMES but are not assigned to any "
            "ColumnTransformer branch:\n"
            f"  {sorted(missing)}\n"
            "Assign each missing feature to exactly one preprocessing group "
            "in constants.py."
        )

    # ------------------------------------------------------------------
    # Extra guard: no phantom features in branches that are absent from
    # the expected set (indicates a stale branch list or typo in a
    # column name).
    # ------------------------------------------------------------------
    extra = assigned_set - expected_set
    if extra:
        raise ValueError(
            "build_pipeline() coverage error [extra features in branches].\n"
            "The following features appear in a ColumnTransformer branch "
            "column list but are absent from FEATURE_COLS and "
            "INTERACTION_FEATURE_NAMES:\n"
            f"  {sorted(extra)}\n"
            "Remove the phantom feature(s) from the branch column lists, "
            "or add them to the appropriate constant in constants.py."
        )

    # ------------------------------------------------------------------
    # Condition B — exclusivity: no feature appears in multiple branches
    # ------------------------------------------------------------------
    counts = Counter(assigned)
    duplicates = sorted(name for name, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(
            "build_pipeline() coverage error [Condition B — exclusivity].\n"
            "The following features are assigned to more than one "
            "ColumnTransformer branch:\n"
            f"  {duplicates}\n"
            "Each feature must appear in exactly one preprocessing group."
        )

    # ------------------------------------------------------------------
    # Condition C — total count must equal 43
    # ------------------------------------------------------------------
    expected_total = len(FEATURE_COLS) + len(INTERACTION_FEATURE_NAMES)
    total = len(assigned)
    if total != expected_total:
        raise ValueError(
            "build_pipeline() coverage error [Condition C — total count].\n"
            f"Expected exactly {expected_total} features across all branches, found {total}.\n"
            f"FEATURE_COLS contributes {len(FEATURE_COLS)} features "
            f"(expected 37); "
            f"INTERACTION_FEATURE_NAMES contributes {len(INTERACTION_FEATURE_NAMES)} "
            f"features (expected 6).\n"
            "Check that constants.py lists are complete and non-overlapping."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_names() -> List[str]:
    """
    Return the 44-element ordered list of feature names in ``ColumnTransformer``
    output column order.

    This is the authoritative source for mapping SHAP value positions [0..43]
    to human-readable feature names.  The mapping is deterministic from
    ``constants.py`` — no fitted pipeline is required.

    Usage (from ``explainability.py``)::

        feature_names = get_feature_names()          # 44 names
        X_model = preprocessor.transform(X_raw)      # shape (n, 44)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_model)             # shape (n, 44)

        # Map SHAP positions to feature names for one order:
        shap_dict = dict(zip(feature_names, shap_values.values[0]))

    Also used for MLflow artifact logging::

        mlflow.log_dict(
            {'feature_names': get_feature_names()},
            artifact_file='feature_names.json',
        )

    Returns
    -------
    List[str]
        44 feature names, one per ``ColumnTransformer`` output column, in
        fixed branch-concatenation order:
          [0–7]   log_scale branch
          [8–26]  scale_only branch
          [27–34] binary branch
          [35–38] ordinal branch
          [39–41] passthrough_counts branch
          [42–43] zero_variance branch
    """
    return list(_ORDERED_FEATURE_NAMES)


def build_pipeline() -> Pipeline:
    """
    Build and return an *unfitted* 3-step preprocessing pipeline.

    The pipeline intentionally contains **no model**.  Add a model via::

        full_pipeline = Pipeline([
            ('preprocessor', build_pipeline()),
            ('model', XGBClassifier(**best_params)),
        ])

    Pipeline steps
    --------------
    ``'column_selector'`` — :class:`ColumnSelector`
        Validates the 37-column schema, rejects target columns, coerces
        dtypes, fills rolling-feature NaN with training-set cold-start
        defaults, asserts zero-variance columns, and returns a clean
        37-column DataFrame.

    ``'interaction_adder'`` — :class:`InteractionFeatureAdder`
        Computes 4 derived interaction features and appends them, returning
        a 41-column DataFrame.

    ``'column_transformer'`` — :class:`sklearn.compose.ColumnTransformer`
        Six named branches in concatenation order:

        ==================  =================================  ========
        Branch name         Transformer                        # cols
        ==================  =================================  ========
        ``log_scale``       Pipeline(log1p → StandardScaler)  8
        ``scale_only``      StandardScaler                     19
        ``binary``          passthrough                        8
        ``ordinal``         passthrough                        4
        ``passthrough_counts``  passthrough                    3
        ``zero_variance``   passthrough                        2
        ==================  =================================  ========

        Settings: ``remainder='drop'``, ``verbose_feature_names_out=False``,
        ``n_jobs=1``.

    Training pattern
    ----------------
    ::

        preprocessor = build_pipeline()
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBClassifier(**best_params)),
        ])

        X_train = train_df.drop(TARGET_COLS, axis=1)
        y_train = train_df['is_delayed']

        full_pipeline.fit(X_train, y_train)

    MLflow logging pattern
    ----------------------
    ::

        mlflow.sklearn.log_model(
            full_pipeline,
            artifact_path='pipeline',
            signature=mlflow.models.infer_signature(
                X_train, full_pipeline.predict_proba(X_train)
            ),
        )
        mlflow.sklearn.log_model(
            full_pipeline.named_steps['preprocessor'],
            artifact_path='preprocessing_pipeline',
        )
        mlflow.log_dict(
            {'feature_names': get_feature_names()},
            artifact_file='feature_names.json',
        )

    SHAP extraction pattern
    -----------------------
    ::

        preprocessor = full_pipeline.named_steps['preprocessor']
        model        = full_pipeline.named_steps['model']
        feature_names = get_feature_names()

        X_model = preprocessor.transform(X_raw)          # (n, 41)
        explainer = shap.TreeExplainer(model, data=background_sample)
        shap_values = explainer(X_model)                  # shape (n, 41)
        shap_dict = dict(zip(feature_names, shap_values.values[0]))

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline with pandas DataFrame output enabled via
        ``set_output(transform='pandas')``.

    Raises
    ------
    ValueError
        Raised **at call time** (before any data is seen) if the branch
        column assignments do not satisfy all three coverage conditions:
        completeness (Condition A), exclusivity (Condition B), and total
        count == 41 (Condition C).
    """
    # ------------------------------------------------------------------
    # 1. Build-time coverage validation.
    #
    #    The spec mandates this as the FIRST executable statement inside
    #    build_pipeline(), before any sklearn object is constructed.
    #    Any misconfiguration in constants.py is caught here at training
    #    time, not silently during prediction.
    # ------------------------------------------------------------------
    _validate_feature_coverage()

    # ------------------------------------------------------------------
    # 2. Branch 1 sub-pipeline: log1p then StandardScaler.
    #
    #    FunctionTransformer(validate=False) is REQUIRED.  With the
    #    default validate=True, the transformer converts the input to a
    #    2-D numpy array before applying np.log1p, which discards pandas
    #    column names and breaks the downstream ColumnTransformer's
    #    name-based column selection.
    # ------------------------------------------------------------------
    log_scale_pipeline = Pipeline(
        steps=[
            ("log",    FunctionTransformer(func=np.log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    # ------------------------------------------------------------------
    # 3. ColumnTransformer with 6 named branches.
    #
    #    remainder='drop'
    #        Columns outside the declared branches are discarded.  The
    #        coverage validation above guarantees no needed column is
    #        silently lost.
    #
    #    verbose_feature_names_out=False
    #        Output names are plain column names rather than branch-
    #        prefixed names.  Required for correct SHAP position-to-name
    #        mapping via get_feature_names().
    #
    #    n_jobs=1
    #        Single-threaded.  The dataset is small; parallel branch
    #        fitting overhead exceeds the benefit.
    # ------------------------------------------------------------------
    column_transformer = ColumnTransformer(
        transformers=[
            # ── Branch 1: log1p → StandardScaler (output positions [0–6]) ──
            (
                "log_scale",
                log_scale_pipeline,
                _LOG_COLS,
            ),
            # ── Branch 2: StandardScaler only (output positions [7–23]) ─────
            (
                "scale_only",
                StandardScaler(),
                _SCALE_COLS,
            ),
            # ── Branch 3: passthrough — binary flags (positions [24–31]) ────
            (
                "binary",
                "passthrough",
                _BINARY_COLS,
            ),
            # ── Branch 4: passthrough — ordinal codes (positions [32–35]) ───
            (
                "ordinal",
                "passthrough",
                _ORDINAL_COLS,
            ),
            # ── Branch 5: passthrough — small counts (positions [36–38]) ────
            #    operator_concurrent_order_count is intentionally absent here;
            #    it is zero-variance and routed to the zero_variance branch.
            (
                "passthrough_counts",
                "passthrough",
                _PASSTHROUGH_COLS,
            ),
            # ── Branch 6: passthrough — zero-variance (positions [39–40]) ───
            #    These columns MUST NOT enter any StandardScaler: std=0 at
            #    fit time would produce NaN values silently.  The
            #    ColumnSelector emits a WARNING if these columns become
            #    non-zero, signalling that the pipeline needs retraining.
            (
                "zero_variance",
                "passthrough",
                _ZERO_VAR_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        n_jobs=1,
    )

    # ------------------------------------------------------------------
    # 4. Assemble the 3-step preprocessing Pipeline.
    # ------------------------------------------------------------------
    preprocessing_pipeline = Pipeline(
        steps=[
            ("column_selector",    ColumnSelector()),
            ("interaction_adder",  InteractionFeatureAdder()),
            ("column_transformer", column_transformer),
        ]
    )

    # ------------------------------------------------------------------
    # 5. Enable pandas DataFrame output (sklearn >= 1.2; required >= 1.4).
    #
    #    Named columns are required by:
    #      - SHAP TreeExplainer (uses column names for display/matching)
    #      - MLflow infer_signature (uses DataFrame dtypes and column names)
    #      - Backend services that select output columns by name
    #
    #    set_output propagates to all steps in the Pipeline, including the
    #    nested sub-pipeline inside log_scale.
    # ------------------------------------------------------------------
    preprocessing_pipeline.set_output(transform="pandas")

    logger.debug(
        "build_pipeline(): pipeline constructed. "
        "log_scale=%d, scale_only=%d, binary=%d, ordinal=%d, "
        "passthrough_counts=%d, zero_variance=%d. Total output cols=%d.",
        len(_LOG_COLS),
        len(_SCALE_COLS),
        len(_BINARY_COLS),
        len(_ORDINAL_COLS),
        len(_PASSTHROUGH_COLS),
        len(_ZERO_VAR_COLS),
        len(_ORDERED_FEATURE_NAMES),
    )

    return preprocessing_pipeline
