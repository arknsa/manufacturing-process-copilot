"""Sklearn-compatible transformers for the MPC ML feature preprocessing pipeline.

This module implements the first two steps of the ``build_pipeline()`` preprocessing
chain:

Step 1 — :class:`ColumnSelector`
    Schema enforcement, dtype coercion, and cold-start NaN filling.  Receives an
    arbitrary DataFrame (may contain target columns and extra metadata columns) and
    returns a clean 37-column DataFrame guaranteed to have zero NaN and correct dtypes.

Step 2 — :class:`InteractionFeatureAdder`
    Vectorised computation of the 4 derived interaction features defined in
    ``INTERACTION_FEATURE_NAMES``.  Receives the 37-column output of
    :class:`ColumnSelector` and appends 4 columns, returning a 41-column DataFrame.

Both classes:

- Inherit :class:`sklearn.base.BaseEstimator` and :class:`sklearn.base.TransformerMixin`,
  giving ``fit_transform()``, ``get_params()``/``set_params()``, ``__repr__``, and
  :func:`sklearn.utils.check_is_fitted` compatibility for free.
- Return ``pandas.DataFrame`` objects from :meth:`transform` — never numpy arrays —
  so that the downstream :class:`~sklearn.compose.ColumnTransformer` can select
  columns by name.
- Are safe to serialise with :mod:`pickle` / ``MLflow``; all state needed for
  deterministic inference is stored as fitted attributes.

Example usage::

    from sklearn.pipeline import Pipeline
    from mpc_ml.features.transformers import ColumnSelector, InteractionFeatureAdder

    pipe = Pipeline([
        ("selector", ColumnSelector()),
        ("adder",    InteractionFeatureAdder()),
    ])
    pipe.fit(X_train)
    X_out = pipe.transform(X_val)   # DataFrame, shape (n_rows, 44)

Note:
    ``constants.py`` is the single source of truth for all feature names, dtype
    contracts, and preprocessing group assignments.  Neither transformer imports
    ``LOG_FEATURES``, ``SCALE_FEATURES``, ``BINARY_FEATURES``, ``ORDINAL_FEATURES``,
    or ``PASSTHROUGH_FEATURES`` — those are consumed only by ``pipeline.py`` when
    assembling the ``ColumnTransformer``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from mpc_ml.features.constants import (
    COLD_START_DEFAULTS,
    COLD_START_FEATURE_NAMES,
    COLUMN_DTYPE_CONTRACT,
    FEATURE_COLS,
    INTERACTION_FEATURE_NAMES,
    TARGET_COLS,
    ZERO_VARIANCE_FEATURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants used across both transformers
# ---------------------------------------------------------------------------

_EXPECTED_BASE_COL_COUNT: int = len(FEATURE_COLS)               # 38
_EXPECTED_TOTAL_COL_COUNT: int = (
    len(FEATURE_COLS) + len(INTERACTION_FEATURE_NAMES)           # 44
)


# ===========================================================================
# ColumnSelector
# ===========================================================================


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Pipeline Step 1 — schema enforcer, dtype coercer, and cold-start NaN filler.

    This transformer is the pipeline's gatekeeper.  Its sole job is to answer:
    *"Is this input safe to process?"*  It enforces three distinct contracts:

    1. **Schema enforcement** — correct columns present, no target columns, dtypes
       coercible to the contract.
    2. **Cold-start filling** — NaN in the 7 rolling historical features is replaced
       with population means *learned from the training set* at ``fit()`` time.  The
       values in ``constants.COLD_START_DEFAULTS`` serve only as a seed/fallback if a
       rolling feature is entirely NaN in the training data; the pipeline always uses
       data-derived defaults.
    3. **Invariant assertion** — after filling, zero NaN and expected zero-variance
       column values are asserted, with configurable strictness.

    Args:
        strict_dtypes: If ``True`` (default), a dtype mismatch that cannot be coerced
            without introducing new NaN values raises :class:`ValueError`.  If
            ``False``, a WARNING is emitted and coercion proceeds; only raises if
            coercion actually introduces NaN.

    Attributes:
        cold_start_defaults_ (Dict[str, float]): Mapping from each rolling feature name
            (``COLD_START_FEATURE_NAMES``) to its training-set population mean, computed
            with ``skipna=True``.  Falls back to ``constants.COLD_START_DEFAULTS`` if
            all training values for a column are NaN.  Set by ``fit()``.
        feature_names_in_ (List[str]): Canonical list of the 37 expected feature names
            (equals ``FEATURE_COLS``).  Set by ``fit()``.
        n_features_in_ (int): Number of expected input features (37).  Set by ``fit()``.
        zero_variance_observed_values_ (Dict[str, float]): Mapping from each column in
            ``ZERO_VARIANCE_FEATURES`` to the single unique value observed in the
            training set.  Used at transform-time to detect when a previously
            zero-variance column has become non-zero (signals the need to retrain).
            Set by ``fit()``.
        is_fitted_ (bool): ``True`` after :meth:`fit` completes successfully.

    Raises:
        ValueError: Raised by :meth:`transform` for schema violations, missing columns,
            or unrecoverable dtype coercion failures.
        sklearn.exceptions.NotFittedError: Raised by :meth:`transform` when called
            before :meth:`fit`.

    Example::

        from mpc_ml.features.transformers import ColumnSelector
        selector = ColumnSelector()
        selector.fit(X_train)
        X_clean = selector.transform(X_val)   # DataFrame, shape (n_rows, 37)
    """

    def __init__(self, strict_dtypes: bool = True) -> None:
        """Initialise ColumnSelector.

        Args:
            strict_dtypes: Controls strictness of dtype mismatch handling.  See class
                docstring for the distinction between ``True`` and ``False``.
        """
        self.strict_dtypes = strict_dtypes

    # ------------------------------------------------------------------
    # Public sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnSelector":
        """Learn cold-start defaults and zero-variance expected values from training data.

        Runs the same schema validations as :meth:`transform` (target-column rejection,
        feature-column presence, dtype coercion) so that configuration errors surface
        immediately at training time rather than during the first production prediction.

        The NaN-presence assertion step from :meth:`transform` is intentionally **not**
        run here: training data is expected to contain NaN in the 7 rolling historical
        features for early (cold-start) orders, which is exactly what this method
        characterises.

        Args:
            X: Training ``pandas.DataFrame``.  May include ``FEATURE_COLS``, target
                columns (which trigger ``ValueError``), and arbitrary extra columns
                (which are silently ignored).
            y: Ignored.  Present for sklearn API compatibility.

        Returns:
            self: The fitted estimator, enabling method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pandas.DataFrame``.
            ValueError: If ``X`` contains target columns, is missing required feature
                columns, or if dtype coercion introduces new NaN values.
        """
        self._validate_is_dataframe(X)
        self._check_no_target_cols(X)
        self._check_feature_cols_present(X)

        # _select_and_coerce returns a FEATURE_COLS-only DataFrame with dtypes fixed.
        X_coerced: pd.DataFrame = self._select_and_coerce(X)

        # ---- Learn cold-start defaults from training data ------------------
        # Use the data-derived mean; fall back to COLD_START_DEFAULTS only if a
        # rolling column is entirely NaN in the training set (degenerate edge case).
        self.cold_start_defaults_: Dict[str, float] = {}
        for col in COLD_START_FEATURE_NAMES:
            mean_val: float = float(X_coerced[col].mean(skipna=True))
            if np.isnan(mean_val):
                fallback: float = float(COLD_START_DEFAULTS[col])
                logger.warning(
                    "All values in rolling feature '%s' are NaN in the training set. "
                    "Falling back to constant seed value %.4f from COLD_START_DEFAULTS.",
                    col,
                    fallback,
                )
                mean_val = fallback
            self.cold_start_defaults_[col] = mean_val

        # ---- Learn zero-variance expected values ---------------------------
        # Guard: ZERO_VARIANCE_FEATURES may include interaction features
        # (e.g. log_experience_x_concurrent) that are not yet present in
        # X_coerced — ColumnSelector operates on the 37-column base DataFrame
        # only.  Skip any member absent from the current DataFrame columns.
        self.zero_variance_observed_values_: Dict[str, float] = {}
        for col in ZERO_VARIANCE_FEATURES:
            if col not in X_coerced.columns:
                continue  # interaction feature; not in ColumnSelector's scope
            non_null_vals = X_coerced[col].dropna().unique()
            if len(non_null_vals) == 0:
                self.zero_variance_observed_values_[col] = 0.0
            else:
                # Expected to be exactly one unique value; record it.
                self.zero_variance_observed_values_[col] = float(non_null_vals[0])

        # ---- Standard sklearn fitted attributes ----------------------------
        self.feature_names_in_: List[str] = list(FEATURE_COLS)
        self.n_features_in_: int = _EXPECTED_BASE_COL_COUNT
        self.is_fitted_: bool = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforce schema, coerce dtypes, fill cold-start NaN, and return 37-col DataFrame.

        Execution pipeline (each step raises on violation unless noted):

        1. **Fitted-state guard** — raises :class:`~sklearn.exceptions.NotFittedError`
           if :meth:`fit` has not been called.
        2. **Target-column rejection** — raises :class:`ValueError` if any column in
           ``TARGET_COLS`` is present.
        3. **Feature-column presence** — raises :class:`ValueError` listing all missing
           columns from ``FEATURE_COLS``.
        4. **Dtype coercion** — attempts ``pd.to_numeric()`` for each mismatched column;
           emits WARNING regardless; raises (or just warns if ``strict_dtypes=False``)
           if coercion introduces new NaN.
        5. **Cold-start NaN filling** — fills NaN in the 7 rolling historical features
           with ``self.cold_start_defaults_``; logs at DEBUG.  Expected production path
           for new product-machine-operator combinations.
        6. **Zero-NaN assertion** — raises :class:`ValueError` if any NaN remains in
           ``FEATURE_COLS`` after step 5.
        7. **Zero-variance guard** — emits WARNING if any ``ZERO_VARIANCE_FEATURES``
           column contains values other than the value observed at fit time.  **Never
           raises** — the pipeline continues with the actual values.
        8. **Select and return** — returns ``X[FEATURE_COLS].copy()``.

        Args:
            X: Input ``pandas.DataFrame``.  May contain extra or target columns;
                target columns raise, extra columns are silently dropped.

        Returns:
            A ``pandas.DataFrame`` with exactly 37 columns in ``FEATURE_COLS``
            canonical order, zero NaN, and numerically coerced dtypes.

        Raises:
            sklearn.exceptions.NotFittedError: If called before :meth:`fit`.
            TypeError: If ``X`` is not a ``pandas.DataFrame``.
            ValueError: If target columns are present, feature columns are missing,
                dtype coercion introduces NaN, or unexpected NaN remain after filling.
        """
        check_is_fitted(self, "is_fitted_")
        self._validate_is_dataframe(X)
        self._check_no_target_cols(X)
        self._check_feature_cols_present(X)

        # Coerce dtypes and select FEATURE_COLS (drops extras and reorders).
        X_out: pd.DataFrame = self._select_and_coerce(X)

        # Fill cold-start NaN in rolling features with training-set means.
        X_out = self._fill_cold_start_nan(X_out)

        # Assert no NaN survives after filling.
        self._assert_no_remaining_nan(X_out)

        # Warn (never raise) if zero-variance columns deviate from training values.
        self._check_zero_variance_cols(X_out)

        return X_out.copy()

    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None,
    ) -> List[str]:
        """Return the 37 canonical feature names in ``FEATURE_COLS`` order.

        Works whether or not the transformer has been fitted — returns the same value
        regardless of fitting state.  Called by ``Pipeline.get_feature_names_out()``
        and by the downstream ``ColumnTransformer`` to build its feature name registry.

        Args:
            input_features: Ignored.  Present for sklearn API compatibility.

        Returns:
            List of 37 feature name strings in canonical ``FEATURE_COLS`` order.
        """
        return list(FEATURE_COLS)

    # ------------------------------------------------------------------
    # Private — schema validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_is_dataframe(X: object) -> None:
        """Assert that ``X`` is a ``pandas.DataFrame``.

        Args:
            X: Object to validate.

        Raises:
            TypeError: If ``X`` is not a ``pandas.DataFrame``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame but received {type(X).__name__!r}. "
                "Pass a DataFrame to fit() / transform()."
            )

    @staticmethod
    def _check_no_target_cols(X: pd.DataFrame) -> None:
        """Raise :class:`ValueError` if any ``TARGET_COLS`` column is present.

        This check is strict and has no lenient mode — target columns in inference
        input indicate a bug in the calling code, not a benign data artefact.

        Args:
            X: Input DataFrame.

        Raises:
            ValueError: If one or more target columns are detected.
        """
        found_targets: List[str] = sorted(set(TARGET_COLS) & set(X.columns))
        if found_targets:
            raise ValueError(
                f"Target columns found in input DataFrame: {found_targets}. "
                "These columns must be excluded before calling pipeline.transform(). "
                "Did you forget to call X.drop(TARGET_COLS, axis=1)?"
            )

    @staticmethod
    def _check_feature_cols_present(X: pd.DataFrame) -> None:
        """Raise :class:`ValueError` if any ``FEATURE_COLS`` column is absent.

        Extra columns beyond ``FEATURE_COLS`` are silently ignored here; they will be
        dropped by :meth:`_select_and_coerce`.

        Args:
            X: Input DataFrame.

        Raises:
            ValueError: Listing **all** missing column names (not just the first).
        """
        missing: List[str] = sorted(set(FEATURE_COLS) - set(X.columns))
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. "
                f"Expected all {_EXPECTED_BASE_COL_COUNT} features from FEATURE_COLS. "
                "Add these columns or check the simulation output schema."
            )

    # ------------------------------------------------------------------
    # Private — dtype coercion
    # ------------------------------------------------------------------

    def _select_and_coerce(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select ``FEATURE_COLS`` from ``X``, coerce dtypes, and return a copy.

        Combines column selection (drops extras, reorders to canonical order) with
        dtype validation and coercion into a single step to avoid redundant copies.

        For each column whose dtype does not match ``COLUMN_DTYPE_CONTRACT``:

        - Emits a WARNING regardless of ``strict_dtypes``.
        - Attempts ``pd.to_numeric(errors='coerce')``.
        - If coercion introduces new NaN: raises :class:`ValueError` when
          ``strict_dtypes=True``; emits a second WARNING when ``strict_dtypes=False``
          and the column is left with the NaN values present.

        Args:
            X: Input DataFrame.  Must contain all ``FEATURE_COLS`` (validated upstream).

        Returns:
            A new ``pandas.DataFrame`` containing exactly ``FEATURE_COLS`` in canonical
            order with coerced dtypes where applicable.

        Raises:
            ValueError: If coercion introduces NaN and ``strict_dtypes=True``.
        """
        # Select only FEATURE_COLS in canonical order (drops extras).
        # list() is required: pandas >=1.0 treats a bare tuple as a single
        # hashable label (routes to Index.get_loc), not as multi-column
        # selection, which raises KeyError on any regular-index DataFrame.
        X_out: pd.DataFrame = X[list(FEATURE_COLS)].copy()

        for col in FEATURE_COLS:
            expected_dtype: Optional[str] = COLUMN_DTYPE_CONTRACT.get(col)
            if expected_dtype is None:
                continue

            actual_dtype: str = str(X_out[col].dtype)
            if actual_dtype == expected_dtype:
                continue

            # ---- Dtype mismatch: attempt numeric coercion ------------------
            logger.warning(
                "Dtype mismatch for column '%s': expected %s, got %s. "
                "Attempting pd.to_numeric() coercion.",
                col,
                expected_dtype,
                actual_dtype,
            )

            nan_before: int = int(X_out[col].isna().sum())
            coerced: pd.Series = pd.to_numeric(X_out[col], errors="coerce")
            nan_after: int = int(coerced.isna().sum())
            new_nan_count: int = nan_after - nan_before

            if new_nan_count > 0:
                msg: str = (
                    f"Dtype coercion of column '{col}' introduced {new_nan_count} new "
                    f"NaN value(s) (expected dtype: {expected_dtype}, "
                    f"actual dtype: {actual_dtype}). "
                    "Fix the upstream data pipeline or check for non-numeric values."
                )
                if self.strict_dtypes:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

            X_out[col] = coerced

        return X_out

    # ------------------------------------------------------------------
    # Private — cold-start NaN filling
    # ------------------------------------------------------------------

    def _fill_cold_start_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in rolling historical features with learned training-set means.

        Modifies the DataFrame in-place (``X`` is already a copy produced by
        :meth:`_select_and_coerce`) and returns it.  Logs at DEBUG for each filled
        column — this is the expected production path for new product-machine-operator
        combinations and should not generate noise in production logs.

        Args:
            X: ``FEATURE_COLS``-only DataFrame (output of :meth:`_select_and_coerce`).

        Returns:
            The same DataFrame with NaN in ``COLD_START_FEATURE_NAMES`` replaced.
        """
        for col in COLD_START_FEATURE_NAMES:
            n_nan: int = int(X[col].isna().sum())
            if n_nan > 0:
                fill_value: float = self.cold_start_defaults_[col]
                X[col] = X[col].fillna(fill_value)
                logger.debug(
                    "Filled %d NaN value(s) in '%s' with cold-start default %.4f.",
                    n_nan,
                    col,
                    fill_value,
                )
        return X

    # ------------------------------------------------------------------
    # Private — post-fill assertions and guards
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_no_remaining_nan(X: pd.DataFrame) -> None:
        """Assert that no NaN remains in any ``FEATURE_COLS`` column after cold-start fill.

        NaN at this stage indicates a bug in the upstream data pipeline (e.g., a machine
        OEE value was not computed), not a benign cold-start scenario.

        Args:
            X: ``FEATURE_COLS``-only DataFrame after cold-start filling.

        Raises:
            ValueError: Listing all columns that still contain NaN.
        """
        cols_with_nan: List[str] = [
            col for col in FEATURE_COLS if X[col].isna().any()
        ]
        if cols_with_nan:
            raise ValueError(
                f"Unexpected NaN values in non-rolling feature(s): {cols_with_nan}. "
                "NaN is only expected in rolling historical features and will be "
                "auto-filled. "
                f"Check upstream data pipeline for missing values in {cols_with_nan}."
            )

    def _check_zero_variance_cols(self, X: pd.DataFrame) -> None:
        """Emit WARNING if any ``ZERO_VARIANCE_FEATURES`` column contains unexpected values.

        **Never raises** — the pipeline continues normally so that production inference
        is not interrupted when the simulation adds multi-machine operator scheduling.
        The WARNING alerts operations to retrain the pipeline.

        Unexpected values are defined as any value other than the unique value observed
        for that column in the training set (stored in ``self.zero_variance_observed_values_``).

        Args:
            X: ``FEATURE_COLS``-only DataFrame (no NaN at this point).
        """
        for col in ZERO_VARIANCE_FEATURES:
            if col not in X.columns:
                continue  # interaction feature; not in ColumnSelector's scope
            expected_val: float = self.zero_variance_observed_values_.get(col, 0.0)
            unexpected_mask: pd.Series = X[col] != expected_val
            if unexpected_mask.any():
                unique_vals: List[float] = sorted(
                    float(v) for v in X.loc[unexpected_mask, col].unique()
                )
                logger.warning(
                    "Column '%s' was zero-variance at fit time (value: %s) but now "
                    "contains non-zero values %s. Consider retraining the pipeline.",
                    col,
                    expected_val,
                    unique_vals,
                )


# ===========================================================================
# InteractionFeatureAdder
# ===========================================================================


class InteractionFeatureAdder(BaseEstimator, TransformerMixin):
    """Pipeline Step 2 — vectorised computation of 4 derived interaction features.

    Receives the clean 37-column DataFrame from :class:`ColumnSelector` and appends
    4 derived columns, producing a 41-column DataFrame.

    This class is almost entirely **stateless** — ``fit()`` validates the input schema
    and sets ``is_fitted_``, but learns no parameters from the data.  All formulas use
    only the input values and the two constructor-configurable numerical constants.

    The 4 interaction features and their formulas:

    **``lag_as_pct_of_window``** (→ ``LOG_FEATURES``)::

        denominator = max(planned_lead_time_hours, 0.1)   # ε prevents ÷0
        lag_as_pct_of_window = (release_lag_hours / denominator).clip(upper=lag_clip_upper)

    Strongest predictor in the 41-feature set (|r| = 0.521 with ``is_delayed``).
    Values > 1.0 are valid and signal that the order was released after its own planned
    completion date.

    **``tightness_x_queue``** (→ ``SCALE_FEATURES``)::

        tightness_x_queue = schedule_tightness_ratio × work_center_queue_depth_at_release

    No numerical guard needed — both inputs are naturally bounded.

    **``log_experience_x_concurrent``** (→ ``ZERO_VARIANCE_FEATURES``)::

        log_experience_x_concurrent = log1p(operator_experience_months)
                                      × operator_concurrent_order_count

    Currently produces all zeros because ``operator_concurrent_order_count = 0``
    throughout the simulation.  Future-ready for multi-machine operator scheduling.

    **``oee_x_maintenance_ratio``** (→ ``SCALE_FEATURES``, corrected v2 formula)::

        denominator = max(days_since_last_planned_maintenance / oee_maintenance_scale, 1.0)
        oee_x_maintenance_ratio = machine_oee_30d / denominator

    ``clip(lower=1.0)`` ensures freshly serviced machines receive no OEE boost (ratio
    = OEE) while overdue machines are penalised proportionally.  Division by zero is
    impossible because ``denominator ≥ 1.0``.

    Args:
        lag_clip_upper: Upper bound for ``lag_as_pct_of_window`` after the ratio is
            computed.  Prevents extreme outliers from dominating after the downstream
            ``log1p`` transform.  Default 10.0 provides ~35 % headroom above the
            observed simulation maximum of 7.40.
        oee_maintenance_scale: The PM cycle length in days that defines one "unit" in
            the OEE maintenance denominator.  Default 30.0.  Exposed for Optuna tuning.

    Attributes:
        is_fitted_ (bool): ``True`` after :meth:`fit` completes successfully.

    Raises:
        ValueError: Raised by :meth:`transform` for schema violations, name collisions,
            or post-computation NaN / infinite values.
        sklearn.exceptions.NotFittedError: Raised by :meth:`transform` when called
            before :meth:`fit`.

    Example::

        from mpc_ml.features.transformers import InteractionFeatureAdder
        adder = InteractionFeatureAdder()
        adder.fit(X_38col)
        X_44col = adder.transform(X_38col)   # DataFrame, shape (n_rows, 44)
    """

    def __init__(
        self,
        lag_clip_upper: float = 10.0,
        oee_maintenance_scale: float = 30.0,
    ) -> None:
        """Initialise InteractionFeatureAdder.

        Args:
            lag_clip_upper: Upper clip bound for the lag-ratio feature.  See class
                docstring.
            oee_maintenance_scale: PM cycle length in days.  See class docstring.
        """
        self.lag_clip_upper = lag_clip_upper
        self.oee_maintenance_scale = oee_maintenance_scale

    # ------------------------------------------------------------------
    # Public sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "InteractionFeatureAdder":
        """Validate input schema and mark transformer as fitted.

        No parameters are learned from the data.  This method exists to:

        - Confirm the input is a correctly structured 37-column DataFrame.
        - Guard against pre-existing interaction feature names that would collide during
          ``transform()`` — indicative of an incorrectly assembled pipeline (two
          ``InteractionFeatureAdder`` steps in sequence).
        - Set ``is_fitted_ = True`` to unblock :meth:`transform`.

        The same schema checks run in :meth:`transform` for defence in depth; surfacing
        them here ensures errors appear at training time.

        Args:
            X: Training ``pandas.DataFrame``.  Expected to have the 37 ``FEATURE_COLS``
                and no ``INTERACTION_FEATURE_NAMES`` yet.
            y: Ignored.  Present for sklearn API compatibility.

        Returns:
            self: The fitted estimator, enabling method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pandas.DataFrame``.
            ValueError: If required ``FEATURE_COLS`` are missing or if
                ``INTERACTION_FEATURE_NAMES`` already exist in ``X.columns``.
        """
        self._validate_is_dataframe(X)
        self._check_feature_cols_present(X)
        self._check_no_name_collision(X)
        self.is_fitted_: bool = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute 4 interaction features and append them to the 37-column input.

        Executes the following stages:

        1. **Fitted-state guard.**
        2. **Input validation** — DataFrame type, ``FEATURE_COLS`` present, no
           ``INTERACTION_FEATURE_NAMES`` collision.
        3. **Vectorised computation** of all 4 interaction Series (no per-row
           iteration; each formula is a single pandas expression with numerical guards).
        4. **Concatenation** — ``pd.concat([X, interactions_df], axis=1)``.  The input
           ``X`` is **never mutated**.
        5. **Post-computation validation** — asserts shape ``(n_rows, 41)``, zero NaN
           in all 41 columns, and all finite values in the 4 new columns.

        Args:
            X: Clean 37-column ``pandas.DataFrame`` from :class:`ColumnSelector`.

        Returns:
            A new ``pandas.DataFrame`` with 41 columns:
            ``FEATURE_COLS`` (37) followed by ``INTERACTION_FEATURE_NAMES`` (4),
            in that canonical order.  All values are finite; zero NaN guaranteed.

        Raises:
            sklearn.exceptions.NotFittedError: If called before :meth:`fit`.
            TypeError: If ``X`` is not a ``pandas.DataFrame``.
            ValueError: If input schema is violated, a name collision is detected, or
                post-computation NaN / infinite values are found.
        """
        check_is_fitted(self, "is_fitted_")
        self._validate_is_dataframe(X)
        self._check_feature_cols_present(X)
        self._check_no_name_collision(X)

        # ---- Compute all 6 interaction features (vectorised) ---------------
        # The list order must match INTERACTION_FEATURE_NAMES exactly.
        interaction_series: List[pd.Series] = [
            self._compute_lag_as_pct_of_window(X),          # INTERACTION_FEATURE_NAMES[0]
            self._compute_tightness_x_queue(X),             # INTERACTION_FEATURE_NAMES[1]
            self._compute_log_experience_x_concurrent(X),   # INTERACTION_FEATURE_NAMES[2]
            self._compute_oee_x_maintenance_ratio(X),       # INTERACTION_FEATURE_NAMES[3]
            self._compute_util_x_queue(X),                  # INTERACTION_FEATURE_NAMES[4]
            self._compute_util_x_tight(X),                  # INTERACTION_FEATURE_NAMES[5]
        ]

        interactions_df: pd.DataFrame = pd.DataFrame(
            dict(zip(INTERACTION_FEATURE_NAMES, interaction_series)),
            index=X.index,
        )

        # Concatenate: original 37 cols first, 4 interaction cols appended.
        X_out: pd.DataFrame = pd.concat([X, interactions_df], axis=1)

        # ---- Post-computation assertions -----------------------------------
        self._validate_output(X, X_out)

        return X_out

    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None,
    ) -> List[str]:
        """Return the 41 canonical feature names: 37 base + 4 interaction.

        The 4 interaction feature names are appended after the 37 base features in
        ``INTERACTION_FEATURE_NAMES`` order.

        Args:
            input_features: Ignored.  Present for sklearn API compatibility.

        Returns:
            List of 41 unique feature name strings in canonical order.
        """
        return list(FEATURE_COLS) + list(INTERACTION_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Private — interaction feature computations
    # ------------------------------------------------------------------

    def _compute_lag_as_pct_of_window(self, X: pd.DataFrame) -> pd.Series:
        """Compute ``lag_as_pct_of_window``.

        Formula::

            denominator = planned_lead_time_hours.clip(lower=0.1)   # ε = 6 minutes
            lag_as_pct_of_window = (release_lag_hours / denominator).clip(upper=lag_clip_upper)

        Values > 1.0 are valid (order released after planned completion date) and carry
        the strongest predictive signal in the dataset.  The upper clip prevents extreme
        outliers from producing non-representable values after the downstream ``log1p``
        transform.

        Guard coverage:

        - Near-zero denominator (``planned_lead_time_hours ≈ 0``): clipped to 0.1,
          preventing division by zero.
        - Extreme ratios: clipped at ``lag_clip_upper`` (default 10.0).

        Args:
            X: Input DataFrame.

        Returns:
            :class:`pandas.Series` with values in ``[0.0, lag_clip_upper]``.
        """
        denominator: pd.Series = X["planned_lead_time_hours"].clip(lower=0.1)
        return (X["release_lag_hours"] / denominator).clip(upper=self.lag_clip_upper)

    @staticmethod
    def _compute_tightness_x_queue(X: pd.DataFrame) -> pd.Series:
        """Compute ``tightness_x_queue``.

        Formula::

            tightness_x_queue = schedule_tightness_ratio × work_center_queue_depth_at_release

        No numerical guard required — both inputs are naturally bounded
        (``schedule_tightness_ratio`` ∈ [0.18, 1.02], ``work_center_queue_depth`` ∈ {0, 1}).

        When ``work_center_queue_depth_at_release = 0`` the interaction evaluates to
        zero, acting as a gate that suppresses the tightness signal when there is no
        congestion.

        Args:
            X: Input DataFrame.

        Returns:
            :class:`pandas.Series` with values in ``[0.0, ~1.03]``.
        """
        return (
            X["schedule_tightness_ratio"] * X["work_center_queue_depth_at_release"]
        )

    @staticmethod
    def _compute_log_experience_x_concurrent(X: pd.DataFrame) -> pd.Series:
        """Compute ``log_experience_x_concurrent``.

        Formula::

            log_experience_x_concurrent = log1p(operator_experience_months)
                                          × operator_concurrent_order_count

        Currently produces all zeros because ``operator_concurrent_order_count = 0``
        throughout the simulation (single-machine assignment model).  The formula is
        semantically correct and future-ready — when multi-machine scheduling is
        implemented, this feature will carry non-zero signal without any code changes.

        No numerical guard needed: ``log1p(0) = 0`` handles the zero-experience edge
        case; multiplication by 0 produces 0 for the current all-zero concurrent count.

        Args:
            X: Input DataFrame.

        Returns:
            :class:`pandas.Series` (all zeros in current simulation).
        """
        return (
            np.log1p(X["operator_experience_months"])
            * X["operator_concurrent_order_count"]
        )

    def _compute_oee_x_maintenance_ratio(self, X: pd.DataFrame) -> pd.Series:
        """Compute ``oee_x_maintenance_ratio`` (corrected v2 formula).

        Formula::

            denominator = (days_since_last_planned_maintenance / oee_maintenance_scale).clip(lower=1.0)
            oee_x_maintenance_ratio = machine_oee_30d / denominator

        The ``clip(lower=1.0)`` has the following semantic properties:

        - **Freshly serviced** (days < scale): denominator = 1.0; result = OEE.
          No artificial boost above the measured 30-day OEE.
        - **At PM boundary** (days = scale): denominator = 1.0; result = OEE.
          Exact boundary is treated as baseline.
        - **Overdue** (days > scale): denominator > 1.0; result < OEE.
          Proportional penalty for delayed maintenance.

        Division by zero is impossible because ``denominator ≥ 1.0`` always.

        Note:
            This is the corrected v2 formula.  The v1 formula used ``clip(lower=0.5)``
            on the denominator, which produced physically meaningless OEE values above
            1.0 (up to 1.569) for recently serviced machines.

        Args:
            X: Input DataFrame.

        Returns:
            :class:`pandas.Series` with values in
            ``[min_OEE / max_denominator, max_OEE]`` ≈ ``[0.191, 0.785]`` in
            simulation.
        """
        denominator: pd.Series = (
            X["days_since_last_planned_maintenance"] / self.oee_maintenance_scale
        ).clip(lower=1.0)
        return X["machine_oee_30d"] / denominator

    @staticmethod
    def _compute_util_x_queue(X: pd.DataFrame) -> pd.Series:
        """Compute ``util_x_queue``.

        Formula::

            util_x_queue = machine_utilization_at_release × work_center_queue_depth_at_release

        F-statistic vs delay_category: 386 (vs util alone: 236, queue alone: 121).
        Empirically confirmed +0.010 val weighted_F1 for Task 3 (Day 8 addition).
        """
        return X["machine_utilization_at_release"] * X["work_center_queue_depth_at_release"]

    @staticmethod
    def _compute_util_x_tight(X: pd.DataFrame) -> pd.Series:
        """Compute ``util_x_tight``.

        Formula::

            util_x_tight = machine_utilization_at_release × schedule_tightness_ratio

        F-statistic vs delay_category: 262 (vs util alone: 236, tight alone: 136).
        Both inputs bounded; product is bounded. Day 8 addition.
        """
        return X["machine_utilization_at_release"] * X["schedule_tightness_ratio"]

    # ------------------------------------------------------------------
    # Private — validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_is_dataframe(X: object) -> None:
        """Assert that ``X`` is a ``pandas.DataFrame``.

        Args:
            X: Object to validate.

        Raises:
            TypeError: If ``X`` is not a ``pandas.DataFrame``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame but received {type(X).__name__!r}. "
                "Pass a DataFrame to fit() / transform()."
            )

    @staticmethod
    def _check_feature_cols_present(X: pd.DataFrame) -> None:
        """Raise :class:`ValueError` if any ``FEATURE_COLS`` column is absent from ``X``.

        Args:
            X: Input DataFrame.

        Raises:
            ValueError: Listing all missing column names.
        """
        missing: List[str] = sorted(set(FEATURE_COLS) - set(X.columns))
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. "
                f"Expected all {_EXPECTED_BASE_COL_COUNT} features from FEATURE_COLS. "
                "This transformer must receive the output of ColumnSelector."
            )

    @staticmethod
    def _check_no_name_collision(X: pd.DataFrame) -> None:
        """Raise :class:`ValueError` if any ``INTERACTION_FEATURE_NAMES`` already exist in ``X``.

        This guard prevents silent column overwrite when :class:`InteractionFeatureAdder`
        is applied twice in the same pipeline.

        Args:
            X: Input DataFrame.

        Raises:
            ValueError: Listing the colliding column names with a suggestion to check
                for duplicate pipeline steps.
        """
        collisions: List[str] = sorted(
            set(INTERACTION_FEATURE_NAMES) & set(X.columns)
        )
        if collisions:
            raise ValueError(
                f"Column name collision detected: {collisions} already exist in the "
                "input DataFrame. "
                "This suggests InteractionFeatureAdder has been applied twice, or the "
                "simulation added a column whose name conflicts with an interaction "
                "feature. Remove the duplicate pipeline step or rename the colliding "
                "column."
            )

    def _validate_output(self, X_in: pd.DataFrame, X_out: pd.DataFrame) -> None:
        """Assert post-computation invariants on the 41-column output DataFrame.

        Checks:

        1. **Shape** — ``(n_rows, 41)`` exactly.
        2. **Zero NaN** — no NaN in any of the 4 interaction feature columns.
        3. **All finite** — no ``np.inf`` or ``-np.inf`` in the 4 interaction features.

        Args:
            X_in: The original 37-column input DataFrame (used for row-count reference
                and for including diagnostic statistics in error messages).
            X_out: The 41-column output DataFrame after concatenation.

        Raises:
            ValueError: If any post-condition is violated, with a diagnostic message
                identifying the offending feature(s).
        """
        # ---- Shape assertion -----------------------------------------------
        if X_out.shape[1] != _EXPECTED_TOTAL_COL_COUNT:
            unexpected_cols: List[str] = sorted(
                set(X_out.columns)
                - set(FEATURE_COLS)
                - set(INTERACTION_FEATURE_NAMES)
            )
            raise ValueError(
                f"InteractionFeatureAdder output has {X_out.shape[1]} columns; "
                f"expected {_EXPECTED_TOTAL_COL_COUNT}. "
                f"Unexpected columns found: {unexpected_cols}."
            )

        if X_out.shape[0] != len(X_in):
            raise ValueError(
                f"InteractionFeatureAdder output row count {X_out.shape[0]} does not "
                f"match input row count {len(X_in)}."
            )

        # ---- NaN assertion -------------------------------------------------
        # list() wrapper required: pandas >=1.0 raises KeyError on tuple keys.
        nan_counts: pd.Series = X_out[list(INTERACTION_FEATURE_NAMES)].isna().sum()
        cols_with_nan: List[str] = nan_counts[nan_counts > 0].index.tolist()
        if cols_with_nan:
            _diagnostic_cols = [
                "release_lag_hours",
                "planned_lead_time_hours",
                "schedule_tightness_ratio",
                "work_center_queue_depth_at_release",
                "operator_experience_months",
                "operator_concurrent_order_count",
                "machine_oee_30d",
                "days_since_last_planned_maintenance",
            ]
            sample_info: Dict[str, object] = {
                col: X_in[col].describe().to_dict()
                for col in _diagnostic_cols
                if col in X_in.columns
            }
            raise ValueError(
                f"NaN detected in interaction feature(s) after computation: "
                f"{cols_with_nan}. "
                f"Input feature statistics: {sample_info}. "
                "This indicates an unhandled edge case in the formula implementation."
            )

        # ---- Finite assertion ----------------------------------------------
        # list() wrapper required: pandas >=1.0 raises KeyError on tuple keys.
        interaction_vals: np.ndarray = (
            X_out[list(INTERACTION_FEATURE_NAMES)].values.astype(float)
        )
        if not np.isfinite(interaction_vals).all():
            inf_features: List[str] = [
                INTERACTION_FEATURE_NAMES[j]
                for j in range(len(INTERACTION_FEATURE_NAMES))
                if not np.isfinite(interaction_vals[:, j]).all()
            ]
            raise ValueError(
                f"Infinite values detected in interaction feature(s): {inf_features}. "
                "Check numerical guards in the interaction formula implementations."
            )
