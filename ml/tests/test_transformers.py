"""
ml/tests/test_transformers.py

Pytest suite for mpc_ml.features.transformers.
Coverage target: >95% of transformers.py

Classes under test
------------------
ColumnSelector       -- Pipeline Step 1 (schema, coercion, cold-start fill)
InteractionFeatureAdder -- Pipeline Step 2 (4 derived interaction features)
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
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
from mpc_ml.features.transformers import ColumnSelector, InteractionFeatureAdder

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_N_BASE_FEATURES: int = 37
_N_TOTAL_FEATURES: int = 41
_ZERO_VAR_BASE_COL: str = "operator_concurrent_order_count"


def _base_row() -> dict:
    """Return one dict with valid values for all 37 FEATURE_COLS."""
    return {
        "planned_lead_time_hours": 48.0,
        "release_lag_hours": 4.0,
        "schedule_revision_count": 0.0,
        "is_expedited": 0,
        "priority_encoded": 1,
        "quantity": 10,
        "operation_count": 3,
        "estimated_total_hours": 2.5,
        "schedule_tightness_ratio": 0.6,
        "product_complexity_score": 0.55,
        "material_bom_complexity": 4,
        "is_month_end": 0,
        "is_quarter_end": 0,
        "machine_utilization_at_release": 0.7,
        "work_center_queue_depth_at_release": 1.0,
        "machine_oee_30d": 0.72,
        "machine_unplanned_downtime_hours_30d": 1.5,
        "days_since_last_planned_maintenance": 20.0,
        "maintenance_due_within_order_window": 0,
        "changeover_required": 1,
        "changeover_complexity_score": 2.0,
        "operator_experience_months": 36,
        "operator_skill_tier_encoded": 1.0,
        "operator_concurrent_order_count": 0.0,
        "hours_into_shift_at_start": 3.0,
        "shift_type_encoded": 0,
        "material_availability_at_release": 1,
        "component_shortage_count": 0.0,
        "product_delay_rate_90d": 0.35,
        "machine_delay_rate_90d": 0.32,
        "operator_delay_rate_90d": 0.28,
        "product_x_machine_delay_rate_90d": 0.31,
        "product_first_pass_yield_90d": 0.93,
        "machine_setup_overrun_rate_90d": 0.45,
        "shift_delay_rate_30d": 0.36,
        "planned_start_day_of_week": 0.0,
        "planned_start_hour": 6,
    }


def _make_valid_df(n_rows: int = 10) -> pd.DataFrame:
    """Clean DataFrame with n_rows, correct contract dtypes, and zero NaN."""
    row = _base_row()
    df = pd.DataFrame([row] * n_rows)
    for col, dtype in COLUMN_DTYPE_CONTRACT.items():
        df[col] = df[col].astype(dtype)
    return df


def _make_single_row_df(overrides: dict | None = None) -> pd.DataFrame:
    """Single-row DataFrame with optional column overrides, contract dtypes applied."""
    row = _base_row()
    if overrides:
        row.update(overrides)
    df = pd.DataFrame([row])
    for col, dtype in COLUMN_DTYPE_CONTRACT.items():
        df[col] = df[col].astype(dtype)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return _make_valid_df(n_rows=10)


@pytest.fixture
def fitted_selector(valid_df: pd.DataFrame) -> ColumnSelector:
    sel = ColumnSelector()
    sel.fit(valid_df)
    return sel


@pytest.fixture
def valid_37col_df(valid_df: pd.DataFrame, fitted_selector: ColumnSelector) -> pd.DataFrame:
    """Clean 37-column output of ColumnSelector — feeds InteractionFeatureAdder."""
    return fitted_selector.transform(valid_df)


@pytest.fixture
def fitted_adder(valid_37col_df: pd.DataFrame) -> InteractionFeatureAdder:
    adder = InteractionFeatureAdder()
    adder.fit(valid_37col_df)
    return adder


# ===========================================================================
# TestColumnSelector
# ===========================================================================


class TestColumnSelector:

    # -----------------------------------------------------------------------
    # fit() — happy path
    # -----------------------------------------------------------------------

    def test_fit_returns_self(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        assert sel.fit(valid_df) is sel

    def test_fit_sets_is_fitted_true(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        assert sel.is_fitted_ is True

    def test_fit_sets_feature_names_in(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        assert sel.feature_names_in_ == list(FEATURE_COLS)

    def test_fit_sets_n_features_in_37(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        assert sel.n_features_in_ == _N_BASE_FEATURES

    def test_fit_cold_start_defaults_all_keys_present(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        assert set(sel.cold_start_defaults_.keys()) == set(COLD_START_FEATURE_NAMES)

    def test_fit_cold_start_defaults_values_are_finite(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        for col, val in sel.cold_start_defaults_.items():
            assert math.isfinite(val), f"cold_start_defaults_[{col!r}] is not finite: {val}"

    def test_fit_cold_start_defaults_use_training_mean_not_seed(self) -> None:
        """Stored mean must equal training-data mean, not the COLD_START_DEFAULTS seed."""
        known_mean = 0.40  # deliberately != COLD_START_DEFAULTS["product_delay_rate_90d"] (0.343)
        df = _make_valid_df()
        df["product_delay_rate_90d"] = known_mean
        sel = ColumnSelector()
        sel.fit(df)
        assert math.isclose(
            sel.cold_start_defaults_["product_delay_rate_90d"], known_mean, rel_tol=1e-6
        )

    def test_fit_cold_start_defaults_differ_from_seed_constant_when_data_differs(self) -> None:
        df = _make_valid_df()
        df["product_delay_rate_90d"] = 0.40
        sel = ColumnSelector()
        sel.fit(df)
        assert not math.isclose(
            sel.cold_start_defaults_["product_delay_rate_90d"],
            COLD_START_DEFAULTS["product_delay_rate_90d"],
            rel_tol=1e-3,
        )

    def test_fit_cold_start_all_nan_falls_back_to_seed_constant(
        self, valid_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = valid_df.copy()
        df["product_delay_rate_90d"] = np.nan
        sel = ColumnSelector()
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            sel.fit(df)
        assert math.isclose(
            sel.cold_start_defaults_["product_delay_rate_90d"],
            COLD_START_DEFAULTS["product_delay_rate_90d"],
            rel_tol=1e-6,
        )
        assert "Falling back to constant seed value" in caplog.text

    def test_fit_cold_start_all_nan_warning_names_column(
        self, valid_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = valid_df.copy()
        df["machine_delay_rate_90d"] = np.nan
        sel = ColumnSelector()
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            sel.fit(df)
        assert "machine_delay_rate_90d" in caplog.text

    def test_fit_zero_variance_observed_values_base_col_present(
        self, valid_df: pd.DataFrame
    ) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        assert _ZERO_VAR_BASE_COL in sel.zero_variance_observed_values_

    def test_fit_zero_variance_observed_value_is_zero(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        assert sel.zero_variance_observed_values_[_ZERO_VAR_BASE_COL] == 0.0

    def test_fit_interaction_zero_var_col_skipped_in_selector(
        self, valid_df: pd.DataFrame
    ) -> None:
        """log_experience_x_concurrent is not a base col; ColumnSelector must skip it."""
        sel = ColumnSelector()
        sel.fit(valid_df)
        assert "log_experience_x_concurrent" not in sel.zero_variance_observed_values_

    def test_fit_extra_columns_silently_ignored(self) -> None:
        df = _make_valid_df()
        df["totally_unrelated_column"] = 99.9
        sel = ColumnSelector()
        sel.fit(df)
        assert sel.is_fitted_ is True

    def test_fit_raises_type_error_on_non_dataframe(self) -> None:
        sel = ColumnSelector()
        with pytest.raises(TypeError, match="DataFrame"):
            sel.fit(np.ones((5, _N_BASE_FEATURES)))

    def test_fit_raises_type_error_on_list(self) -> None:
        sel = ColumnSelector()
        with pytest.raises(TypeError):
            sel.fit([[1, 2, 3]])

    def test_fit_raises_on_target_column_present(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["is_delayed"] = 0
        sel = ColumnSelector()
        with pytest.raises(ValueError, match="Target columns found"):
            sel.fit(df)

    def test_fit_raises_on_missing_feature_columns(self) -> None:
        df = _make_valid_df()
        df = df.drop(columns=["release_lag_hours", "machine_oee_30d"])
        sel = ColumnSelector()
        with pytest.raises(ValueError, match="Missing feature columns"):
            sel.fit(df)

    # -----------------------------------------------------------------------
    # transform() — happy path
    # -----------------------------------------------------------------------

    def test_transform_returns_dataframe(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        result = fitted_selector.transform(valid_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_has_37_columns(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        result = fitted_selector.transform(valid_df)
        assert result.shape[1] == _N_BASE_FEATURES

    def test_transform_preserves_row_count(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        result = fitted_selector.transform(valid_df)
        assert result.shape[0] == len(valid_df)

    def test_transform_column_order_matches_feature_cols(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        result = fitted_selector.transform(valid_df)
        assert list(result.columns) == list(FEATURE_COLS)

    def test_transform_zero_nan(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        result = fitted_selector.transform(valid_df)
        assert result.isna().sum().sum() == 0

    def test_transform_returns_copy_not_view(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        """Mutating the output must not mutate the original input."""
        col_idx = list(FEATURE_COLS).index("release_lag_hours")
        original_val = valid_df["release_lag_hours"].iloc[0]
        result = fitted_selector.transform(valid_df)
        result.iloc[0, col_idx] = -999.0
        assert valid_df["release_lag_hours"].iloc[0] == original_val

    def test_transform_drops_extra_columns(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["metadata_extra"] = "ignored"
        result = fitted_selector.transform(df)
        assert "metadata_extra" not in result.columns
        assert result.shape[1] == _N_BASE_FEATURES

    def test_transform_is_idempotent(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        first = fitted_selector.transform(valid_df)
        second = fitted_selector.transform(first)
        pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))

    # -----------------------------------------------------------------------
    # transform() — missing required columns
    # -----------------------------------------------------------------------

    def test_transform_missing_single_column_raises(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.drop(columns=["release_lag_hours"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            fitted_selector.transform(df)

    def test_transform_missing_column_names_listed_in_error(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        missing_cols = ["release_lag_hours", "machine_oee_30d", "shift_delay_rate_30d"]
        df = valid_df.drop(columns=missing_cols)
        with pytest.raises(ValueError) as exc_info:
            fitted_selector.transform(df)
        msg = str(exc_info.value)
        for col in missing_cols:
            assert col in msg

    def test_transform_missing_multiple_columns_raises(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.drop(
            columns=["release_lag_hours", "planned_lead_time_hours", "quantity"]
        )
        with pytest.raises(ValueError):
            fitted_selector.transform(df)

    # -----------------------------------------------------------------------
    # transform() — target leakage detection
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("target_col", list(TARGET_COLS))
    def test_transform_raises_on_any_target_column(
        self,
        fitted_selector: ColumnSelector,
        valid_df: pd.DataFrame,
        target_col: str,
    ) -> None:
        df = valid_df.copy()
        df[target_col] = 0
        with pytest.raises(ValueError, match="Target columns found"):
            fitted_selector.transform(df)

    def test_transform_target_column_error_names_the_column(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["is_delayed"] = 0
        with pytest.raises(ValueError, match="is_delayed"):
            fitted_selector.transform(df)

    def test_transform_multiple_target_columns_raises(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        for col in TARGET_COLS:
            df[col] = 0
        with pytest.raises(ValueError, match="Target columns found"):
            fitted_selector.transform(df)

    # -----------------------------------------------------------------------
    # transform() — dtype coercion success
    # -----------------------------------------------------------------------

    def test_transform_coercion_success_float_col_as_string(
        self,
        fitted_selector: ColumnSelector,
        valid_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Numeric strings coerce without error; WARNING is always emitted."""
        df = valid_df.copy()
        df["planned_lead_time_hours"] = df["planned_lead_time_hours"].astype(str)
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            result = fitted_selector.transform(df)
        assert "planned_lead_time_hours" in caplog.text
        assert result["planned_lead_time_hours"].isna().sum() == 0

    def test_transform_coercion_success_preserves_values(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["planned_lead_time_hours"] = df["planned_lead_time_hours"].astype(str)
        result = fitted_selector.transform(df)
        expected = valid_df["planned_lead_time_hours"].values
        np.testing.assert_allclose(result["planned_lead_time_hours"].values, expected)

    def test_transform_coercion_int_as_float_warns(
        self,
        fitted_selector: ColumnSelector,
        valid_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """int64 column passed as float64 emits WARNING."""
        df = valid_df.copy()
        df["quantity"] = df["quantity"].astype("float64")
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            fitted_selector.transform(df)
        assert "quantity" in caplog.text

    # -----------------------------------------------------------------------
    # transform() — dtype coercion failure
    # -----------------------------------------------------------------------

    def test_transform_coercion_failure_strict_true_raises(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        """Non-numeric string in non-rolling col raises immediately (strict_dtypes=True)."""
        df = valid_df.copy()
        df["planned_lead_time_hours"] = "not_a_number"
        with pytest.raises(ValueError, match="Dtype coercion"):
            fitted_selector.transform(df)

    def test_transform_coercion_failure_strict_true_names_column(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["planned_lead_time_hours"] = "BAD"
        with pytest.raises(ValueError, match="planned_lead_time_hours"):
            fitted_selector.transform(df)

    def test_transform_coercion_failure_strict_false_emits_warning(
        self, valid_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """strict_dtypes=False: coercion failure emits WARNING before NaN assertion."""
        df = valid_df.copy()
        df["planned_lead_time_hours"] = "not_a_number"
        sel = ColumnSelector(strict_dtypes=False)
        sel.fit(valid_df)
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            with pytest.raises(Exception):
                sel.transform(df)
        assert "planned_lead_time_hours" in caplog.text

    def test_transform_coercion_failure_strict_false_raises_from_nan_assertion(
        self, valid_df: pd.DataFrame
    ) -> None:
        """strict_dtypes=False still raises from _assert_no_remaining_nan for non-rolling cols."""
        df = valid_df.copy()
        df["planned_lead_time_hours"] = "not_a_number"
        sel = ColumnSelector(strict_dtypes=False)
        sel.fit(valid_df)
        with pytest.raises(ValueError, match="Unexpected NaN"):
            sel.transform(df)

    def test_transform_coercion_failure_strict_false_rolling_col_recovers(
        self, valid_df: pd.DataFrame
    ) -> None:
        """strict_dtypes=False: NaN from coercion in a COLD_START feature is filled → no error."""
        df = valid_df.copy()
        df["product_delay_rate_90d"] = "not_a_number"  # rolling col → NaN filled by cold-start
        sel = ColumnSelector(strict_dtypes=False)
        sel.fit(valid_df)
        result = sel.transform(df)
        assert result["product_delay_rate_90d"].isna().sum() == 0

    # -----------------------------------------------------------------------
    # transform() — cold-start NaN filling
    # -----------------------------------------------------------------------

    def test_transform_cold_start_fills_nan_in_rolling_feature(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["product_delay_rate_90d"] = np.nan
        result = fitted_selector.transform(df)
        assert result["product_delay_rate_90d"].isna().sum() == 0

    def test_transform_cold_start_fills_all_rolling_features(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        for col in COLD_START_FEATURE_NAMES:
            df[col] = np.nan
        result = fitted_selector.transform(df)
        for col in COLD_START_FEATURE_NAMES:
            assert result[col].isna().sum() == 0, f"NaN remains in {col}"

    def test_transform_cold_start_uses_training_mean_value(self) -> None:
        training_mean = 0.40
        df_train = _make_valid_df()
        df_train["product_delay_rate_90d"] = training_mean
        sel = ColumnSelector()
        sel.fit(df_train)
        df_infer = _make_valid_df()
        df_infer["product_delay_rate_90d"] = np.nan
        result = sel.transform(df_infer)
        assert all(
            math.isclose(v, training_mean, rel_tol=1e-6)
            for v in result["product_delay_rate_90d"]
        )

    def test_transform_cold_start_partial_nan_leaves_non_nan_unchanged(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        known_value = 0.17
        df = valid_df.copy()
        df.loc[df.index[:5], "product_delay_rate_90d"] = np.nan
        df.loc[df.index[5:], "product_delay_rate_90d"] = known_value
        result = fitted_selector.transform(df)
        for v in result.loc[result.index[5:], "product_delay_rate_90d"]:
            assert math.isclose(v, known_value, rel_tol=1e-6)

    def test_transform_cold_start_does_not_debug_log_when_no_nan(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="mpc_ml.features.transformers"):
            fitted_selector.transform(valid_df)
        assert "Filled" not in caplog.text

    def test_transform_cold_start_debug_logs_when_nan_present(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = valid_df.copy()
        df["product_delay_rate_90d"] = np.nan
        with caplog.at_level(logging.DEBUG, logger="mpc_ml.features.transformers"):
            fitted_selector.transform(df)
        assert "Filled" in caplog.text
        assert "product_delay_rate_90d" in caplog.text

    # -----------------------------------------------------------------------
    # transform() — no remaining NaN assertion
    # -----------------------------------------------------------------------

    def test_transform_non_rolling_nan_raises(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["machine_oee_30d"] = np.nan
        with pytest.raises(ValueError, match="Unexpected NaN"):
            fitted_selector.transform(df)

    def test_transform_non_rolling_nan_error_names_column(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["machine_utilization_at_release"] = np.nan
        with pytest.raises(ValueError, match="machine_utilization_at_release"):
            fitted_selector.transform(df)

    def test_transform_non_rolling_nan_error_lists_all_bad_cols(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        bad_cols = ["machine_oee_30d", "hours_into_shift_at_start"]
        df = valid_df.copy()
        for col in bad_cols:
            df[col] = np.nan
        with pytest.raises(ValueError) as exc_info:
            fitted_selector.transform(df)
        msg = str(exc_info.value)
        for col in bad_cols:
            assert col in msg

    # -----------------------------------------------------------------------
    # transform() — zero-variance warning behaviour
    # -----------------------------------------------------------------------

    def test_transform_zero_variance_deviation_emits_warning(
        self,
        fitted_selector: ColumnSelector,
        valid_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = valid_df.copy()
        df[_ZERO_VAR_BASE_COL] = 1.0
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            fitted_selector.transform(df)
        assert _ZERO_VAR_BASE_COL in caplog.text
        assert "zero-variance at fit time" in caplog.text

    def test_transform_zero_variance_deviation_does_not_raise(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df[_ZERO_VAR_BASE_COL] = 2.0
        result = fitted_selector.transform(df)
        assert result is not None

    def test_transform_zero_variance_actual_value_preserved_in_output(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df[_ZERO_VAR_BASE_COL] = 3.0
        result = fitted_selector.transform(df)
        assert (result[_ZERO_VAR_BASE_COL] == 3.0).all()

    def test_transform_zero_variance_no_warning_when_value_matches_fit(
        self,
        fitted_selector: ColumnSelector,
        valid_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            fitted_selector.transform(valid_df)
        assert "zero-variance at fit time" not in caplog.text

    def test_transform_zero_variance_warning_mentions_retraining(
        self,
        fitted_selector: ColumnSelector,
        valid_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = valid_df.copy()
        df[_ZERO_VAR_BASE_COL] = 1.0
        with caplog.at_level(logging.WARNING, logger="mpc_ml.features.transformers"):
            fitted_selector.transform(df)
        assert "retrain" in caplog.text.lower()

    # -----------------------------------------------------------------------
    # get_feature_names_out()
    # -----------------------------------------------------------------------

    def test_get_feature_names_out_returns_list(self) -> None:
        sel = ColumnSelector()
        assert isinstance(sel.get_feature_names_out(), list)

    def test_get_feature_names_out_length_37(self) -> None:
        sel = ColumnSelector()
        assert len(sel.get_feature_names_out()) == _N_BASE_FEATURES

    def test_get_feature_names_out_matches_feature_cols(self) -> None:
        sel = ColumnSelector()
        assert sel.get_feature_names_out() == list(FEATURE_COLS)

    def test_get_feature_names_out_works_before_fit(self) -> None:
        sel = ColumnSelector()
        names = sel.get_feature_names_out()
        assert names == list(FEATURE_COLS)

    def test_get_feature_names_out_works_after_fit(self, fitted_selector: ColumnSelector) -> None:
        assert fitted_selector.get_feature_names_out() == list(FEATURE_COLS)

    def test_get_feature_names_out_ignores_input_features_arg(self) -> None:
        sel = ColumnSelector()
        names = sel.get_feature_names_out(input_features=["a", "b", "z"])
        assert names == list(FEATURE_COLS)

    # -----------------------------------------------------------------------
    # check_is_fitted behaviour
    # -----------------------------------------------------------------------

    def test_transform_raises_not_fitted_error_before_fit(
        self, valid_df: pd.DataFrame
    ) -> None:
        sel = ColumnSelector()
        with pytest.raises(NotFittedError):
            sel.transform(valid_df)

    def test_check_is_fitted_does_not_raise_after_fit(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        sel.fit(valid_df)
        check_is_fitted(sel, "is_fitted_")

    def test_check_is_fitted_raises_before_fit(self) -> None:
        sel = ColumnSelector()
        with pytest.raises(NotFittedError):
            check_is_fitted(sel, "is_fitted_")

    def test_fit_transform_from_mixin_works(self, valid_df: pd.DataFrame) -> None:
        sel = ColumnSelector()
        result = sel.fit_transform(valid_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == _N_BASE_FEATURES

    # -----------------------------------------------------------------------
    # Constructor / sklearn API
    # -----------------------------------------------------------------------

    def test_strict_dtypes_default_true(self) -> None:
        sel = ColumnSelector()
        assert sel.strict_dtypes is True

    def test_strict_dtypes_false_stored(self) -> None:
        sel = ColumnSelector(strict_dtypes=False)
        assert sel.strict_dtypes is False

    def test_get_params_contains_strict_dtypes(self) -> None:
        sel = ColumnSelector(strict_dtypes=False)
        assert sel.get_params()["strict_dtypes"] is False

    def test_set_params_updates_strict_dtypes(self) -> None:
        sel = ColumnSelector()
        sel.set_params(strict_dtypes=False)
        assert sel.strict_dtypes is False

    def test_transform_raises_type_error_on_numpy_array(
        self, fitted_selector: ColumnSelector, valid_df: pd.DataFrame
    ) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            fitted_selector.transform(valid_df.values)

    def test_transform_raises_type_error_on_list(
        self, fitted_selector: ColumnSelector
    ) -> None:
        with pytest.raises(TypeError):
            fitted_selector.transform([[1, 2]])


# ===========================================================================
# TestInteractionFeatureAdder
# ===========================================================================


class TestInteractionFeatureAdder:

    # -----------------------------------------------------------------------
    # fit() — happy path
    # -----------------------------------------------------------------------

    def test_fit_returns_self(self, valid_37col_df: pd.DataFrame) -> None:
        adder = InteractionFeatureAdder()
        assert adder.fit(valid_37col_df) is adder

    def test_fit_sets_is_fitted_true(self, valid_37col_df: pd.DataFrame) -> None:
        adder = InteractionFeatureAdder()
        adder.fit(valid_37col_df)
        assert adder.is_fitted_ is True

    def test_fit_ignores_y_argument(self, valid_37col_df: pd.DataFrame) -> None:
        adder = InteractionFeatureAdder()
        y = pd.Series(range(len(valid_37col_df)))
        adder.fit(valid_37col_df, y=y)
        assert adder.is_fitted_ is True

    def test_fit_raises_type_error_on_non_dataframe(self) -> None:
        adder = InteractionFeatureAdder()
        with pytest.raises(TypeError, match="DataFrame"):
            adder.fit(np.ones((5, _N_BASE_FEATURES)))

    def test_fit_raises_on_missing_feature_cols(self) -> None:
        adder = InteractionFeatureAdder()
        df = _make_valid_df().drop(columns=["release_lag_hours"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            adder.fit(df)

    def test_fit_raises_on_interaction_col_collision(self, valid_37col_df: pd.DataFrame) -> None:
        adder = InteractionFeatureAdder()
        df = valid_37col_df.copy()
        df["lag_as_pct_of_window"] = 0.5
        with pytest.raises(ValueError, match="[Cc]ollision|collision"):
            adder.fit(df)

    @pytest.mark.parametrize("interaction_col", list(INTERACTION_FEATURE_NAMES))
    def test_fit_raises_on_any_interaction_col_present(
        self, valid_37col_df: pd.DataFrame, interaction_col: str
    ) -> None:
        adder = InteractionFeatureAdder()
        df = valid_37col_df.copy()
        df[interaction_col] = 0.0
        with pytest.raises(ValueError):
            adder.fit(df)

    # -----------------------------------------------------------------------
    # transform() — happy path
    # -----------------------------------------------------------------------

    def test_transform_returns_dataframe(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        assert isinstance(fitted_adder.transform(valid_37col_df), pd.DataFrame)

    def test_transform_has_41_columns(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        assert fitted_adder.transform(valid_37col_df).shape[1] == _N_TOTAL_FEATURES

    def test_transform_preserves_row_count(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        assert result.shape[0] == len(valid_37col_df)

    def test_transform_base_cols_all_present(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_transform_interaction_cols_all_present(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        for col in INTERACTION_FEATURE_NAMES:
            assert col in result.columns

    def test_transform_interaction_cols_are_last_four(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        assert list(result.columns[-4:]) == list(INTERACTION_FEATURE_NAMES)

    def test_transform_base_col_values_unchanged(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        pd.testing.assert_frame_equal(
            result[list(FEATURE_COLS)].reset_index(drop=True),
            valid_37col_df[list(FEATURE_COLS)].reset_index(drop=True),
        )

    # -----------------------------------------------------------------------
    # Interaction formula — lag_as_pct_of_window
    # -----------------------------------------------------------------------

    def test_lag_formula_basic_ratio(self) -> None:
        """lag=4, window=48 → 4/48 ≈ 0.0833."""
        df = _make_single_row_df({"release_lag_hours": 4.0, "planned_lead_time_hours": 48.0})
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["lag_as_pct_of_window"].iloc[0], 4.0 / 48.0, rel_tol=1e-6)

    def test_lag_clip_upper_applied_at_default_10(self) -> None:
        """1000/48 > 10 → clipped to 10.0."""
        df = _make_single_row_df({"release_lag_hours": 1000.0, "planned_lead_time_hours": 48.0})
        adder = InteractionFeatureAdder(lag_clip_upper=10.0)
        adder.fit(df)
        result = adder.transform(df)
        assert result["lag_as_pct_of_window"].iloc[0] == 10.0

    def test_lag_clip_upper_custom_value_respected(self) -> None:
        """Custom lag_clip_upper=5.0: 100/10 = 10 → clipped to 5.0."""
        df = _make_single_row_df({"release_lag_hours": 100.0, "planned_lead_time_hours": 10.0})
        adder = InteractionFeatureAdder(lag_clip_upper=5.0)
        adder.fit(df)
        result = adder.transform(df)
        assert result["lag_as_pct_of_window"].iloc[0] == 5.0

    def test_lag_denominator_clip_near_zero_lead_time(self) -> None:
        """planned_lead_time_hours=0.05 → denominator clipped to 0.1 → no div/0."""
        df = _make_single_row_df({"release_lag_hours": 4.0, "planned_lead_time_hours": 0.05})
        adder = InteractionFeatureAdder(lag_clip_upper=10.0)
        adder.fit(df)
        result = adder.transform(df)
        # 4.0 / 0.1 = 40 → clipped to 10.0
        assert result["lag_as_pct_of_window"].iloc[0] == 10.0

    def test_lag_denominator_clip_exactly_zero_lead_time(self) -> None:
        """planned_lead_time_hours=0.0 → denominator=0.1 → finite result."""
        df = _make_single_row_df({"release_lag_hours": 1.0, "planned_lead_time_hours": 0.0})
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        val = result["lag_as_pct_of_window"].iloc[0]
        assert math.isfinite(val)

    def test_lag_zero_release_lag_gives_zero(self) -> None:
        df = _make_single_row_df({"release_lag_hours": 0.0, "planned_lead_time_hours": 48.0})
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert result["lag_as_pct_of_window"].iloc[0] == 0.0

    def test_lag_greater_than_one_is_valid(self) -> None:
        """Order released after planned window: lag > window → ratio > 1.0, not clipped."""
        df = _make_single_row_df({"release_lag_hours": 60.0, "planned_lead_time_hours": 48.0})
        adder = InteractionFeatureAdder(lag_clip_upper=10.0)
        adder.fit(df)
        result = adder.transform(df)
        expected = 60.0 / 48.0
        assert math.isclose(result["lag_as_pct_of_window"].iloc[0], expected, rel_tol=1e-6)

    # -----------------------------------------------------------------------
    # Interaction formula — tightness_x_queue
    # -----------------------------------------------------------------------

    def test_tightness_x_queue_basic_product(self) -> None:
        """tightness=0.6, queue=1.0 → 0.6."""
        df = _make_single_row_df(
            {"schedule_tightness_ratio": 0.6, "work_center_queue_depth_at_release": 1.0}
        )
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["tightness_x_queue"].iloc[0], 0.6, rel_tol=1e-6)

    def test_tightness_x_queue_zero_when_no_queue(self) -> None:
        """queue=0 gates the signal completely."""
        df = _make_single_row_df(
            {"schedule_tightness_ratio": 0.9, "work_center_queue_depth_at_release": 0.0}
        )
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert result["tightness_x_queue"].iloc[0] == 0.0

    def test_tightness_x_queue_high_tightness_with_queue(self) -> None:
        df = _make_single_row_df(
            {"schedule_tightness_ratio": 0.95, "work_center_queue_depth_at_release": 1.0}
        )
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["tightness_x_queue"].iloc[0], 0.95, rel_tol=1e-6)

    # -----------------------------------------------------------------------
    # Interaction formula — log_experience_x_concurrent
    # -----------------------------------------------------------------------

    def test_log_experience_x_concurrent_zero_when_concurrent_zero(self) -> None:
        """All-zero concurrent count → all-zero result (current simulation default)."""
        df = _make_single_row_df(
            {"operator_experience_months": 36, "operator_concurrent_order_count": 0.0}
        )
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert result["log_experience_x_concurrent"].iloc[0] == 0.0

    def test_log_experience_x_concurrent_formula_nonzero_concurrent(self) -> None:
        """log1p(exp) × concurrent gives expected value when concurrent > 0."""
        exp_months = 36
        concurrent = 2.0
        expected = math.log1p(exp_months) * concurrent
        df = _make_single_row_df(
            {
                "operator_experience_months": exp_months,
                "operator_concurrent_order_count": concurrent,
            }
        )
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(
            result["log_experience_x_concurrent"].iloc[0], expected, rel_tol=1e-6
        )

    def test_log_experience_x_concurrent_zero_experience_with_nonzero_concurrent(
        self,
    ) -> None:
        """log1p(0) = 0 → result = 0 even when concurrent > 0."""
        df = _make_single_row_df(
            {"operator_experience_months": 0, "operator_concurrent_order_count": 5.0}
        )
        # Override dtype manually for zero experience
        df["operator_experience_months"] = pd.array([0], dtype="int64")
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert result["log_experience_x_concurrent"].iloc[0] == 0.0

    # -----------------------------------------------------------------------
    # Interaction formula — oee_x_maintenance_ratio
    # -----------------------------------------------------------------------

    def test_oee_freshly_serviced_denominator_clipped_to_one(self) -> None:
        """days < scale → denom clipped to 1.0 → result = OEE (no boost)."""
        df = _make_single_row_df(
            {"machine_oee_30d": 0.72, "days_since_last_planned_maintenance": 10.0}
        )
        adder = InteractionFeatureAdder(oee_maintenance_scale=30.0)
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["oee_x_maintenance_ratio"].iloc[0], 0.72, rel_tol=1e-6)

    def test_oee_exactly_at_pm_boundary(self) -> None:
        """days = scale → denom = 1.0 → result = OEE."""
        df = _make_single_row_df(
            {"machine_oee_30d": 0.72, "days_since_last_planned_maintenance": 30.0}
        )
        adder = InteractionFeatureAdder(oee_maintenance_scale=30.0)
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["oee_x_maintenance_ratio"].iloc[0], 0.72, rel_tol=1e-6)

    def test_oee_overdue_penalises_proportionally(self) -> None:
        """days = 2× scale → denom = 2.0 → result = OEE / 2."""
        df = _make_single_row_df(
            {"machine_oee_30d": 0.60, "days_since_last_planned_maintenance": 60.0}
        )
        adder = InteractionFeatureAdder(oee_maintenance_scale=30.0)
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["oee_x_maintenance_ratio"].iloc[0], 0.30, rel_tol=1e-6)

    def test_oee_result_never_exceeds_measured_oee(self) -> None:
        """Denominator ≥ 1.0 guarantees result ≤ OEE."""
        df = _make_single_row_df(
            {"machine_oee_30d": 0.80, "days_since_last_planned_maintenance": 0.0}
        )
        adder = InteractionFeatureAdder(oee_maintenance_scale=30.0)
        adder.fit(df)
        result = adder.transform(df)
        assert result["oee_x_maintenance_ratio"].iloc[0] <= 0.80 + 1e-9

    def test_oee_custom_maintenance_scale_shifts_penalty_threshold(self) -> None:
        """With scale=15, days=15 → denom=1.0 → result = OEE (boundary not yet penalised)."""
        df = _make_single_row_df(
            {"machine_oee_30d": 0.70, "days_since_last_planned_maintenance": 15.0}
        )
        adder = InteractionFeatureAdder(oee_maintenance_scale=15.0)
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["oee_x_maintenance_ratio"].iloc[0], 0.70, rel_tol=1e-6)

    def test_oee_custom_maintenance_scale_penalises_beyond_boundary(self) -> None:
        """With scale=15, days=30 → denom=2.0 → result = OEE / 2."""
        df = _make_single_row_df(
            {"machine_oee_30d": 0.80, "days_since_last_planned_maintenance": 30.0}
        )
        adder = InteractionFeatureAdder(oee_maintenance_scale=15.0)
        adder.fit(df)
        result = adder.transform(df)
        assert math.isclose(result["oee_x_maintenance_ratio"].iloc[0], 0.40, rel_tol=1e-6)

    def test_oee_denominator_always_at_least_one_prevents_div_by_zero(self) -> None:
        """days = 0 → raw denom = 0.0/scale = 0.0 → clipped to 1.0 → no error."""
        df = _make_single_row_df(
            {"machine_oee_30d": 0.75, "days_since_last_planned_maintenance": 0.0}
        )
        adder = InteractionFeatureAdder(oee_maintenance_scale=30.0)
        adder.fit(df)
        result = adder.transform(df)
        assert math.isfinite(result["oee_x_maintenance_ratio"].iloc[0])

    # -----------------------------------------------------------------------
    # No NaN output
    # -----------------------------------------------------------------------

    def test_no_nan_anywhere_in_output(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        assert result.isna().sum().sum() == 0

    @pytest.mark.parametrize("interaction_col", list(INTERACTION_FEATURE_NAMES))
    def test_no_nan_in_each_interaction_col(
        self,
        fitted_adder: InteractionFeatureAdder,
        valid_37col_df: pd.DataFrame,
        interaction_col: str,
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        assert result[interaction_col].isna().sum() == 0

    # -----------------------------------------------------------------------
    # No inf output
    # -----------------------------------------------------------------------

    def test_no_inf_in_interaction_cols(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        result = fitted_adder.transform(valid_37col_df)
        interaction_vals = result[list(INTERACTION_FEATURE_NAMES)].values.astype(float)
        assert np.isfinite(interaction_vals).all()

    def test_no_inf_with_extreme_lag_and_zero_lead_time(self) -> None:
        """Near-zero denominator in lag formula must not produce inf."""
        df = _make_single_row_df(
            {"release_lag_hours": 5.0, "planned_lead_time_hours": 0.0}
        )
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        interaction_vals = result[list(INTERACTION_FEATURE_NAMES)].values.astype(float)
        assert np.isfinite(interaction_vals).all()

    def test_no_inf_with_zero_oee_maintenance_days(self) -> None:
        df = _make_single_row_df(
            {"machine_oee_30d": 0.65, "days_since_last_planned_maintenance": 0.0}
        )
        adder = InteractionFeatureAdder()
        adder.fit(df)
        result = adder.transform(df)
        assert np.isfinite(result[list(INTERACTION_FEATURE_NAMES)].values.astype(float)).all()

    # -----------------------------------------------------------------------
    # Input immutability
    # -----------------------------------------------------------------------

    def test_transform_does_not_mutate_input_df(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        snapshot = valid_37col_df.copy()
        fitted_adder.transform(valid_37col_df)
        pd.testing.assert_frame_equal(valid_37col_df, snapshot)

    def test_transform_does_not_add_cols_to_input_df(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        original_cols = list(valid_37col_df.columns)
        fitted_adder.transform(valid_37col_df)
        assert list(valid_37col_df.columns) == original_cols

    def test_transform_does_not_mutate_input_values(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        first_val = valid_37col_df["release_lag_hours"].iloc[0]
        fitted_adder.transform(valid_37col_df)
        assert valid_37col_df["release_lag_hours"].iloc[0] == first_val

    # -----------------------------------------------------------------------
    # Collision detection
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("interaction_col", list(INTERACTION_FEATURE_NAMES))
    def test_transform_raises_on_any_existing_interaction_col(
        self,
        fitted_adder: InteractionFeatureAdder,
        valid_37col_df: pd.DataFrame,
        interaction_col: str,
    ) -> None:
        df = valid_37col_df.copy()
        df[interaction_col] = 0.0
        with pytest.raises(ValueError):
            fitted_adder.transform(df)

    def test_transform_collision_error_names_the_colliding_col(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        df = valid_37col_df.copy()
        df["tightness_x_queue"] = 0.0
        with pytest.raises(ValueError, match="tightness_x_queue"):
            fitted_adder.transform(df)

    # -----------------------------------------------------------------------
    # get_feature_names_out()
    # -----------------------------------------------------------------------

    def test_get_feature_names_out_returns_list(
        self, fitted_adder: InteractionFeatureAdder
    ) -> None:
        assert isinstance(fitted_adder.get_feature_names_out(), list)

    def test_get_feature_names_out_length_41(
        self, fitted_adder: InteractionFeatureAdder
    ) -> None:
        assert len(fitted_adder.get_feature_names_out()) == _N_TOTAL_FEATURES

    def test_get_feature_names_out_base_cols_first_37(
        self, fitted_adder: InteractionFeatureAdder
    ) -> None:
        names = fitted_adder.get_feature_names_out()
        assert names[:37] == list(FEATURE_COLS)

    def test_get_feature_names_out_interaction_cols_last_4(
        self, fitted_adder: InteractionFeatureAdder
    ) -> None:
        names = fitted_adder.get_feature_names_out()
        assert names[37:] == list(INTERACTION_FEATURE_NAMES)

    def test_get_feature_names_out_works_before_fit(self) -> None:
        adder = InteractionFeatureAdder()
        names = adder.get_feature_names_out()
        assert names == list(FEATURE_COLS) + list(INTERACTION_FEATURE_NAMES)

    def test_get_feature_names_out_ignores_input_features_arg(
        self, fitted_adder: InteractionFeatureAdder
    ) -> None:
        names = fitted_adder.get_feature_names_out(input_features=["x", "y"])
        assert len(names) == _N_TOTAL_FEATURES

    # -----------------------------------------------------------------------
    # check_is_fitted behaviour
    # -----------------------------------------------------------------------

    def test_transform_raises_not_fitted_error_before_fit(
        self, valid_37col_df: pd.DataFrame
    ) -> None:
        adder = InteractionFeatureAdder()
        with pytest.raises(NotFittedError):
            adder.transform(valid_37col_df)

    def test_check_is_fitted_does_not_raise_after_fit(
        self, valid_37col_df: pd.DataFrame
    ) -> None:
        adder = InteractionFeatureAdder()
        adder.fit(valid_37col_df)
        check_is_fitted(adder, "is_fitted_")

    def test_check_is_fitted_raises_before_fit(self) -> None:
        adder = InteractionFeatureAdder()
        with pytest.raises(NotFittedError):
            check_is_fitted(adder, "is_fitted_")

    # -----------------------------------------------------------------------
    # Constructor / sklearn API
    # -----------------------------------------------------------------------

    def test_default_lag_clip_upper_is_10(self) -> None:
        assert InteractionFeatureAdder().lag_clip_upper == 10.0

    def test_default_oee_maintenance_scale_is_30(self) -> None:
        assert InteractionFeatureAdder().oee_maintenance_scale == 30.0

    def test_custom_constructor_params_stored(self) -> None:
        adder = InteractionFeatureAdder(lag_clip_upper=7.5, oee_maintenance_scale=45.0)
        assert adder.lag_clip_upper == 7.5
        assert adder.oee_maintenance_scale == 45.0

    def test_get_params_returns_constructor_params(self) -> None:
        adder = InteractionFeatureAdder(lag_clip_upper=6.0, oee_maintenance_scale=60.0)
        params = adder.get_params()
        assert params["lag_clip_upper"] == 6.0
        assert params["oee_maintenance_scale"] == 60.0

    def test_set_params_updates_lag_clip_upper(self) -> None:
        adder = InteractionFeatureAdder()
        adder.set_params(lag_clip_upper=3.0)
        assert adder.lag_clip_upper == 3.0

    def test_set_params_updates_oee_maintenance_scale(self) -> None:
        adder = InteractionFeatureAdder()
        adder.set_params(oee_maintenance_scale=90.0)
        assert adder.oee_maintenance_scale == 90.0

    def test_transform_raises_type_error_on_non_dataframe(
        self, fitted_adder: InteractionFeatureAdder
    ) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            fitted_adder.transform(np.ones((5, _N_BASE_FEATURES)))

    def test_transform_raises_on_missing_feature_cols(
        self, fitted_adder: InteractionFeatureAdder, valid_37col_df: pd.DataFrame
    ) -> None:
        df = valid_37col_df.drop(columns=["release_lag_hours"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            fitted_adder.transform(df)

    def test_fit_transform_from_mixin_works(self, valid_37col_df: pd.DataFrame) -> None:
        adder = InteractionFeatureAdder()
        result = adder.fit_transform(valid_37col_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == _N_TOTAL_FEATURES

    # -----------------------------------------------------------------------
    # Multi-row formula correctness (spot-check all 4 simultaneously)
    # -----------------------------------------------------------------------

    def test_all_four_formulas_correct_on_multi_row_df(self) -> None:
        rows = []
        params = [
            {"release_lag_hours": 4.0, "planned_lead_time_hours": 48.0,
             "schedule_tightness_ratio": 0.5, "work_center_queue_depth_at_release": 1.0,
             "operator_experience_months": 12, "operator_concurrent_order_count": 0.0,
             "machine_oee_30d": 0.70, "days_since_last_planned_maintenance": 20.0},
            {"release_lag_hours": 24.0, "planned_lead_time_hours": 48.0,
             "schedule_tightness_ratio": 0.8, "work_center_queue_depth_at_release": 0.0,
             "operator_experience_months": 60, "operator_concurrent_order_count": 0.0,
             "machine_oee_30d": 0.65, "days_since_last_planned_maintenance": 60.0},
            {"release_lag_hours": 0.0, "planned_lead_time_hours": 10.0,
             "schedule_tightness_ratio": 0.3, "work_center_queue_depth_at_release": 1.0,
             "operator_experience_months": 6, "operator_concurrent_order_count": 0.0,
             "machine_oee_30d": 0.75, "days_since_last_planned_maintenance": 30.0},
        ]
        for p in params:
            row = _base_row()
            row.update(p)
            rows.append(row)
        df = pd.DataFrame(rows)
        for col, dtype in COLUMN_DTYPE_CONTRACT.items():
            df[col] = df[col].astype(dtype)

        scale = 30.0
        clip_upper = 10.0
        adder = InteractionFeatureAdder(lag_clip_upper=clip_upper, oee_maintenance_scale=scale)
        adder.fit(df)
        result = adder.transform(df)

        for i, p in enumerate(params):
            denom_lag = max(p["planned_lead_time_hours"], 0.1)
            exp_lag = min(p["release_lag_hours"] / denom_lag, clip_upper)
            assert math.isclose(
                result["lag_as_pct_of_window"].iloc[i], exp_lag, rel_tol=1e-6
            ), f"lag row {i}"

            exp_tightness = (
                p["schedule_tightness_ratio"] * p["work_center_queue_depth_at_release"]
            )
            assert math.isclose(
                result["tightness_x_queue"].iloc[i], exp_tightness, rel_tol=1e-6
            ), f"tightness row {i}"

            exp_concurrent = (
                math.log1p(p["operator_experience_months"])
                * p["operator_concurrent_order_count"]
            )
            assert math.isclose(
                result["log_experience_x_concurrent"].iloc[i], exp_concurrent, rel_tol=1e-6
            ), f"log_exp row {i}"

            denom_oee = max(p["days_since_last_planned_maintenance"] / scale, 1.0)
            exp_oee = p["machine_oee_30d"] / denom_oee
            assert math.isclose(
                result["oee_x_maintenance_ratio"].iloc[i], exp_oee, rel_tol=1e-6
            ), f"oee row {i}"
