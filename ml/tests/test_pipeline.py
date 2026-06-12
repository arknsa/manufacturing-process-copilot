"""
ml/tests/test_pipeline.py

Pytest suite for mpc_ml.features.pipeline.
Coverage target: >95% of pipeline.py

Verified areas
--------------
build_pipeline()              -- structure, step types, ColumnTransformer branches
get_feature_names()           -- order, count, consistency
_validate_feature_coverage()  -- happy path + all 3 error conditions
fit() / transform()           -- output shape, dtypes, NaN, passthrough values
fit_transform()               -- matches fit-then-transform
serialization                 -- pickle roundtrip before and after fit
invalid input                 -- TypeError / ValueError on bad data
no target leakage             -- all 4 TARGET_COLS rejected
feature name consistency      -- output columns == get_feature_names()
output feature count          -- always 41 across batch sizes
transformer assignment        -- each feature in exactly one branch
"""
from __future__ import annotations

import math
import pickle
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import mpc_ml.features.pipeline as pipeline_module
from mpc_ml.features.constants import (
    BINARY_FEATURES,
    COLUMN_DTYPE_CONTRACT,
    FEATURE_COLS,
    INTERACTION_FEATURE_NAMES,
    LOG_FEATURES,
    ORDINAL_FEATURES,
    PASSTHROUGH_FEATURES,
    SCALE_FEATURES,
    TARGET_COLS,
    ZERO_VARIANCE_FEATURES,
)
from mpc_ml.features.pipeline import (
    _BINARY_COLS,
    _LOG_COLS,
    _ORDERED_FEATURE_NAMES,
    _ORDINAL_COLS,
    _PASSTHROUGH_COLS,
    _SCALE_COLS,
    _ZERO_VAR_COLS,
    _validate_feature_coverage,
    build_pipeline,
    get_feature_names,
)
from mpc_ml.features.transformers import ColumnSelector, InteractionFeatureAdder

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_N_OUTPUT_FEATURES: int = 41
_N_BASE_FEATURES: int = 37

_EXPECTED_BRANCH_SIZES = {
    "log_scale": 7,           # 6 LOG_FEATURES + lag_as_pct_of_window
    "scale_only": 17,         # 15 SCALE_FEATURES + tightness_x_queue + oee_x_maintenance_ratio
    "binary": 8,
    "ordinal": 4,
    "passthrough_counts": 3,  # PASSTHROUGH_FEATURES minus the 1 zero-variance member
    "zero_variance": 2,       # operator_concurrent_order_count + log_experience_x_concurrent
}

_EXPECTED_STEP_NAMES: List[str] = [
    "column_selector",
    "interaction_adder",
    "column_transformer",
]

# Branch ordering in _ORDERED_FEATURE_NAMES:
#   [0..6]   log_scale
#   [7..23]  scale_only
#   [24..31] binary
#   [32..35] ordinal
#   [36..38] passthrough_counts
#   [39..40] zero_variance
_BRANCH_START = {
    "log_scale": 0,
    "scale_only": 7,
    "binary": 24,
    "ordinal": 32,
    "passthrough_counts": 36,
    "zero_variance": 39,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_row() -> dict:
    """Valid values for all 37 FEATURE_COLS."""
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


def _make_valid_df(n_rows: int = 50) -> pd.DataFrame:
    """n_rows DataFrame with contract dtypes and zero NaN."""
    row = _base_row()
    df = pd.DataFrame([row] * n_rows)
    for col, dtype in COLUMN_DTYPE_CONTRACT.items():
        df[col] = df[col].astype(dtype)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def valid_df() -> pd.DataFrame:
    return _make_valid_df(n_rows=50)


@pytest.fixture(scope="module")
def fitted_pipeline(valid_df: pd.DataFrame) -> Pipeline:
    pipe = build_pipeline()
    pipe.fit(valid_df)
    return pipe


@pytest.fixture(scope="module")
def transformed_output(fitted_pipeline: Pipeline, valid_df: pd.DataFrame) -> pd.DataFrame:
    return fitted_pipeline.transform(valid_df)


# ===========================================================================
# TestBuildPipelineStructure
# ===========================================================================


class TestBuildPipelineStructure:
    """build_pipeline() must return a correctly structured 3-step Pipeline."""

    def test_returns_pipeline_instance(self) -> None:
        assert isinstance(build_pipeline(), Pipeline)

    def test_pipeline_has_three_steps(self) -> None:
        assert len(build_pipeline().steps) == 3

    def test_pipeline_step_names(self) -> None:
        pipe = build_pipeline()
        assert [name for name, _ in pipe.steps] == _EXPECTED_STEP_NAMES

    def test_first_step_is_column_selector(self) -> None:
        pipe = build_pipeline()
        assert isinstance(pipe.named_steps["column_selector"], ColumnSelector)

    def test_second_step_is_interaction_adder(self) -> None:
        pipe = build_pipeline()
        assert isinstance(pipe.named_steps["interaction_adder"], InteractionFeatureAdder)

    def test_third_step_is_column_transformer(self) -> None:
        pipe = build_pipeline()
        assert isinstance(pipe.named_steps["column_transformer"], ColumnTransformer)

    def test_each_call_returns_new_independent_instance(self) -> None:
        pipe1 = build_pipeline()
        pipe2 = build_pipeline()
        assert pipe1 is not pipe2

    def test_new_pipeline_is_unfitted(self) -> None:
        pipe = build_pipeline()
        selector = pipe.named_steps["column_selector"]
        assert not getattr(selector, "is_fitted_", False)


# ===========================================================================
# TestColumnTransformerBranches
# ===========================================================================


class TestColumnTransformerBranches:
    """Verify ColumnTransformer branch names, transformers, and column assignments."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.ct: ColumnTransformer = build_pipeline().named_steps["column_transformer"]

    def test_has_six_branches(self) -> None:
        assert len(self.ct.transformers) == 6

    def test_branch_names_in_order(self) -> None:
        names = [name for name, _, _ in self.ct.transformers]
        assert names == list(_EXPECTED_BRANCH_SIZES.keys())

    def test_remainder_is_drop(self) -> None:
        assert self.ct.remainder == "drop"

    def test_verbose_feature_names_out_is_false(self) -> None:
        assert self.ct.verbose_feature_names_out is False

    def test_n_jobs_is_1(self) -> None:
        assert self.ct.n_jobs == 1

    # ---- Branch 1: log_scale -----------------------------------------------

    def test_log_scale_transformer_is_pipeline(self) -> None:
        _, transformer, _ = self.ct.transformers[0]
        assert isinstance(transformer, Pipeline)

    def test_log_scale_sub_pipeline_log_step_is_function_transformer(self) -> None:
        _, transformer, _ = self.ct.transformers[0]
        assert isinstance(transformer.named_steps["log"], FunctionTransformer)

    def test_log_scale_sub_pipeline_log_step_uses_log1p(self) -> None:
        _, transformer, _ = self.ct.transformers[0]
        assert transformer.named_steps["log"].func is np.log1p

    def test_log_scale_sub_pipeline_log_step_validate_false(self) -> None:
        _, transformer, _ = self.ct.transformers[0]
        assert transformer.named_steps["log"].validate is False

    def test_log_scale_sub_pipeline_scaler_step_is_standard_scaler(self) -> None:
        _, transformer, _ = self.ct.transformers[0]
        assert isinstance(transformer.named_steps["scaler"], StandardScaler)

    def test_log_scale_columns(self) -> None:
        _, _, cols = self.ct.transformers[0]
        assert cols == _LOG_COLS

    def test_log_scale_column_count(self) -> None:
        _, _, cols = self.ct.transformers[0]
        assert len(cols) == _EXPECTED_BRANCH_SIZES["log_scale"]

    # ---- Branch 2: scale_only ----------------------------------------------

    def test_scale_only_transformer_is_standard_scaler(self) -> None:
        _, transformer, _ = self.ct.transformers[1]
        assert isinstance(transformer, StandardScaler)

    def test_scale_only_columns(self) -> None:
        _, _, cols = self.ct.transformers[1]
        assert cols == _SCALE_COLS

    def test_scale_only_column_count(self) -> None:
        _, _, cols = self.ct.transformers[1]
        assert len(cols) == _EXPECTED_BRANCH_SIZES["scale_only"]

    # ---- Branches 3–6: passthrough -----------------------------------------

    @pytest.mark.parametrize(
        "branch_idx, branch_name",
        [(2, "binary"), (3, "ordinal"), (4, "passthrough_counts"), (5, "zero_variance")],
    )
    def test_passthrough_branch_transformer(
        self, branch_idx: int, branch_name: str
    ) -> None:
        _, transformer, _ = self.ct.transformers[branch_idx]
        assert transformer == "passthrough"

    def test_binary_columns(self) -> None:
        _, _, cols = self.ct.transformers[2]
        assert cols == _BINARY_COLS

    def test_ordinal_columns(self) -> None:
        _, _, cols = self.ct.transformers[3]
        assert cols == _ORDINAL_COLS

    def test_passthrough_counts_columns(self) -> None:
        _, _, cols = self.ct.transformers[4]
        assert cols == _PASSTHROUGH_COLS

    def test_zero_variance_columns(self) -> None:
        _, _, cols = self.ct.transformers[5]
        assert cols == _ZERO_VAR_COLS

    @pytest.mark.parametrize(
        "branch_name, expected_size",
        list(_EXPECTED_BRANCH_SIZES.items()),
    )
    def test_each_branch_column_count(
        self, branch_name: str, expected_size: int
    ) -> None:
        by_name = {name: cols for name, _, cols in self.ct.transformers}
        assert len(by_name[branch_name]) == expected_size


# ===========================================================================
# TestGetFeatureNames
# ===========================================================================


class TestGetFeatureNames:
    """get_feature_names() must return the ordered 41-element feature list."""

    def test_returns_list(self) -> None:
        assert isinstance(get_feature_names(), list)

    def test_length_41(self) -> None:
        assert len(get_feature_names()) == _N_OUTPUT_FEATURES

    def test_no_duplicates(self) -> None:
        names = get_feature_names()
        assert len(names) == len(set(names))

    def test_covers_all_feature_cols(self) -> None:
        names = set(get_feature_names())
        for feat in FEATURE_COLS:
            assert feat in names

    def test_covers_all_interaction_feature_names(self) -> None:
        names = set(get_feature_names())
        for feat in INTERACTION_FEATURE_NAMES:
            assert feat in names

    def test_matches_ordered_feature_names_module_constant(self) -> None:
        assert get_feature_names() == list(_ORDERED_FEATURE_NAMES)

    def test_returns_independent_copy(self) -> None:
        names1 = get_feature_names()
        names2 = get_feature_names()
        names1.append("__sentinel__")
        assert "__sentinel__" not in names2

    def test_stateless_no_pipeline_required(self) -> None:
        # Must work without constructing or fitting any pipeline
        names = get_feature_names()
        assert len(names) == _N_OUTPUT_FEATURES

    # ---- Position ranges ---------------------------------------------------

    def test_log_cols_occupy_positions_0_to_6(self) -> None:
        assert get_feature_names()[:7] == _LOG_COLS

    def test_scale_cols_occupy_positions_7_to_23(self) -> None:
        assert get_feature_names()[7:24] == _SCALE_COLS

    def test_binary_cols_occupy_positions_24_to_31(self) -> None:
        assert get_feature_names()[24:32] == _BINARY_COLS

    def test_ordinal_cols_occupy_positions_32_to_35(self) -> None:
        assert get_feature_names()[32:36] == _ORDINAL_COLS

    def test_passthrough_counts_occupy_positions_36_to_38(self) -> None:
        assert get_feature_names()[36:39] == _PASSTHROUGH_COLS

    def test_zero_var_cols_occupy_positions_39_to_40(self) -> None:
        assert get_feature_names()[39:41] == _ZERO_VAR_COLS


# ===========================================================================
# TestValidateFeatureCoverage
# ===========================================================================


class TestValidateFeatureCoverage:
    """_validate_feature_coverage() checks all 3 coverage conditions."""

    def test_happy_path_does_not_raise(self) -> None:
        _validate_feature_coverage()

    # ---- Condition A: completeness -----------------------------------------

    def test_condition_a_missing_feature_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        truncated = list(_ORDERED_FEATURE_NAMES[:-1])
        monkeypatch.setattr(pipeline_module, "_ORDERED_FEATURE_NAMES", truncated)
        with pytest.raises(ValueError, match="[Cc]ondition A|completeness|not assigned"):
            pipeline_module._validate_feature_coverage()

    def test_condition_a_error_names_missing_feature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = _ORDERED_FEATURE_NAMES[-1]
        monkeypatch.setattr(
            pipeline_module, "_ORDERED_FEATURE_NAMES", list(_ORDERED_FEATURE_NAMES[:-1])
        )
        with pytest.raises(ValueError) as exc_info:
            pipeline_module._validate_feature_coverage()
        assert missing in str(exc_info.value)

    # ---- Extra feature guard -----------------------------------------------

    def test_phantom_feature_in_branch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with_extra = list(_ORDERED_FEATURE_NAMES) + ["__phantom__"]
        monkeypatch.setattr(pipeline_module, "_ORDERED_FEATURE_NAMES", with_extra)
        with pytest.raises(ValueError, match="extra|__phantom__|absent"):
            pipeline_module._validate_feature_coverage()

    def test_phantom_feature_error_names_the_phantom(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with_extra = list(_ORDERED_FEATURE_NAMES) + ["__phantom__"]
        monkeypatch.setattr(pipeline_module, "_ORDERED_FEATURE_NAMES", with_extra)
        with pytest.raises(ValueError) as exc_info:
            pipeline_module._validate_feature_coverage()
        assert "__phantom__" in str(exc_info.value)

    # ---- Condition B: exclusivity ------------------------------------------

    def test_condition_b_duplicate_feature_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dup = _ORDERED_FEATURE_NAMES[0]
        with_dup = list(_ORDERED_FEATURE_NAMES) + [dup]
        monkeypatch.setattr(pipeline_module, "_ORDERED_FEATURE_NAMES", with_dup)
        with pytest.raises(ValueError, match="[Cc]ondition B|exclusivity|more than one"):
            pipeline_module._validate_feature_coverage()

    def test_condition_b_error_names_the_duplicate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dup = _ORDERED_FEATURE_NAMES[0]
        with_dup = list(_ORDERED_FEATURE_NAMES) + [dup]
        monkeypatch.setattr(pipeline_module, "_ORDERED_FEATURE_NAMES", with_dup)
        with pytest.raises(ValueError) as exc_info:
            pipeline_module._validate_feature_coverage()
        assert dup in str(exc_info.value)

    # ---- Condition C: total count = 41 -------------------------------------

    def test_condition_c_wrong_total_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build a valid 40-feature set: first 36 of FEATURE_COLS + 4 INTERACTION_FEATURE_NAMES
        short_feature_cols = list(FEATURE_COLS[:36])
        short_ordered = short_feature_cols + list(INTERACTION_FEATURE_NAMES)  # 40, no dups
        monkeypatch.setattr(pipeline_module, "FEATURE_COLS", short_feature_cols)
        monkeypatch.setattr(pipeline_module, "_ORDERED_FEATURE_NAMES", short_ordered)
        with pytest.raises(ValueError, match="[Cc]ondition C|41"):
            pipeline_module._validate_feature_coverage()

    # ---- Integration: build_pipeline() calls validation --------------------

    def test_build_pipeline_propagates_coverage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        truncated = list(_ORDERED_FEATURE_NAMES[:-1])
        monkeypatch.setattr(pipeline_module, "_ORDERED_FEATURE_NAMES", truncated)
        with pytest.raises(ValueError):
            pipeline_module.build_pipeline()


# ===========================================================================
# TestPipelineFit
# ===========================================================================


class TestPipelineFit:

    def test_fit_returns_self(self, valid_df: pd.DataFrame) -> None:
        pipe = build_pipeline()
        assert pipe.fit(valid_df) is pipe

    def test_fit_marks_column_selector_fitted(
        self, fitted_pipeline: Pipeline
    ) -> None:
        assert fitted_pipeline.named_steps["column_selector"].is_fitted_ is True

    def test_fit_marks_interaction_adder_fitted(
        self, fitted_pipeline: Pipeline
    ) -> None:
        assert fitted_pipeline.named_steps["interaction_adder"].is_fitted_ is True

    def test_fit_with_y_argument_accepted(self, valid_df: pd.DataFrame) -> None:
        pipe = build_pipeline()
        pipe.fit(valid_df, pd.Series([0] * len(valid_df)))
        assert pipe.named_steps["column_selector"].is_fitted_

    def test_fit_raises_type_error_on_ndarray(self) -> None:
        pipe = build_pipeline()
        with pytest.raises(TypeError):
            pipe.fit(np.ones((10, _N_BASE_FEATURES)))

    def test_fit_raises_type_error_on_list(self) -> None:
        pipe = build_pipeline()
        with pytest.raises(TypeError):
            pipe.fit([[1, 2, 3]])

    def test_fit_raises_on_missing_feature_col(self) -> None:
        pipe = build_pipeline()
        df = _make_valid_df().drop(columns=["release_lag_hours"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            pipe.fit(df)

    def test_fit_raises_on_missing_col_names_it(self) -> None:
        pipe = build_pipeline()
        df = _make_valid_df().drop(columns=["machine_oee_30d"])
        with pytest.raises(ValueError, match="machine_oee_30d"):
            pipe.fit(df)

    def test_fit_raises_on_target_column_present(self, valid_df: pd.DataFrame) -> None:
        pipe = build_pipeline()
        df = valid_df.copy()
        df["is_delayed"] = 0
        with pytest.raises(ValueError, match="[Tt]arget"):
            pipe.fit(df)

    def test_fit_with_nan_in_rolling_feature_does_not_raise(self) -> None:
        pipe = build_pipeline()
        df = _make_valid_df()
        df["product_delay_rate_90d"] = np.nan
        pipe.fit(df)  # cold-start NaN is expected during fit

    def test_fit_with_extra_columns_silently_ignored(
        self, valid_df: pd.DataFrame
    ) -> None:
        pipe = build_pipeline()
        df = valid_df.copy()
        df["__extra__"] = 999.0
        pipe.fit(df)
        assert pipe.named_steps["column_selector"].is_fitted_


# ===========================================================================
# TestPipelineTransform
# ===========================================================================


class TestPipelineTransform:

    def test_transform_returns_dataframe(
        self, transformed_output: pd.DataFrame
    ) -> None:
        assert isinstance(transformed_output, pd.DataFrame)

    def test_transform_has_41_columns(
        self, transformed_output: pd.DataFrame
    ) -> None:
        assert transformed_output.shape[1] == _N_OUTPUT_FEATURES

    def test_transform_preserves_row_count(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        assert fitted_pipeline.transform(valid_df).shape[0] == len(valid_df)

    def test_transform_single_row(self, fitted_pipeline: Pipeline) -> None:
        out = fitted_pipeline.transform(_make_valid_df(n_rows=1))
        assert out.shape == (1, _N_OUTPUT_FEATURES)

    def test_transform_column_names_match_get_feature_names(
        self, transformed_output: pd.DataFrame
    ) -> None:
        assert list(transformed_output.columns) == get_feature_names()

    def test_transform_no_nan(self, transformed_output: pd.DataFrame) -> None:
        assert transformed_output.isna().sum().sum() == 0

    def test_transform_all_finite(self, transformed_output: pd.DataFrame) -> None:
        assert np.isfinite(transformed_output.values.astype(float)).all()

    def test_transform_before_fit_raises(self, valid_df: pd.DataFrame) -> None:
        pipe = build_pipeline()
        with pytest.raises(Exception):
            pipe.transform(valid_df)

    def test_transform_raises_type_error_on_ndarray(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        with pytest.raises(TypeError):
            fitted_pipeline.transform(valid_df.values)

    def test_transform_raises_on_missing_feature_col(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.drop(columns=["planned_lead_time_hours"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            fitted_pipeline.transform(df)

    def test_transform_drops_extra_columns(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        df = valid_df.copy()
        df["__extra__"] = 999.0
        out = fitted_pipeline.transform(df)
        assert "__extra__" not in out.columns
        assert out.shape[1] == _N_OUTPUT_FEATURES

    def test_transform_passthrough_values_unchanged(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        """Binary / ordinal / count / zero-var base columns must not be scaled.

        log_experience_x_concurrent is an interaction feature produced by
        InteractionFeatureAdder and is absent from valid_df; it is skipped here
        and tested separately via test_transform_zero_var_interaction_is_zero.
        """
        out = fitted_pipeline.transform(valid_df)
        passthrough = _BINARY_COLS + _ORDINAL_COLS + _PASSTHROUGH_COLS + _ZERO_VAR_COLS
        for col in passthrough:
            if col not in valid_df.columns:
                continue  # interaction-derived column; not present in raw input
            np.testing.assert_array_almost_equal(
                out[col].values,
                valid_df[col].values,
                err_msg=f"Passthrough column '{col}' was modified by transform",
            )

    def test_transform_zero_var_interaction_is_zero(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        """log_experience_x_concurrent is always 0 (concurrent_count = 0 in sim)."""
        out = fitted_pipeline.transform(valid_df)
        assert (out["log_experience_x_concurrent"] == 0.0).all()

    def test_transform_log_cols_differ_from_raw_values(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        out = fitted_pipeline.transform(valid_df)
        for col in _LOG_COLS:
            raw = valid_df[col].values if col in valid_df.columns else None
            if raw is not None:
                # log1p + StandardScaler must change the values
                assert not np.allclose(out[col].values, raw, rtol=1e-3), (
                    f"Log+scale column '{col}' appears unchanged after transform"
                )

    def test_transform_is_idempotent_on_same_input(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        out1 = fitted_pipeline.transform(valid_df)
        out2 = fitted_pipeline.transform(valid_df)
        pd.testing.assert_frame_equal(
            out1.reset_index(drop=True), out2.reset_index(drop=True)
        )

    def test_transform_does_not_mutate_input(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        snapshot = valid_df.copy()
        fitted_pipeline.transform(valid_df)
        pd.testing.assert_frame_equal(valid_df.reset_index(drop=True), snapshot.reset_index(drop=True))

    def test_transform_cold_start_nan_filled(
        self, fitted_pipeline: Pipeline
    ) -> None:
        df = _make_valid_df()
        df["product_delay_rate_90d"] = np.nan
        out = fitted_pipeline.transform(df)
        assert out["product_delay_rate_90d"].isna().sum() == 0

    def test_transform_nan_in_non_rolling_raises(
        self, fitted_pipeline: Pipeline
    ) -> None:
        df = _make_valid_df()
        df["machine_oee_30d"] = np.nan
        with pytest.raises(ValueError, match="Unexpected NaN"):
            fitted_pipeline.transform(df)

    def test_transform_no_branch_prefix_in_column_names(
        self, transformed_output: pd.DataFrame
    ) -> None:
        """verbose_feature_names_out=False must suppress 'branch__feature' prefixes."""
        for col in transformed_output.columns:
            assert "__" not in col, (
                f"Column '{col}' looks branch-prefixed; "
                "set verbose_feature_names_out=False"
            )


# ===========================================================================
# TestPipelineFitTransform
# ===========================================================================


class TestPipelineFitTransform:
    """fit_transform() must agree exactly with fit-then-transform."""

    def test_fit_transform_returns_dataframe(self, valid_df: pd.DataFrame) -> None:
        assert isinstance(build_pipeline().fit_transform(valid_df), pd.DataFrame)

    def test_fit_transform_has_41_columns(self, valid_df: pd.DataFrame) -> None:
        assert build_pipeline().fit_transform(valid_df).shape[1] == _N_OUTPUT_FEATURES

    def test_fit_transform_column_names_match_get_feature_names(
        self, valid_df: pd.DataFrame
    ) -> None:
        out = build_pipeline().fit_transform(valid_df)
        assert list(out.columns) == get_feature_names()

    def test_fit_transform_no_nan(self, valid_df: pd.DataFrame) -> None:
        out = build_pipeline().fit_transform(valid_df)
        assert out.isna().sum().sum() == 0

    def test_fit_transform_matches_fit_then_transform(
        self, valid_df: pd.DataFrame
    ) -> None:
        pipe_a = build_pipeline()
        result_a = pipe_a.fit_transform(valid_df)

        pipe_b = build_pipeline()
        pipe_b.fit(valid_df)
        result_b = pipe_b.transform(valid_df)

        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True),
            result_b.reset_index(drop=True),
        )


# ===========================================================================
# TestNoTargetLeakage
# ===========================================================================


class TestNoTargetLeakage:
    """All TARGET_COLS must be rejected at both fit and transform time."""

    @pytest.mark.parametrize("target_col", list(TARGET_COLS))
    def test_fit_raises_on_each_target_col(
        self, valid_df: pd.DataFrame, target_col: str
    ) -> None:
        pipe = build_pipeline()
        df = valid_df.copy()
        df[target_col] = 0
        with pytest.raises(ValueError):
            pipe.fit(df)

    @pytest.mark.parametrize("target_col", list(TARGET_COLS))
    def test_transform_raises_on_each_target_col(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame, target_col: str
    ) -> None:
        df = valid_df.copy()
        df[target_col] = 0
        with pytest.raises(ValueError):
            fitted_pipeline.transform(df)

    def test_fit_raises_with_all_target_cols_present(
        self, valid_df: pd.DataFrame
    ) -> None:
        pipe = build_pipeline()
        df = valid_df.copy()
        for col in TARGET_COLS:
            df[col] = 0
        with pytest.raises(ValueError, match="[Tt]arget"):
            pipe.fit(df)

    def test_output_contains_no_target_cols(
        self, transformed_output: pd.DataFrame
    ) -> None:
        for col in TARGET_COLS:
            assert col not in transformed_output.columns


# ===========================================================================
# TestFeatureNamesConsistency
# ===========================================================================


class TestFeatureNamesConsistency:

    def test_output_columns_equal_get_feature_names(
        self, transformed_output: pd.DataFrame
    ) -> None:
        assert list(transformed_output.columns) == get_feature_names()

    @pytest.mark.parametrize(
        "feature, expected_pos",
        [
            ("planned_lead_time_hours", 0),          # first LOG col
            ("schedule_tightness_ratio", 7),         # first SCALE col
            ("schedule_revision_count", 24),         # first BINARY col
            ("priority_encoded", 32),                # first ORDINAL col
            ("operation_count", 36),                 # first PASSTHROUGH_COUNTS col
            ("operator_concurrent_order_count", 39), # first ZERO_VAR col
            ("log_experience_x_concurrent", 40),     # last ZERO_VAR col
        ],
    )
    def test_feature_at_expected_position(
        self,
        transformed_output: pd.DataFrame,
        feature: str,
        expected_pos: int,
    ) -> None:
        cols = list(transformed_output.columns)
        assert cols[expected_pos] == feature, (
            f"Expected '{feature}' at position {expected_pos}, "
            f"found '{cols[expected_pos]}'"
        )

    def test_lag_as_pct_of_window_in_log_scale_range(
        self, transformed_output: pd.DataFrame
    ) -> None:
        pos = list(transformed_output.columns).index("lag_as_pct_of_window")
        assert _BRANCH_START["log_scale"] <= pos < _BRANCH_START["scale_only"]

    def test_tightness_x_queue_in_scale_only_range(
        self, transformed_output: pd.DataFrame
    ) -> None:
        pos = list(transformed_output.columns).index("tightness_x_queue")
        assert _BRANCH_START["scale_only"] <= pos < _BRANCH_START["binary"]

    def test_oee_x_maintenance_ratio_in_scale_only_range(
        self, transformed_output: pd.DataFrame
    ) -> None:
        pos = list(transformed_output.columns).index("oee_x_maintenance_ratio")
        assert _BRANCH_START["scale_only"] <= pos < _BRANCH_START["binary"]

    def test_log_experience_x_concurrent_in_zero_var_range(
        self, transformed_output: pd.DataFrame
    ) -> None:
        pos = list(transformed_output.columns).index("log_experience_x_concurrent")
        assert _BRANCH_START["zero_variance"] <= pos < _N_OUTPUT_FEATURES


# ===========================================================================
# TestOutputFeatureCount
# ===========================================================================


class TestOutputFeatureCount:

    def test_standard_batch(self, transformed_output: pd.DataFrame) -> None:
        assert transformed_output.shape[1] == _N_OUTPUT_FEATURES

    def test_single_row(self, fitted_pipeline: Pipeline) -> None:
        assert fitted_pipeline.transform(_make_valid_df(1)).shape[1] == _N_OUTPUT_FEATURES

    def test_large_batch(self, fitted_pipeline: Pipeline) -> None:
        assert fitted_pipeline.transform(_make_valid_df(200)).shape[1] == _N_OUTPUT_FEATURES

    def test_with_extra_input_columns(self, fitted_pipeline: Pipeline) -> None:
        df = _make_valid_df()
        df["extra1"] = 1.0
        df["extra2"] = 2.0
        assert fitted_pipeline.transform(df).shape[1] == _N_OUTPUT_FEATURES

    def test_branch_sizes_sum_to_41(self) -> None:
        assert sum(_EXPECTED_BRANCH_SIZES.values()) == _N_OUTPUT_FEATURES

    @pytest.mark.parametrize(
        "branch_name, expected_size",
        list(_EXPECTED_BRANCH_SIZES.items()),
    )
    def test_individual_branch_size(
        self, branch_name: str, expected_size: int
    ) -> None:
        branch_cols = {
            "log_scale": _LOG_COLS,
            "scale_only": _SCALE_COLS,
            "binary": _BINARY_COLS,
            "ordinal": _ORDINAL_COLS,
            "passthrough_counts": _PASSTHROUGH_COLS,
            "zero_variance": _ZERO_VAR_COLS,
        }
        assert len(branch_cols[branch_name]) == expected_size


# ===========================================================================
# TestTransformerAssignment
# ===========================================================================


class TestTransformerAssignment:
    """Verify every feature is in exactly one branch, and in the correct one."""

    def test_no_feature_in_multiple_branches(self) -> None:
        all_cols = (
            _LOG_COLS + _SCALE_COLS + _BINARY_COLS
            + _ORDINAL_COLS + _PASSTHROUGH_COLS + _ZERO_VAR_COLS
        )
        dups = [k for k, v in Counter(all_cols).items() if v > 1]
        assert dups == [], f"Features assigned to multiple branches: {dups}"

    def test_all_features_assigned(self) -> None:
        assigned = set(
            _LOG_COLS + _SCALE_COLS + _BINARY_COLS
            + _ORDINAL_COLS + _PASSTHROUGH_COLS + _ZERO_VAR_COLS
        )
        expected = set(FEATURE_COLS) | set(INTERACTION_FEATURE_NAMES)
        assert assigned == expected

    def test_operator_concurrent_in_zero_var_not_passthrough_counts(self) -> None:
        assert "operator_concurrent_order_count" in _ZERO_VAR_COLS
        assert "operator_concurrent_order_count" not in _PASSTHROUGH_COLS

    def test_log_experience_x_concurrent_in_zero_var(self) -> None:
        assert "log_experience_x_concurrent" in _ZERO_VAR_COLS

    def test_lag_as_pct_of_window_in_log_branch(self) -> None:
        assert "lag_as_pct_of_window" in _LOG_COLS

    def test_tightness_x_queue_in_scale_branch(self) -> None:
        assert "tightness_x_queue" in _SCALE_COLS

    def test_oee_x_maintenance_ratio_in_scale_branch(self) -> None:
        assert "oee_x_maintenance_ratio" in _SCALE_COLS

    def test_log_branch_disjoint_from_scale_branch(self) -> None:
        assert set(_LOG_COLS).isdisjoint(set(_SCALE_COLS))

    @pytest.mark.parametrize("feat", list(LOG_FEATURES))
    def test_log_feature_in_log_branch(self, feat: str) -> None:
        assert feat in _LOG_COLS

    @pytest.mark.parametrize("feat", list(SCALE_FEATURES))
    def test_scale_feature_in_scale_branch(self, feat: str) -> None:
        assert feat in _SCALE_COLS

    @pytest.mark.parametrize("feat", list(BINARY_FEATURES))
    def test_binary_feature_in_binary_branch(self, feat: str) -> None:
        assert feat in _BINARY_COLS

    @pytest.mark.parametrize("feat", list(ORDINAL_FEATURES))
    def test_ordinal_feature_in_ordinal_branch(self, feat: str) -> None:
        assert feat in _ORDINAL_COLS


# ===========================================================================
# TestSerializationCompatibility
# ===========================================================================


class TestSerializationCompatibility:

    def test_unfitted_pipeline_pickle_roundtrip(self) -> None:
        pipe = build_pipeline()
        loaded = pickle.loads(pickle.dumps(pipe))
        assert isinstance(loaded, Pipeline)

    def test_fitted_pipeline_pickle_roundtrip_type(
        self, fitted_pipeline: Pipeline
    ) -> None:
        loaded = pickle.loads(pickle.dumps(fitted_pipeline))
        assert isinstance(loaded, Pipeline)

    def test_pickle_roundtrip_produces_identical_output(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        loaded = pickle.loads(pickle.dumps(fitted_pipeline))
        out_orig = fitted_pipeline.transform(valid_df)
        out_loaded = loaded.transform(valid_df)
        pd.testing.assert_frame_equal(
            out_orig.reset_index(drop=True),
            out_loaded.reset_index(drop=True),
        )

    def test_pickle_loaded_pipeline_column_names(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        loaded = pickle.loads(pickle.dumps(fitted_pipeline))
        assert list(loaded.transform(valid_df).columns) == get_feature_names()

    def test_pickle_loaded_pipeline_output_no_nan(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        loaded = pickle.loads(pickle.dumps(fitted_pipeline))
        assert loaded.transform(valid_df).isna().sum().sum() == 0

    def test_pickle_loaded_pipeline_output_41_cols(
        self, fitted_pipeline: Pipeline, valid_df: pd.DataFrame
    ) -> None:
        loaded = pickle.loads(pickle.dumps(fitted_pipeline))
        assert loaded.transform(valid_df).shape[1] == _N_OUTPUT_FEATURES
