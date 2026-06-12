"""
ml/tests/test_mlflow_utils.py

Pytest suite for mpc_ml.tracking.mlflow_utils.

Coverage areas
--------------
Public API surface          -- __all__ matches declared functions
get_experiment_name()       -- happy path + unknown task raises
start_run()                 -- context manager creates/ends run; exception re-raised
log_standard_params()       -- F-01: happy path, empty dict, bad types, active-run guard
log_pipeline()              -- F-04: baseline path, is_champion guard, no-run guard
log_model_with_signature()  -- F-07: column-count guard; F-03: artifact_path subdir;
                               target-leakage guard, bad-type guard, no-run guard
log_standard_metrics()      -- happy path, non-finite raises, no-run guard
log_standard_artifacts()    -- missing file raises, partial log (all-None ok)
promote_to_production()     -- no version raises; archive flag forwarded
_to_numpy()                 -- DataFrame, ndarray, bad type
_sample_background()        -- shape clamping, reproducibility
_assert_active_run()        -- raises outside run, silent inside run
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, call, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import mpc_ml.tracking.mlflow_utils as utils_module
from mpc_ml.features.constants import FEATURE_COLS, INTERACTION_FEATURE_NAMES, TARGET_COLS
from mpc_ml.features.pipeline import build_pipeline, get_feature_names
from mpc_ml.tracking.mlflow_utils import (
    _ARTIFACT_COLD_START,
    _ARTIFACT_FEATURE_NAMES,
    _ARTIFACT_PATH_PIPELINE,
    _ARTIFACT_PATH_PREPROCESSOR,
    _ARTIFACT_SHAP_BG,
    _ARTIFACT_SHAP_BG_DIR,
    _assert_active_run,
    _sample_background,
    _to_numpy,
    get_experiment_name,
    log_model_with_signature,
    log_pipeline,
    log_standard_artifacts,
    log_standard_metrics,
    log_standard_params,
    promote_to_production,
    start_run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_BASE_FEATURES: int = len(FEATURE_COLS)          # 37
_N_OUTPUT_FEATURES: int = len(get_feature_names())  # 41

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_row() -> dict:
    from mpc_ml.features.constants import COLUMN_DTYPE_CONTRACT
    return {col: (1 if "int" in COLUMN_DTYPE_CONTRACT[col] else 0.5) for col in FEATURE_COLS}


def _make_raw_df(n_rows: int = 20) -> pd.DataFrame:
    """37-column raw feature DataFrame — no target columns."""
    from mpc_ml.features.constants import COLUMN_DTYPE_CONTRACT
    row = _base_row()
    df = pd.DataFrame([row] * n_rows)
    for col, dtype in COLUMN_DTYPE_CONTRACT.items():
        df[col] = df[col].astype(dtype)
    return df


def _make_transformed_array(n_rows: int = 20) -> np.ndarray:
    """41-column transformed numpy array simulating preprocessor output."""
    return np.random.default_rng(0).random((n_rows, _N_OUTPUT_FEATURES)).astype(np.float64)


def _make_transformed_df(n_rows: int = 20) -> pd.DataFrame:
    """41-column transformed DataFrame (set_output='pandas' style)."""
    arr = _make_transformed_array(n_rows)
    return pd.DataFrame(arr, columns=get_feature_names())


def _make_fitted_full_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build and fit a full Pipeline([preprocessor, model]) on X_train."""
    preprocessor = build_pipeline()
    model = LogisticRegression(max_iter=10, random_state=42)
    full = Pipeline([("preprocessor", preprocessor), ("model", model)])
    y = pd.Series([i % 2 for i in range(len(X_train))])
    full.fit(X_train, y)
    return full


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return _make_raw_df(n_rows=30)


@pytest.fixture(scope="module")
def full_pipeline(raw_df: pd.DataFrame) -> Pipeline:
    return _make_fitted_full_pipeline(raw_df)


@pytest.fixture(scope="module")
def X_transformed_np(full_pipeline: Pipeline, raw_df: pd.DataFrame) -> np.ndarray:
    preprocessor = full_pipeline.named_steps["preprocessor"]
    return preprocessor.transform(raw_df).to_numpy(dtype=np.float64)


@pytest.fixture(scope="module")
def X_transformed_df(full_pipeline: Pipeline, raw_df: pd.DataFrame) -> pd.DataFrame:
    preprocessor = full_pipeline.named_steps["preprocessor"]
    return preprocessor.transform(raw_df)


@pytest.fixture
def mlflow_local_uri(tmp_path: Path) -> Generator[str, None, None]:
    """Configure MLflow to use a temporary local tracking URI for isolation."""
    uri = f"file:///{tmp_path / 'mlruns'}"
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(original)


# ===========================================================================
# TestPublicAPI
# ===========================================================================


class TestPublicAPI:
    """__all__ must contain exactly the declared public functions."""

    _EXPECTED_PUBLIC = {
        "get_experiment_name",
        "start_run",
        "log_pipeline",
        "log_model_with_signature",
        "log_standard_params",
        "log_standard_metrics",
        "log_standard_artifacts",
        "promote_to_production",
    }

    def test_all_contains_log_standard_params(self) -> None:
        assert "log_standard_params" in utils_module.__all__

    def test_all_expected_functions_present(self) -> None:
        assert self._EXPECTED_PUBLIC.issubset(set(utils_module.__all__))

    def test_log_standard_params_is_callable(self) -> None:
        assert callable(log_standard_params)

    def test_log_pipeline_has_is_champion_parameter(self) -> None:
        import inspect
        sig = inspect.signature(log_pipeline)
        assert "is_champion" in sig.parameters

    def test_artifact_shap_bg_dir_constant_defined(self) -> None:
        assert _ARTIFACT_SHAP_BG_DIR == "shap_background"


# ===========================================================================
# TestGetExperimentName
# ===========================================================================


class TestGetExperimentName:

    @pytest.mark.parametrize(
        "task, expected",
        [
            ("is_delayed",       "mpc/delay_prediction"),
            ("delay_minutes",    "mpc/delay_regression"),
            ("delay_category",   "mpc/delay_category"),
            ("delay_root_cause", "mpc/root_cause"),
        ],
    )
    def test_known_task_returns_canonical_path(self, task: str, expected: str) -> None:
        assert get_experiment_name(task) == expected

    def test_unknown_task_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            get_experiment_name("not_a_task")

    def test_unknown_task_error_names_the_task(self) -> None:
        with pytest.raises(ValueError, match="unknown_task"):
            get_experiment_name("unknown_task")

    def test_unknown_task_error_lists_valid_options(self) -> None:
        with pytest.raises(ValueError, match="is_delayed"):
            get_experiment_name("wrong")

    def test_all_four_target_cols_are_valid_tasks(self) -> None:
        for col in TARGET_COLS:
            name = get_experiment_name(col)
            assert isinstance(name, str) and name.startswith("mpc/")


# ===========================================================================
# TestAssertActiveRun
# ===========================================================================


class TestAssertActiveRun:

    def test_raises_outside_run(self) -> None:
        with pytest.raises(RuntimeError, match="No active MLflow run"):
            _assert_active_run()

    def test_silent_inside_run(self, mlflow_local_uri: str) -> None:
        mlflow.set_experiment("test_experiment")
        with mlflow.start_run():
            _assert_active_run()  # must not raise


# ===========================================================================
# TestStartRun
# ===========================================================================


class TestStartRun:

    def test_context_manager_yields_active_run(self, mlflow_local_uri: str) -> None:
        with start_run("test/start_run", "test_run") as run:
            assert mlflow.active_run() is not None
            assert run.info.run_id == mlflow.active_run().info.run_id

    def test_run_ends_after_context_exits(self, mlflow_local_uri: str) -> None:
        with start_run("test/start_run", "end_test"):
            pass
        assert mlflow.active_run() is None

    def test_run_ends_even_on_exception(self, mlflow_local_uri: str) -> None:
        with pytest.raises(ZeroDivisionError):
            with start_run("test/start_run", "exc_test"):
                raise ZeroDivisionError("test")
        assert mlflow.active_run() is None

    def test_exception_is_re_raised(self, mlflow_local_uri: str) -> None:
        with pytest.raises(RuntimeError, match="sentinel"):
            with start_run("test/start_run", "reraise_test"):
                raise RuntimeError("sentinel")

    def test_run_has_provided_name(self, mlflow_local_uri: str) -> None:
        with start_run("test/start_run", "my_run") as run:
            assert run.info.run_name == "my_run"

    def test_tags_are_applied(self, mlflow_local_uri: str) -> None:
        tags = {"phase": "baseline", "model_type": "XGBoost"}
        with start_run("test/start_run", "tagged_run", tags=tags) as run:
            active_tags = mlflow.active_run().data.tags
            assert active_tags.get("phase") == "baseline"
            assert active_tags.get("model_type") == "XGBoost"

    def test_no_tags_argument_accepted(self, mlflow_local_uri: str) -> None:
        with start_run("test/start_run", "no_tags_run") as run:
            assert run is not None  # no exception


# ===========================================================================
# TestLogStandardParams  [F-01]
# ===========================================================================


class TestLogStandardParams:
    """F-01: log_standard_params() must be the single call site for param logging."""

    # ---- Active-run guard --------------------------------------------------

    def test_raises_outside_run(self) -> None:
        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_standard_params({"n_estimators": 300})

    # ---- Empty dict --------------------------------------------------------

    def test_empty_dict_raises_value_error(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "empty_params"):
            with pytest.raises(ValueError, match="empty"):
                log_standard_params({})

    def test_empty_dict_error_is_actionable(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "empty_error"):
            with pytest.raises(ValueError, match="get_params"):
                log_standard_params({})

    # ---- Type validation ---------------------------------------------------

    def test_numpy_array_value_raises_type_error(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "np_array"):
            with pytest.raises(TypeError):
                log_standard_params({"bad_param": np.array([1, 2, 3])})

    def test_dict_value_raises_type_error(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "dict_val"):
            with pytest.raises(TypeError):
                log_standard_params({"nested": {"a": 1}})

    def test_list_value_raises_type_error(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "list_val"):
            with pytest.raises(TypeError):
                log_standard_params({"param": [1, 2]})

    def test_type_error_names_all_bad_keys(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "bad_keys_error"):
            with pytest.raises(TypeError) as exc_info:
                log_standard_params({
                    "good_param": 300,
                    "bad_array": np.array([1]),
                    "bad_dict": {"x": 1},
                })
            msg = str(exc_info.value)
            assert "bad_array" in msg
            assert "bad_dict" in msg

    def test_none_value_raises_type_error(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "none_val"):
            with pytest.raises(TypeError):
                log_standard_params({"param": None})

    # ---- Accepted types ----------------------------------------------------

    @pytest.mark.parametrize(
        "params",
        [
            {"n_estimators": 300},
            {"learning_rate": 0.05},
            {"use_gpu": True},
            {"model_name": "xgboost"},
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
            {"n_estimators": 300, "use_label": True, "tag": "champion", "subsample": 0.8},
        ],
    )
    def test_valid_params_do_not_raise(
        self, mlflow_local_uri: str, params: Dict[str, Any]
    ) -> None:
        with start_run("test/params", "valid_types"):
            log_standard_params(params)  # must not raise

    def test_bool_accepted(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "bool_ok"):
            log_standard_params({"use_gpu": False})  # must not raise

    def test_int_accepted(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "int_ok"):
            log_standard_params({"n": 42})

    def test_float_accepted(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "float_ok"):
            log_standard_params({"lr": 0.01})

    def test_string_accepted(self, mlflow_local_uri: str) -> None:
        with start_run("test/params", "str_ok"):
            log_standard_params({"objective": "binary:logistic"})

    # ---- MLflow call is made -----------------------------------------------

    def test_mlflow_log_params_called(self, mlflow_local_uri: str) -> None:
        params = {"n_estimators": 500, "max_depth": 6}
        with start_run("test/params", "call_check"):
            with patch.object(mlflow, "log_params") as mock_log:
                log_standard_params(params)
                mock_log.assert_called_once_with(params)

    def test_params_recorded_in_mlflow_run(self, mlflow_local_uri: str) -> None:
        params = {"n_estimators": 200, "learning_rate": 0.1}
        with start_run("test/params", "recorded") as run:
            log_standard_params(params)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        recorded = client.get_run(run_id).data.params
        assert recorded.get("n_estimators") == "200"
        assert recorded.get("learning_rate") == "0.1"

    # ---- Mixed good/bad raises without logging anything -------------------

    def test_partial_bad_raises_type_error(self, mlflow_local_uri: str) -> None:
        """A single bad value among many good ones must still raise."""
        with start_run("test/params", "partial_bad"):
            with pytest.raises(TypeError):
                log_standard_params({
                    "n_estimators": 300,
                    "bad": np.zeros(5),
                    "max_depth": 5,
                })


# ===========================================================================
# TestLogPipelineChampionGuard  [F-04]
# ===========================================================================


class TestLogPipelineChampionGuard:
    """F-04: log_pipeline(is_champion=True) must raise RuntimeError."""

    # ---- is_champion guard -------------------------------------------------

    def test_is_champion_true_raises_runtime_error(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        with start_run("test/pipeline", "champion_guard"):
            with pytest.raises(RuntimeError):
                log_pipeline(full_pipeline, is_champion=True)

    def test_is_champion_true_error_mentions_log_model_with_signature(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        with start_run("test/pipeline", "champion_msg"):
            with pytest.raises(RuntimeError, match="log_model_with_signature"):
                log_pipeline(full_pipeline, is_champion=True)

    def test_is_champion_true_error_mentions_shap_background(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        with start_run("test/pipeline", "shap_msg"):
            with pytest.raises(RuntimeError, match="shap_background"):
                log_pipeline(full_pipeline, is_champion=True)

    def test_is_champion_true_error_raised_before_mlflow_calls(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        """No MLflow API calls must be made when is_champion=True."""
        with start_run("test/pipeline", "no_mlflow_call"):
            with patch.object(mlflow.sklearn, "log_model") as mock_log:
                with pytest.raises(RuntimeError):
                    log_pipeline(full_pipeline, is_champion=True)
                mock_log.assert_not_called()

    # ---- Baseline path (is_champion=False default) -------------------------

    def test_baseline_does_not_raise(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        with start_run("test/pipeline", "baseline_ok"):
            log_pipeline(full_pipeline)  # is_champion=False by default — must not raise

    def test_explicit_false_does_not_raise(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        with start_run("test/pipeline", "explicit_false"):
            log_pipeline(full_pipeline, is_champion=False)  # must not raise

    def test_baseline_logs_pipeline_artifact(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        with start_run("test/pipeline", "baseline_logs") as run:
            log_pipeline(full_pipeline)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_PATH_PIPELINE in artifacts

    def test_baseline_logs_preprocessor_artifact(
        self, mlflow_local_uri: str, full_pipeline: Pipeline
    ) -> None:
        with start_run("test/pipeline", "baseline_preprocessor") as run:
            log_pipeline(full_pipeline)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_PATH_PREPROCESSOR in artifacts

    # ---- Active-run guard --------------------------------------------------

    def test_raises_outside_run(self, full_pipeline: Pipeline) -> None:
        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_pipeline(full_pipeline)

    # ---- No preprocessor step raises KeyError ------------------------------

    def test_missing_preprocessor_step_raises_key_error(
        self, mlflow_local_uri: str
    ) -> None:
        bad_pipeline = Pipeline([("model", LogisticRegression(max_iter=5))])
        bad_pipeline.fit(np.ones((5, 5)), [0, 0, 1, 1, 0])
        with start_run("test/pipeline", "no_preprocessor"):
            with pytest.raises(KeyError, match="preprocessor"):
                log_pipeline(bad_pipeline)


# ===========================================================================
# TestLogModelWithSignatureF03F07  [F-03, F-07]
# ===========================================================================


class TestLogModelWithSignatureF03F07:

    # ---- F-07: column-count validation ------------------------------------

    def test_raw_df_instead_of_transformed_raises_value_error(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
    ) -> None:
        """Passing X_train_raw (37 cols) as X_train_transformed must raise."""
        with start_run("test/signature", "raw_instead_of_transformed"):
            with pytest.raises(ValueError, match="41"):
                log_model_with_signature(
                    full_pipeline,
                    raw_df,
                    raw_df,  # <-- wrong: should be 41-col transformed output
                )

    def test_wrong_column_count_error_mentions_actual_count(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
    ) -> None:
        wrong_arr = np.ones((20, 37))  # 37, not 41
        with start_run("test/signature", "wrong_count_msg"):
            with pytest.raises(ValueError, match="37"):
                log_model_with_signature(full_pipeline, raw_df, wrong_arr)

    def test_wrong_column_count_error_mentions_expected_count(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
    ) -> None:
        wrong_arr = np.ones((20, 20))
        with start_run("test/signature", "expected_count_msg"):
            with pytest.raises(ValueError, match=str(_N_OUTPUT_FEATURES)):
                log_model_with_signature(full_pipeline, raw_df, wrong_arr)

    def test_wrong_column_count_error_actionable_message(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
    ) -> None:
        wrong_arr = np.ones((20, 10))
        with start_run("test/signature", "actionable_msg"):
            with pytest.raises(ValueError, match="preprocessor"):
                log_model_with_signature(full_pipeline, raw_df, wrong_arr)

    def test_correct_41col_array_does_not_raise(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        """41-col numpy array must pass validation without raising."""
        with start_run("test/signature", "correct_array"):
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)

    def test_correct_41col_dataframe_does_not_raise(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_df: pd.DataFrame,
    ) -> None:
        """41-col DataFrame must pass validation without raising."""
        with start_run("test/signature", "correct_df"):
            log_model_with_signature(full_pipeline, raw_df, X_transformed_df)

    # ---- F-03: artifact_path subdirectory ---------------------------------

    def test_shap_background_logged_to_subdir(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        """shap_background_sample.npy must be under the 'shap_background/' subdir."""
        with start_run("test/signature", "shap_subdir") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level_paths = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_SHAP_BG_DIR in top_level_paths, (
            f"Expected '{_ARTIFACT_SHAP_BG_DIR}/' directory in top-level artifacts, "
            f"got: {top_level_paths}"
        )

    def test_shap_background_not_in_artifact_root(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        """shap_background_sample.npy must NOT be a top-level artifact."""
        with start_run("test/signature", "no_root_bg") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level_paths = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_SHAP_BG not in top_level_paths, (
            f"'{_ARTIFACT_SHAP_BG}' must not appear at artifact root; "
            f"top-level artifacts: {top_level_paths}"
        )

    def test_shap_background_file_inside_subdir(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        """shap_background/shap_background_sample.npy must exist inside the subdir."""
        with start_run("test/signature", "file_in_subdir") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        subdir_files = [a.path for a in client.list_artifacts(run_id, _ARTIFACT_SHAP_BG_DIR)]
        expected = f"{_ARTIFACT_SHAP_BG_DIR}/{_ARTIFACT_SHAP_BG}"
        assert expected in subdir_files, (
            f"Expected '{expected}' inside subdir, got: {subdir_files}"
        )

    def test_shap_background_is_loadable_npy(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        """The logged .npy file must load as a valid float64 array with 41 cols."""
        with start_run("test/signature", "loadable_npy") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        artifact_uri = mlflow.get_run(run_id).info.artifact_uri
        # artifact_uri is a file:// URI; convert to local path
        bg_path = Path(artifact_uri.replace("file:///", "").replace("file://", "")) / _ARTIFACT_SHAP_BG_DIR / _ARTIFACT_SHAP_BG
        # Normalize the path
        bg_path = Path(str(bg_path).lstrip("/"))
        if not bg_path.exists():
            # Try stripping the drive letter prefix quirk on Windows
            bg_path = Path(artifact_uri.split("///")[-1]) / _ARTIFACT_SHAP_BG_DIR / _ARTIFACT_SHAP_BG

        loaded = np.load(str(bg_path))
        assert loaded.ndim == 2
        assert loaded.shape[1] == _N_OUTPUT_FEATURES

    def test_shap_background_sample_shape_matches_n_background_samples(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        """Logged background sample must have the requested number of rows."""
        n = 10
        with start_run("test/signature", "bg_shape") as run:
            log_model_with_signature(
                full_pipeline, raw_df, X_transformed_np, n_background_samples=n
            )
            run_id = run.info.run_id

        artifact_uri = mlflow.get_run(run_id).info.artifact_uri
        bg_path = Path(artifact_uri.split("///")[-1]) / _ARTIFACT_SHAP_BG_DIR / _ARTIFACT_SHAP_BG
        loaded = np.load(str(bg_path))
        assert loaded.shape[0] == min(n, len(X_transformed_np))

    # ---- Complete artifact set is logged ----------------------------------

    def test_feature_names_json_logged(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        with start_run("test/signature", "feat_names") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_FEATURE_NAMES in top_level

    def test_cold_start_defaults_json_logged(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        with start_run("test/signature", "cold_start") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_COLD_START in top_level

    def test_pipeline_artifact_logged(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        with start_run("test/signature", "pipe_art") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_PATH_PIPELINE in top_level

    def test_preprocessor_artifact_logged(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        with start_run("test/signature", "preproc_art") as run:
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level = [a.path for a in client.list_artifacts(run_id)]
        assert _ARTIFACT_PATH_PREPROCESSOR in top_level

    # ---- Target leakage guard ---------------------------------------------

    def test_target_col_in_raw_df_raises_value_error(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        df_with_target = raw_df.copy()
        df_with_target["is_delayed"] = 0
        with start_run("test/signature", "target_leak"):
            with pytest.raises(ValueError, match="is_delayed"):
                log_model_with_signature(full_pipeline, df_with_target, X_transformed_np)

    # ---- TypeError for bad X_train_transformed ----------------------------

    def test_non_array_transformed_raises_type_error(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
    ) -> None:
        with start_run("test/signature", "bad_type"):
            with pytest.raises(TypeError):
                log_model_with_signature(full_pipeline, raw_df, [[1, 2, 3]])

    # ---- Active-run guard -------------------------------------------------

    def test_raises_outside_run(
        self, full_pipeline: Pipeline, raw_df: pd.DataFrame, X_transformed_np: np.ndarray
    ) -> None:
        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)


# ===========================================================================
# TestLogStandardMetrics
# ===========================================================================


class TestLogStandardMetrics:

    def test_raises_outside_run(self) -> None:
        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_standard_metrics({"val_roc_auc": 0.85})

    def test_nan_value_raises_value_error(self, mlflow_local_uri: str) -> None:
        with start_run("test/metrics", "nan_metric"):
            with pytest.raises(ValueError, match="val_roc_auc"):
                log_standard_metrics({"val_roc_auc": float("nan")})

    def test_inf_value_raises_value_error(self, mlflow_local_uri: str) -> None:
        with start_run("test/metrics", "inf_metric"):
            with pytest.raises(ValueError):
                log_standard_metrics({"val_auc": float("inf")})

    def test_valid_metrics_do_not_raise(self, mlflow_local_uri: str) -> None:
        with start_run("test/metrics", "valid"):
            log_standard_metrics({"val_roc_auc": 0.84, "val_f1": 0.71})

    def test_metrics_recorded_in_run(self, mlflow_local_uri: str) -> None:
        with start_run("test/metrics", "recorded") as run:
            log_standard_metrics({"val_roc_auc": 0.84})
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        recorded = client.get_run(run_id).data.metrics
        assert abs(recorded["val_roc_auc"] - 0.84) < 1e-6

    def test_step_argument_forwarded(self, mlflow_local_uri: str) -> None:
        with start_run("test/metrics", "step_test"):
            with patch.object(mlflow, "log_metrics") as mock_log:
                log_standard_metrics({"cv_auc": 0.82}, step=3)
                mock_log.assert_called_once_with({"cv_auc": 0.82}, step=3)


# ===========================================================================
# TestLogStandardArtifacts
# ===========================================================================


class TestLogStandardArtifacts:

    def test_raises_outside_run(self) -> None:
        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_standard_artifacts()

    def test_all_none_does_not_raise(self, mlflow_local_uri: str) -> None:
        with start_run("test/artifacts", "all_none"):
            log_standard_artifacts()  # must not raise

    def test_missing_file_path_raises_file_not_found(
        self, mlflow_local_uri: str
    ) -> None:
        with start_run("test/artifacts", "missing_file"):
            with pytest.raises(FileNotFoundError):
                log_standard_artifacts(
                    confusion_matrix_path=Path("/nonexistent/cm.png")
                )

    def test_classification_report_string_logged(
        self, mlflow_local_uri: str
    ) -> None:
        with start_run("test/artifacts", "cls_report") as run:
            log_standard_artifacts(classification_report="precision: 0.80\nrecall: 0.75\n")
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level = [a.path for a in client.list_artifacts(run_id)]
        assert "classification_report.txt" in top_level

    def test_valid_png_file_logged(
        self, mlflow_local_uri: str, tmp_path: Path
    ) -> None:
        png = tmp_path / "cm.png"
        png.write_bytes(b"\x89PNG\r\n")  # minimal PNG header bytes
        with start_run("test/artifacts", "png_log") as run:
            log_standard_artifacts(confusion_matrix_path=png)
            run_id = run.info.run_id

        client = mlflow.MlflowClient()
        top_level = [a.path for a in client.list_artifacts(run_id)]
        assert "confusion_matrix.png" in top_level


# ===========================================================================
# TestPromoteToProduction
# ===========================================================================


class TestPromoteToProduction:

    def test_no_registered_version_raises_value_error(
        self, mlflow_local_uri: str
    ) -> None:
        with pytest.raises(ValueError, match="No registered version"):
            promote_to_production(
                "nonexistent_model",
                "fake_run_id_that_does_not_exist",
            )

    def test_no_registered_version_error_names_the_model(
        self, mlflow_local_uri: str
    ) -> None:
        with pytest.raises(ValueError, match="nonexistent_model"):
            promote_to_production(
                "nonexistent_model",
                "fake_run_id",
            )

    def test_no_registered_version_error_names_run_id(
        self, mlflow_local_uri: str
    ) -> None:
        fake_run_id = "aabbccdd1234"
        with pytest.raises(ValueError, match=fake_run_id):
            promote_to_production("some_model", fake_run_id)

    def test_archive_previous_flag_forwarded_to_mlflow_client(
        self, mlflow_local_uri: str
    ) -> None:
        """promote_to_production must pass archive_existing_versions=archive_previous."""
        mock_client = MagicMock()
        # Simulate one registered version for the run
        mock_version = MagicMock()
        mock_version.name = "test_model"
        mock_version.version = "1"
        mock_client.search_model_versions.return_value = [mock_version]

        with patch("mpc_ml.tracking.mlflow_utils.mlflow.MlflowClient", return_value=mock_client):
            promote_to_production("test_model", "run123", archive_previous=False)

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test_model",
            version="1",
            stage="Production",
            archive_existing_versions=False,
        )

    def test_archive_previous_defaults_to_true(
        self, mlflow_local_uri: str
    ) -> None:
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.name = "test_model"
        mock_version.version = "2"
        mock_client.search_model_versions.return_value = [mock_version]

        with patch("mpc_ml.tracking.mlflow_utils.mlflow.MlflowClient", return_value=mock_client):
            promote_to_production("test_model", "run_abc")

        call_kwargs = mock_client.transition_model_version_stage.call_args.kwargs
        assert call_kwargs["archive_existing_versions"] is True


# ===========================================================================
# TestToNumpy (private helper)
# ===========================================================================


class TestToNumpy:

    def test_dataframe_converted_to_float64_array(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_dataframe_shape_preserved(self) -> None:
        df = pd.DataFrame(np.ones((5, 3)))
        result = _to_numpy(df)
        assert result.shape == (5, 3)

    def test_float64_ndarray_returned_without_copy(self) -> None:
        arr = np.ones((4, 4), dtype=np.float64)
        result = _to_numpy(arr)
        assert result is arr  # no copy because copy=False when dtype matches

    def test_float32_ndarray_converted_to_float64(self) -> None:
        arr = np.ones((3, 3), dtype=np.float32)
        result = _to_numpy(arr)
        assert result.dtype == np.float64

    def test_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            _to_numpy([[1, 2], [3, 4]])

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            _to_numpy(None)  # type: ignore[arg-type]

    def test_error_message_names_the_type(self) -> None:
        with pytest.raises(TypeError, match="list"):
            _to_numpy([[1, 2]])


# ===========================================================================
# TestSampleBackground (private helper)
# ===========================================================================


class TestSampleBackground:

    def test_returns_ndarray(self) -> None:
        X = np.random.default_rng(0).random((100, _N_OUTPUT_FEATURES))
        result = _sample_background(X, 20, 42)
        assert isinstance(result, np.ndarray)

    def test_correct_number_of_rows_returned(self) -> None:
        X = np.random.default_rng(0).random((100, _N_OUTPUT_FEATURES))
        result = _sample_background(X, 30, 42)
        assert result.shape[0] == 30

    def test_column_count_preserved(self) -> None:
        X = np.random.default_rng(0).random((50, _N_OUTPUT_FEATURES))
        result = _sample_background(X, 20, 42)
        assert result.shape[1] == _N_OUTPUT_FEATURES

    def test_clamped_to_n_train_when_n_samples_larger(self) -> None:
        X = np.random.default_rng(0).random((10, _N_OUTPUT_FEATURES))
        result = _sample_background(X, 200, 42)
        assert result.shape[0] == 10  # clamped to len(X)

    def test_no_duplicate_rows_when_n_less_than_n_train(self) -> None:
        X = np.eye(_N_OUTPUT_FEATURES)  # each row is unique
        result = _sample_background(X, _N_OUTPUT_FEATURES // 2, 42)
        # All rows should be unique
        unique_rows = {tuple(row) for row in result}
        assert len(unique_rows) == result.shape[0]

    def test_reproducible_with_same_seed(self) -> None:
        X = np.random.default_rng(99).random((200, _N_OUTPUT_FEATURES))
        r1 = _sample_background(X, 50, 42)
        r2 = _sample_background(X, 50, 42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_produces_different_sample(self) -> None:
        X = np.random.default_rng(99).random((200, _N_OUTPUT_FEATURES))
        r1 = _sample_background(X, 50, 0)
        r2 = _sample_background(X, 50, 1)
        assert not np.array_equal(r1, r2)


# ===========================================================================
# TestF07ColumnCountValidationIntegration
# ===========================================================================


class TestF07ColumnCountValidationIntegration:
    """End-to-end: passing the wrong array to log_model_with_signature is caught."""

    def test_37_col_array_raises(
        self, mlflow_local_uri: str, full_pipeline: Pipeline, raw_df: pd.DataFrame
    ) -> None:
        wrong = np.ones((20, 37))
        with start_run("test/f07", "37col"):
            with pytest.raises(ValueError):
                log_model_with_signature(full_pipeline, raw_df, wrong)

    def test_40_col_array_raises(
        self, mlflow_local_uri: str, full_pipeline: Pipeline, raw_df: pd.DataFrame
    ) -> None:
        wrong = np.ones((20, 40))
        with start_run("test/f07", "40col"):
            with pytest.raises(ValueError):
                log_model_with_signature(full_pipeline, raw_df, wrong)

    def test_42_col_array_raises(
        self, mlflow_local_uri: str, full_pipeline: Pipeline, raw_df: pd.DataFrame
    ) -> None:
        wrong = np.ones((20, 42))
        with start_run("test/f07", "42col"):
            with pytest.raises(ValueError):
                log_model_with_signature(full_pipeline, raw_df, wrong)

    def test_41_col_array_does_not_raise(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        assert X_transformed_np.shape[1] == 41
        with start_run("test/f07", "41col_ok"):
            log_model_with_signature(full_pipeline, raw_df, X_transformed_np)

    def test_41_col_dataframe_does_not_raise(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_df: pd.DataFrame,
    ) -> None:
        assert X_transformed_df.shape[1] == 41
        with start_run("test/f07", "41col_df_ok"):
            log_model_with_signature(full_pipeline, raw_df, X_transformed_df)


# ===========================================================================
# TestF03ArtifactPathIntegration
# ===========================================================================


class TestF03ArtifactPathIntegration:
    """F-03 regression: shap_background_sample.npy must never land at root."""

    def test_shap_bg_dir_constant_is_shap_background(self) -> None:
        assert _ARTIFACT_SHAP_BG_DIR == "shap_background"

    def test_log_artifact_call_uses_artifact_path(
        self,
        mlflow_local_uri: str,
        full_pipeline: Pipeline,
        raw_df: pd.DataFrame,
        X_transformed_np: np.ndarray,
    ) -> None:
        """Verify mlflow.log_artifact is called with artifact_path=_ARTIFACT_SHAP_BG_DIR."""
        original_log_artifact = mlflow.log_artifact

        captured_calls = []

        def capturing_log_artifact(local_path: str, artifact_path: str = None) -> None:  # type: ignore[assignment]
            captured_calls.append({"local_path": local_path, "artifact_path": artifact_path})
            return original_log_artifact(local_path, artifact_path=artifact_path)

        with start_run("test/f03", "artifact_path_check"):
            with patch.object(mlflow, "log_artifact", side_effect=capturing_log_artifact):
                log_model_with_signature(full_pipeline, raw_df, X_transformed_np)

        shap_calls = [c for c in captured_calls if _ARTIFACT_SHAP_BG in c["local_path"]]
        assert len(shap_calls) == 1, (
            f"Expected exactly 1 log_artifact call for shap background, "
            f"found: {shap_calls}"
        )
        assert shap_calls[0]["artifact_path"] == _ARTIFACT_SHAP_BG_DIR, (
            f"Expected artifact_path='{_ARTIFACT_SHAP_BG_DIR}', "
            f"got: {shap_calls[0]['artifact_path']!r}"
        )
