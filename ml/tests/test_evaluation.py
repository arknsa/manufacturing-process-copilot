"""
ml/tests/test_evaluation.py

Pytest suite for mpc_ml.models.evaluation.

Coverage target : ≥90% of evaluation.py
Test count      : 75
Estimated coverage: ~93% of executable lines

Functions covered
-----------------
evaluate_model()             binary, regression, delay_category, delay_root_cause,
                             error paths (unknown task, target leakage, single class)
precision_at_recall()        boundary targets (0.0, 1.0), perfect/random classifier,
                             invalid target_recall, DataFrame/Series inputs
calibration_report()         output structure, ECE bounds, n_bins guard,
                             perfect-calibration ECE=0
confusion_matrix_annotated() binary + multi-class shape, precision/recall annotation,
                             threshold guard, no-proba path, TypeError from bad input

Private helpers exercised via public API
-----------------------------------------
_coerce_to_float64   DataFrame path, ndarray path, TypeError on list input
_compute_ece         normal bins, empty-bin skip, last-bin boundary (y_prob=1.0)
_validate_task       unknown-task ValueError
_evaluate_binary     10-key output, single-class ValueError
_evaluate_regression 3-key output, exact MAE/R² values
_evaluate_multiclass with predict_proba (roc_auc present), without predict_proba
                     (roc_auc absent), missing-class ValueError swallowed
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from mpc_ml.models.evaluation import (
    CalibrationResult,
    calibration_report,
    confusion_matrix_annotated,
    evaluate_model,
    precision_at_recall,
)

# ===========================================================================
# Module-level constants — all deterministic
# ===========================================================================

_N = 100  # total samples used across all fixtures

# Binary ground truth: 50 negatives (even indices), 50 positives (odd indices)
_Y_BINARY: np.ndarray = np.array([i % 2 for i in range(_N)], dtype=int)

# Ordinal (5-class) ground truth — 20 samples per class
# Must be in sorted (alphabetical) order to satisfy sklearn's requirement that
# model.classes_ is sorted, which roc_auc_score enforces via labels=model.classes_.
_CATEGORIES: tuple = (
    "critical_delay",
    "major_delay",
    "minor_delay",
    "moderate_delay",
    "on_time",
)
_Y_CATEGORY: np.ndarray = np.array([_CATEGORIES[i % 5] for i in range(_N)])

# Multi-class (7-class) ground truth — 14-15 samples per class
_ROOT_CAUSES: tuple = (
    "machine_breakdown",
    "material_unavailability",
    "multiple_causes",
    "none",
    "planning_schedule_conflict",
    "quality_failure_rework",
    "setup_overrun",
)
_Y_ROOT_CAUSE: np.ndarray = np.array([_ROOT_CAUSES[i % 7] for i in range(_N)])

# Continuous regression target
_Y_REGRESSION: np.ndarray = np.linspace(0.0, 100.0, _N)

# Pre-transformed feature matrices (what pipeline.transform() returns)
_X_ARR: np.ndarray = np.ones((_N, 5), dtype=np.float64)
_X_DF_CLEAN: pd.DataFrame = pd.DataFrame(
    _X_ARR, columns=[f"feat_{i}" for i in range(5)]
)

# Binary probability vectors
_PROBS_PERFECT: np.ndarray = np.where(_Y_BINARY == 1, 1.0, 0.0)
_PROBS_NEAR_PERFECT: np.ndarray = np.where(_Y_BINARY == 1, 0.9, 0.1)
_PROBS_RANDOM: np.ndarray = np.full(_N, 0.5)

# Required key sets per task
_BINARY_KEYS = frozenset(
    {
        "val_roc_auc",
        "val_ap",
        "val_pr_auc",
        "val_brier_score",
        "val_f1",
        "val_ece",
        "val_precision_at_80pct_recall",
        "val_f1_at_040",
        "val_precision_at_040",
        "val_recall_at_040",
    }
)
_REGRESSION_KEYS = frozenset({"val_mae", "val_rmse", "val_r2"})
_MULTICLASS_KEYS = frozenset({"val_weighted_f1", "val_macro_f1"})


# ===========================================================================
# Stub classes — deterministic, ignore input features
# ===========================================================================


class _BinaryClf:
    """Binary classifier backed by a pre-built positive-class probability vector."""

    classes_ = np.array([0, 1])

    def __init__(self, probs_positive: np.ndarray) -> None:
        self._p = np.asarray(probs_positive, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._p[: len(X)]
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._p[: len(X)] >= 0.5).astype(int)


class _Regressor:
    """Regressor backed by a pre-built prediction vector."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._pred = np.asarray(predictions, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pred[: len(X)]


class _MultiClf:
    """Multi-class classifier with optional predict_proba."""

    def __init__(
        self,
        classes: tuple,
        predictions: np.ndarray,
        proba: Optional[np.ndarray] = None,
    ) -> None:
        self.classes_ = np.asarray(classes)
        self._pred = np.asarray(predictions)
        self._proba = proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pred[: len(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._proba is None:
            raise AttributeError("predict_proba not available on this stub.")
        return self._proba[: len(X)]


class _NoProbaClf:
    """Classifier without predict_proba — used to test the no-proba branch."""

    def __init__(self, classes: tuple, predictions: np.ndarray) -> None:
        self.classes_ = np.asarray(classes)
        self._pred = np.asarray(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pred[: len(X)]


class _DummyPipeline:
    """Pipeline stub: transform() coerces input to float64 ndarray."""

    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)


# ===========================================================================
# Helpers
# ===========================================================================


def _near_one_hot(y_labels: np.ndarray, classes: tuple) -> np.ndarray:
    """Near-one-hot proba matrix: 0.85 for true class, equal small for others.

    Each row sums to 1.0.  Designed so roc_auc_score can distinguish classes.
    """
    n = len(y_labels)
    n_cls = len(classes)
    fill = (1.0 - 0.85) / max(n_cls - 1, 1)
    cls_idx = {c: i for i, c in enumerate(classes)}
    proba = np.full((n, n_cls), fill)
    for row, label in enumerate(y_labels):
        proba[row, cls_idx[str(label)]] = 0.85
    return proba


# ===========================================================================
# Pytest fixtures
# ===========================================================================


@pytest.fixture
def pipeline() -> _DummyPipeline:
    return _DummyPipeline()


@pytest.fixture
def clf_perfect() -> _BinaryClf:
    return _BinaryClf(_PROBS_PERFECT)


@pytest.fixture
def clf_near_perfect() -> _BinaryClf:
    return _BinaryClf(_PROBS_NEAR_PERFECT)


@pytest.fixture
def clf_random() -> _BinaryClf:
    return _BinaryClf(_PROBS_RANDOM)


@pytest.fixture
def regressor_perfect() -> _Regressor:
    return _Regressor(_Y_REGRESSION)


@pytest.fixture
def regressor_biased() -> _Regressor:
    return _Regressor(_Y_REGRESSION + 10.0)


@pytest.fixture
def clf_category_with_proba() -> _MultiClf:
    proba = _near_one_hot(_Y_CATEGORY, _CATEGORIES)
    return _MultiClf(_CATEGORIES, _Y_CATEGORY, proba=proba)


@pytest.fixture
def clf_category_no_proba() -> _NoProbaClf:
    return _NoProbaClf(_CATEGORIES, _Y_CATEGORY)


@pytest.fixture
def clf_root_cause_with_proba() -> _MultiClf:
    proba = _near_one_hot(_Y_ROOT_CAUSE, _ROOT_CAUSES)
    return _MultiClf(_ROOT_CAUSES, _Y_ROOT_CAUSE, proba=proba)


@pytest.fixture
def clf_root_cause_no_proba() -> _NoProbaClf:
    return _NoProbaClf(_ROOT_CAUSES, _Y_ROOT_CAUSE)


# ===========================================================================
# Tests — evaluate_model() binary classification
# ===========================================================================


class TestEvaluateModelBinary:

    def test_returns_all_ten_required_keys(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert _BINARY_KEYS.issubset(m.keys())

    def test_no_regression_or_multiclass_keys(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert "val_mae" not in m
        assert "val_weighted_f1" not in m

    def test_all_values_are_floats(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        for key, val in m.items():
            assert isinstance(val, float), f"{key} has type {type(val)}, expected float"

    def test_roc_auc_in_unit_interval(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert 0.0 <= m["val_roc_auc"] <= 1.0

    def test_ap_in_unit_interval(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert 0.0 <= m["val_ap"] <= 1.0

    def test_pr_auc_equals_ap(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert m["val_pr_auc"] == m["val_ap"]

    def test_brier_score_non_negative(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert m["val_brier_score"] >= 0.0

    def test_ece_in_unit_interval(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert 0.0 <= m["val_ece"] <= 1.0

    def test_precision_at_80pct_recall_in_unit_interval(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert 0.0 <= m["val_precision_at_80pct_recall"] <= 1.0

    def test_f1_metrics_in_unit_interval(self, clf_near_perfect, pipeline):
        m = evaluate_model(
            clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        for key in ("val_f1", "val_f1_at_040", "val_precision_at_040", "val_recall_at_040"):
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} outside [0,1]"

    def test_perfect_classifier_roc_auc_is_one(self, clf_perfect, pipeline):
        m = evaluate_model(
            clf_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert m["val_roc_auc"] == pytest.approx(1.0)

    def test_perfect_classifier_brier_score_is_zero(self, clf_perfect, pipeline):
        m = evaluate_model(
            clf_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert m["val_brier_score"] == pytest.approx(0.0)

    def test_perfect_classifier_ece_is_zero(self, clf_perfect, pipeline):
        m = evaluate_model(
            clf_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert m["val_ece"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_classifier_precision_at_80pct_recall_is_one(
        self, clf_perfect, pipeline
    ):
        m = evaluate_model(
            clf_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "is_delayed"
        )
        assert m["val_precision_at_80pct_recall"] == pytest.approx(1.0)

    def test_single_class_y_raises_value_error(self, clf_near_perfect, pipeline):
        y_all_neg = np.zeros(_N, dtype=int)
        with pytest.raises(ValueError, match="unique class"):
            evaluate_model(
                clf_near_perfect, pipeline, _X_DF_CLEAN, y_all_neg, "is_delayed"
            )


# ===========================================================================
# Tests — evaluate_model() regression
# ===========================================================================


class TestEvaluateModelRegression:

    def test_returns_all_three_keys(self, regressor_perfect, pipeline):
        m = evaluate_model(
            regressor_perfect, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert _REGRESSION_KEYS.issubset(m.keys())

    def test_no_binary_or_multiclass_keys(self, regressor_perfect, pipeline):
        m = evaluate_model(
            regressor_perfect, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert "val_roc_auc" not in m
        assert "val_weighted_f1" not in m

    def test_all_values_are_floats(self, regressor_biased, pipeline):
        m = evaluate_model(
            regressor_biased, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        for key, val in m.items():
            assert isinstance(val, float), f"{key} has type {type(val)}"

    def test_mae_non_negative(self, regressor_biased, pipeline):
        m = evaluate_model(
            regressor_biased, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert m["val_mae"] >= 0.0

    def test_rmse_non_negative(self, regressor_biased, pipeline):
        m = evaluate_model(
            regressor_biased, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert m["val_rmse"] >= 0.0

    def test_rmse_geq_mae(self, regressor_biased, pipeline):
        m = evaluate_model(
            regressor_biased, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert m["val_rmse"] >= m["val_mae"] - 1e-9

    def test_perfect_predictor_mae_is_zero(self, regressor_perfect, pipeline):
        m = evaluate_model(
            regressor_perfect, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert m["val_mae"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictor_r2_is_one(self, regressor_perfect, pipeline):
        m = evaluate_model(
            regressor_perfect, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert m["val_r2"] == pytest.approx(1.0)

    def test_constant_bias_produces_expected_mae(self, regressor_biased, pipeline):
        m = evaluate_model(
            regressor_biased, pipeline, _X_DF_CLEAN, _Y_REGRESSION, "delay_minutes"
        )
        assert m["val_mae"] == pytest.approx(10.0)


# ===========================================================================
# Tests — evaluate_model() delay_category (ordinal 5-class)
# ===========================================================================


class TestEvaluateModelDelayCategory:

    def test_returns_weighted_and_macro_f1_keys(self, clf_category_with_proba, pipeline):
        m = evaluate_model(
            clf_category_with_proba, pipeline, _X_DF_CLEAN, _Y_CATEGORY, "delay_category"
        )
        assert _MULTICLASS_KEYS.issubset(m.keys())

    def test_all_values_are_floats(self, clf_category_with_proba, pipeline):
        m = evaluate_model(
            clf_category_with_proba, pipeline, _X_DF_CLEAN, _Y_CATEGORY, "delay_category"
        )
        for key, val in m.items():
            assert isinstance(val, float), f"{key} has type {type(val)}"

    def test_f1_values_in_unit_interval(self, clf_category_with_proba, pipeline):
        m = evaluate_model(
            clf_category_with_proba, pipeline, _X_DF_CLEAN, _Y_CATEGORY, "delay_category"
        )
        assert 0.0 <= m["val_weighted_f1"] <= 1.0
        assert 0.0 <= m["val_macro_f1"] <= 1.0

    def test_roc_auc_present_when_predict_proba_available(
        self, clf_category_with_proba, pipeline
    ):
        m = evaluate_model(
            clf_category_with_proba, pipeline, _X_DF_CLEAN, _Y_CATEGORY, "delay_category"
        )
        assert "val_roc_auc" in m
        assert 0.0 <= m["val_roc_auc"] <= 1.0

    def test_roc_auc_absent_without_predict_proba(self, clf_category_no_proba, pipeline):
        m = evaluate_model(
            clf_category_no_proba, pipeline, _X_DF_CLEAN, _Y_CATEGORY, "delay_category"
        )
        assert "val_roc_auc" not in m

    def test_perfect_predictions_yield_f1_one(self, clf_category_with_proba, pipeline):
        m = evaluate_model(
            clf_category_with_proba, pipeline, _X_DF_CLEAN, _Y_CATEGORY, "delay_category"
        )
        assert m["val_weighted_f1"] == pytest.approx(1.0)
        assert m["val_macro_f1"] == pytest.approx(1.0)

    def test_roc_auc_absent_when_class_missing_from_y(self, pipeline):
        # Model trained on 3 classes; y only contains 2 → roc_auc ValueError swallowed
        three_classes = (0, 1, 2)
        y_two_class = np.array([i % 2 for i in range(_N)], dtype=int)
        preds = y_two_class.copy()
        proba = np.column_stack(
            [1.0 - y_two_class.astype(float), y_two_class.astype(float), np.zeros(_N)]
        )
        clf = _MultiClf(three_classes, preds, proba=proba)
        m = evaluate_model(clf, pipeline, _X_DF_CLEAN, y_two_class, "delay_category")
        assert "val_roc_auc" not in m
        assert "val_weighted_f1" in m


# ===========================================================================
# Tests — evaluate_model() delay_root_cause (multi-class 7-class)
# ===========================================================================


class TestEvaluateModelRootCause:

    def test_returns_required_multiclass_keys(self, clf_root_cause_with_proba, pipeline):
        m = evaluate_model(
            clf_root_cause_with_proba,
            pipeline,
            _X_DF_CLEAN,
            _Y_ROOT_CAUSE,
            "delay_root_cause",
        )
        assert _MULTICLASS_KEYS.issubset(m.keys())

    def test_all_values_are_floats(self, clf_root_cause_with_proba, pipeline):
        m = evaluate_model(
            clf_root_cause_with_proba,
            pipeline,
            _X_DF_CLEAN,
            _Y_ROOT_CAUSE,
            "delay_root_cause",
        )
        for key, val in m.items():
            assert isinstance(val, float)

    def test_roc_auc_present_with_predict_proba(self, clf_root_cause_with_proba, pipeline):
        m = evaluate_model(
            clf_root_cause_with_proba,
            pipeline,
            _X_DF_CLEAN,
            _Y_ROOT_CAUSE,
            "delay_root_cause",
        )
        assert "val_roc_auc" in m

    def test_roc_auc_absent_without_predict_proba(self, clf_root_cause_no_proba, pipeline):
        m = evaluate_model(
            clf_root_cause_no_proba,
            pipeline,
            _X_DF_CLEAN,
            _Y_ROOT_CAUSE,
            "delay_root_cause",
        )
        assert "val_roc_auc" not in m

    def test_perfect_predictions_macro_f1_is_one(self, clf_root_cause_with_proba, pipeline):
        m = evaluate_model(
            clf_root_cause_with_proba,
            pipeline,
            _X_DF_CLEAN,
            _Y_ROOT_CAUSE,
            "delay_root_cause",
        )
        assert m["val_macro_f1"] == pytest.approx(1.0)


# ===========================================================================
# Tests — evaluate_model() error paths
# ===========================================================================


class TestEvaluateModelErrors:

    def test_unknown_task_raises_value_error(self, clf_near_perfect, pipeline):
        with pytest.raises(ValueError, match="Unknown task"):
            evaluate_model(
                clf_near_perfect, pipeline, _X_DF_CLEAN, _Y_BINARY, "not_a_task"
            )

    def test_single_target_column_in_x_raises(self, clf_near_perfect, pipeline):
        X_leaky = _X_DF_CLEAN.copy()
        X_leaky["is_delayed"] = 0
        with pytest.raises(ValueError, match="target column"):
            evaluate_model(clf_near_perfect, pipeline, X_leaky, _Y_BINARY, "is_delayed")

    def test_multiple_target_columns_in_x_raises(self, clf_near_perfect, pipeline):
        X_leaky = _X_DF_CLEAN.copy()
        X_leaky["is_delayed"] = 0
        X_leaky["delay_minutes"] = 0.0
        with pytest.raises(ValueError, match="target column"):
            evaluate_model(clf_near_perfect, pipeline, X_leaky, _Y_BINARY, "is_delayed")


# ===========================================================================
# Tests — precision_at_recall()
# ===========================================================================


class TestPrecisionAtRecall:

    def test_returns_float(self, clf_near_perfect):
        result = precision_at_recall(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert isinstance(result, float)

    def test_perfect_classifier_at_default_target_returns_one(self, clf_perfect):
        result = precision_at_recall(clf_perfect, _X_ARR, _Y_BINARY)
        assert result == pytest.approx(1.0)

    def test_target_recall_zero_returns_positive(self, clf_near_perfect):
        # recall >= 0.0 is always achievable; max precision >= class prior
        result = precision_at_recall(clf_near_perfect, _X_ARR, _Y_BINARY, target_recall=0.0)
        assert result > 0.0

    def test_target_recall_one_perfect_clf_returns_one(self, clf_perfect):
        result = precision_at_recall(clf_perfect, _X_ARR, _Y_BINARY, target_recall=1.0)
        assert result == pytest.approx(1.0)

    def test_near_perfect_clf_at_80pct_recall_in_unit_interval(self, clf_near_perfect):
        result = precision_at_recall(
            clf_near_perfect, _X_ARR, _Y_BINARY, target_recall=0.80
        )
        assert 0.0 <= result <= 1.0

    def test_random_clf_at_50pct_recall_in_unit_interval(self, clf_random):
        result = precision_at_recall(clf_random, _X_ARR, _Y_BINARY, target_recall=0.5)
        assert 0.0 <= result <= 1.0

    def test_invalid_negative_target_recall_raises(self, clf_near_perfect):
        with pytest.raises(ValueError, match="target_recall"):
            precision_at_recall(clf_near_perfect, _X_ARR, _Y_BINARY, target_recall=-0.01)

    def test_invalid_above_one_target_recall_raises(self, clf_near_perfect):
        with pytest.raises(ValueError, match="target_recall"):
            precision_at_recall(clf_near_perfect, _X_ARR, _Y_BINARY, target_recall=1.01)

    def test_accepts_dataframe_x_input(self, clf_near_perfect):
        result = precision_at_recall(clf_near_perfect, _X_DF_CLEAN, _Y_BINARY)
        assert isinstance(result, float)

    def test_accepts_series_y_input(self, clf_near_perfect):
        result = precision_at_recall(
            clf_near_perfect, _X_ARR, pd.Series(_Y_BINARY)
        )
        assert isinstance(result, float)


# ===========================================================================
# Tests — calibration_report()
# ===========================================================================


class TestCalibrationReport:

    def test_returns_calibration_result_instance(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert isinstance(result, CalibrationResult)

    def test_ece_is_float(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert isinstance(result.ece, float)

    def test_ece_non_negative(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert result.ece >= 0.0

    def test_ece_at_most_one(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert result.ece <= 1.0

    def test_n_bins_stored_on_result(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY, n_bins=8)
        assert result.n_bins == 8

    def test_fraction_of_positives_in_unit_interval(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert np.all(result.fraction_of_positives >= 0.0)
        assert np.all(result.fraction_of_positives <= 1.0)

    def test_mean_predicted_value_in_unit_interval(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert np.all(result.mean_predicted_value >= 0.0)
        assert np.all(result.mean_predicted_value <= 1.0)

    def test_output_arrays_are_1d_numpy_arrays(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert isinstance(result.fraction_of_positives, np.ndarray)
        assert isinstance(result.mean_predicted_value, np.ndarray)
        assert result.fraction_of_positives.ndim == 1
        assert result.mean_predicted_value.ndim == 1

    def test_n_bins_less_than_two_raises(self, clf_near_perfect):
        with pytest.raises(ValueError, match="n_bins"):
            calibration_report(clf_near_perfect, _X_ARR, _Y_BINARY, n_bins=1)

    def test_perfect_calibration_ece_is_zero(self, clf_perfect):
        # probs are exactly 0.0 or 1.0 → each bin acc == conf → ECE = 0
        result = calibration_report(clf_perfect, _X_ARR, _Y_BINARY)
        assert result.ece == pytest.approx(0.0, abs=1e-9)

    def test_accepts_dataframe_x_input(self, clf_near_perfect):
        result = calibration_report(clf_near_perfect, _X_DF_CLEAN, _Y_BINARY)
        assert isinstance(result, CalibrationResult)
        assert result.ece >= 0.0


# ===========================================================================
# Tests — confusion_matrix_annotated()
# ===========================================================================


class TestConfusionMatrixAnnotated:

    def test_binary_returns_dataframe(self, clf_perfect):
        df = confusion_matrix_annotated(clf_perfect, _X_ARR, _Y_BINARY)
        assert isinstance(df, pd.DataFrame)

    def test_binary_shape_is_two_rows_four_cols(self, clf_perfect):
        # 2 classes → 2×2 CM + precision + recall → shape (2, 4)
        df = confusion_matrix_annotated(clf_perfect, _X_ARR, _Y_BINARY)
        assert df.shape == (2, 4)

    def test_binary_index_name_is_true_label(self, clf_perfect):
        df = confusion_matrix_annotated(clf_perfect, _X_ARR, _Y_BINARY)
        assert df.index.name == "true_label"

    def test_binary_precision_and_recall_columns_present(self, clf_perfect):
        df = confusion_matrix_annotated(clf_perfect, _X_ARR, _Y_BINARY)
        assert "precision" in df.columns
        assert "recall" in df.columns

    def test_binary_precision_recall_in_unit_interval(self, clf_near_perfect):
        df = confusion_matrix_annotated(clf_near_perfect, _X_ARR, _Y_BINARY)
        assert (df["precision"] >= 0.0).all()
        assert (df["precision"] <= 1.0).all()
        assert (df["recall"] >= 0.0).all()
        assert (df["recall"] <= 1.0).all()

    def test_binary_perfect_classifier_precision_recall_all_one(self, clf_perfect):
        df = confusion_matrix_annotated(clf_perfect, _X_ARR, _Y_BINARY)
        assert df["precision"].values == pytest.approx([1.0, 1.0])
        assert df["recall"].values == pytest.approx([1.0, 1.0])

    def test_binary_custom_threshold_changes_predictions(self, clf_near_perfect):
        # threshold=0.95 pushes most positives into the negative bucket
        # threshold=0.05 captures almost everything as positive
        df_high = confusion_matrix_annotated(
            clf_near_perfect, _X_ARR, _Y_BINARY, threshold=0.95
        )
        df_low = confusion_matrix_annotated(
            clf_near_perfect, _X_ARR, _Y_BINARY, threshold=0.05
        )
        # CM cells (first two cols) must differ between threshold settings
        assert not df_high.iloc[:, :2].equals(df_low.iloc[:, :2])

    def test_binary_threshold_zero_raises(self, clf_perfect):
        with pytest.raises(ValueError, match="threshold"):
            confusion_matrix_annotated(clf_perfect, _X_ARR, _Y_BINARY, threshold=0.0)

    def test_binary_threshold_one_raises(self, clf_perfect):
        with pytest.raises(ValueError, match="threshold"):
            confusion_matrix_annotated(clf_perfect, _X_ARR, _Y_BINARY, threshold=1.0)

    def test_binary_no_proba_model_uses_predict(self):
        clf_no_proba = _NoProbaClf((0, 1), _Y_BINARY)
        df = confusion_matrix_annotated(clf_no_proba, _X_ARR, _Y_BINARY)
        assert isinstance(df, pd.DataFrame)
        assert "precision" in df.columns
        assert "recall" in df.columns

    def test_multiclass_shape_is_n_classes_plus_two_cols(self):
        clf = _NoProbaClf(_CATEGORIES, _Y_CATEGORY)
        df = confusion_matrix_annotated(clf, _X_ARR, _Y_CATEGORY)
        # 5 classes → 5×5 CM + precision + recall → shape (5, 7)
        assert df.shape == (5, 7)

    def test_multiclass_index_name_is_true_label(self):
        clf = _NoProbaClf(_CATEGORIES, _Y_CATEGORY)
        df = confusion_matrix_annotated(clf, _X_ARR, _Y_CATEGORY)
        assert df.index.name == "true_label"

    def test_multiclass_precision_and_recall_columns_present(self):
        clf = _NoProbaClf(_CATEGORIES, _Y_CATEGORY)
        df = confusion_matrix_annotated(clf, _X_ARR, _Y_CATEGORY)
        assert "precision" in df.columns
        assert "recall" in df.columns

    def test_multiclass_perfect_classifier_precision_recall_all_one(self):
        clf = _NoProbaClf(_CATEGORIES, _Y_CATEGORY)
        df = confusion_matrix_annotated(clf, _X_ARR, _Y_CATEGORY)
        assert df["precision"].values == pytest.approx(np.ones(5))
        assert df["recall"].values == pytest.approx(np.ones(5))

    def test_invalid_x_type_raises_type_error(self, clf_perfect):
        # Plain Python list is neither ndarray nor DataFrame → TypeError in _coerce_to_float64
        with pytest.raises(TypeError, match="Expected np.ndarray or pd.DataFrame"):
            confusion_matrix_annotated(clf_perfect, [[1, 2], [3, 4]], [0, 1])
