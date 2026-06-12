"""
ml/src/mpc_ml/models/evaluation.py
=====================================
Model evaluation utilities for all four Manufacturing Process Copilot prediction tasks.

Every metric computation flows through ``evaluate_model()``, which dispatches
to a task-specific evaluation path.  The three auxiliary public functions
(``precision_at_recall``, ``calibration_report``, ``confusion_matrix_annotated``)
can be called independently by notebooks and evaluation scripts for diagnostics.

Task mapping
------------
====================  ============================  ============================
TARGET_COL            Task type                     Primary metric
====================  ============================  ============================
``is_delayed``        Binary classification         ROC-AUC, F1 @ 0.40
``delay_minutes``     Regression                    MAE, RMSE
``delay_category``    Ordinal (5-class)             Weighted F1, Macro F1
``delay_root_cause``  Multi-class (7-class)         Macro F1
====================  ============================  ============================

Metric keys returned by ``evaluate_model()`` per task
------------------------------------------------------
*Binary (is_delayed):*
    ``val_roc_auc``, ``val_ap``, ``val_pr_auc``, ``val_brier_score``,
    ``val_f1``, ``val_ece``, ``val_precision_at_80pct_recall``,
    ``val_f1_at_040``, ``val_precision_at_040``, ``val_recall_at_040``

*Regression (delay_minutes):*
    ``val_mae``, ``val_rmse``, ``val_r2``

*Ordinal / Multi-class (delay_category, delay_root_cause):*
    ``val_weighted_f1``, ``val_macro_f1``, optionally ``val_roc_auc``

Architecture references
-----------------------
* Doc 04 §Day 5 — Baseline evaluation metrics specification
* Doc 04 §Day 6 — Benchmarking protocol: 0.40 operating threshold,
  precision @ 80% recall, per-task primary metrics
* Doc 05 — ``models/evaluation.py`` function contract
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, TypedDict, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve as _sklearn_calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from mpc_ml.features.constants import TARGET_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_TASK_BINARY: str = "is_delayed"
_TASK_REGRESSION: str = "delay_minutes"
_TASK_ORDINAL: str = "delay_category"
_TASK_MULTICLASS: str = "delay_root_cause"

# Operational alert threshold for binary classification (Doc 04 §Day 6).
# Confusion matrix and threshold-based label metrics use this value by default.
_OPERATING_THRESHOLD: float = 0.40

# Recall target for the precision-at-recall metric (Doc 04 §Day 5).
_TARGET_RECALL: float = 0.80

# Number of equal-width bins for calibration reporting.
_CALIBRATION_BINS: int = 10

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "MetricsDict",
    "CalibrationResult",
    "precision_at_recall",
    "calibration_report",
    "confusion_matrix_annotated",
    "evaluate_model",
]


# ===========================================================================
# Public type definitions
# ===========================================================================


class MetricsDict(TypedDict, total=False):
    """Flat mapping of metric name to float value returned by evaluate_model().

    All values are floats.  The keys present depend on the task argument passed
    to evaluate_model().  Keys for tasks not being evaluated are absent (not
    None) from the returned dict.

    Binary classification keys (is_delayed)
    ----------------------------------------
    val_roc_auc:
        ROC-AUC on the evaluation split.  Primary selection metric for the
        binary delay classifier.
    val_ap:
        Average Precision Score — area under the precision-recall curve.
    val_pr_auc:
        Alias for val_ap included for readability in MLflow run comparisons.
    val_brier_score:
        Brier score (mean squared error of probabilities).  Lower is better;
        0 is perfect.  A score below 0.15 indicates reasonable calibration.
    val_f1:
        F1 score at the default 0.5 probability threshold.
    val_ece:
        Expected Calibration Error computed over 10 equal-width bins.  Lower
        is better; ECE < 0.05 is the production acceptance threshold.
    val_precision_at_80pct_recall:
        Precision at the operating point where recall first reaches 0.80.
        Operational target: >= 0.50 (Doc 04 §Day 5).
    val_f1_at_040:
        F1 at the 0.40 operating threshold (Doc 04 §Day 6).
    val_precision_at_040:
        Precision at the 0.40 operating threshold.
    val_recall_at_040:
        Recall at the 0.40 operating threshold.

    Regression keys (delay_minutes)
    ---------------------------------
    val_mae:
        Mean Absolute Error in the target units (minutes, or log1p-minutes if
        the model was trained on a log-transformed target).
    val_rmse:
        Root Mean Squared Error in the target units.
    val_r2:
        R² coefficient of determination.  1.0 is perfect; can be negative for
        models worse than a constant predictor.

    Ordinal / multi-class keys (delay_category, delay_root_cause)
    --------------------------------------------------------------
    val_weighted_f1:
        F1 averaged by class support.  Robust to class imbalance.
    val_macro_f1:
        F1 averaged equally across all classes.  Sensitive to rare classes.
    val_roc_auc:
        ROC-AUC with one-vs-rest strategy, weighted average.  Omitted if
        predict_proba() is unavailable or the split is missing a class.
    """

    # Binary
    val_roc_auc: float
    val_ap: float
    val_pr_auc: float
    val_brier_score: float
    val_f1: float
    val_ece: float
    val_precision_at_80pct_recall: float
    val_f1_at_040: float
    val_precision_at_040: float
    val_recall_at_040: float
    # Regression
    val_mae: float
    val_rmse: float
    val_r2: float
    # Multi-class / ordinal
    val_weighted_f1: float
    val_macro_f1: float


@dataclass
class CalibrationResult:
    """Reliability curve data and Expected Calibration Error for a binary classifier.

    The reliability curve plots ``mean_predicted_value`` (x-axis) against
    ``fraction_of_positives`` (y-axis).  A perfectly calibrated model lies on
    the diagonal y = x.  Points above the diagonal indicate underconfidence;
    below indicate overconfidence.

    Attributes:
        fraction_of_positives: Observed positive-class rate for each non-empty
            calibration bin.  Shape (n_nonempty_bins,).
        mean_predicted_value: Mean predicted probability for each non-empty
            calibration bin.  Shape (n_nonempty_bins,).
        ece: Expected Calibration Error — weighted mean absolute difference
            between observed accuracy and predicted confidence across all bins.
            Float in [0, 1]; lower is better.
        n_bins: Number of equal-width bins requested.  Returned arrays may have
            fewer elements if some bins contained no predictions.
    """

    fraction_of_positives: np.ndarray
    mean_predicted_value: np.ndarray
    ece: float
    n_bins: int


# ===========================================================================
# Private helpers
# ===========================================================================


def _coerce_to_float64(
    X: Union[np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """Coerce X to a 2-D float64 numpy array without copying if possible.

    Args:
        X: Feature matrix as ndarray or DataFrame.

    Returns:
        float64 ndarray with the same shape as X.

    Raises:
        TypeError: If X is neither ndarray nor DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64)
    if isinstance(X, np.ndarray):
        return X.astype(np.float64, copy=False)
    raise TypeError(
        f"Expected np.ndarray or pd.DataFrame for X, got {type(X).__name__!r}. "
        "Pass the output of the fitted preprocessing pipeline."
    )


def _compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> float:
    """Compute Expected Calibration Error using equal-width probability bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples in bin b, acc(B_b) is the observed
    positive-class fraction in that bin, and conf(B_b) is the mean predicted
    probability.

    Args:
        y_true: Binary ground-truth labels, shape (n_samples,).
        y_prob: Predicted probabilities for the positive class, shape (n_samples,).
        n_bins: Number of equal-width bins covering [0, 1].

    Returns:
        ECE as a float in [0, 1].  Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_samples = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include the right endpoint in the last bin to capture y_prob = 1.0.
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (n_bin / n_samples) * abs(acc - conf)

    return float(ece)


def _validate_task(task: str) -> None:
    """Raise ValueError if task is not a recognised TARGET_COL.

    Args:
        task: Task identifier string to validate.

    Raises:
        ValueError: If task is not in TARGET_COLS.
    """
    if task not in TARGET_COLS:
        raise ValueError(
            f"Unknown task {task!r}. "
            f"Must be one of {list(TARGET_COLS)}."
        )


def _evaluate_binary(
    model: Any,
    X_np: np.ndarray,
    y_np: np.ndarray,
) -> MetricsDict:
    """Compute all binary classification metrics for the is_delayed task.

    Args:
        model: Fitted binary classifier with predict_proba().
        X_np: Preprocessed float64 feature matrix, shape (n_samples, n_features).
        y_np: Binary ground-truth labels (0 = on-time, 1 = delayed).

    Returns:
        MetricsDict with all ten binary keys populated.

    Raises:
        ValueError: If y_np contains fewer than 2 unique classes.
    """
    n_unique = len(np.unique(y_np))
    if n_unique < 2:
        raise ValueError(
            f"y contains only {n_unique} unique class(es). "
            "Binary evaluation requires both positive and negative examples. "
            "Verify that the evaluation split contains both delayed and on-time orders."
        )

    y_prob = model.predict_proba(X_np)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_040 = (y_prob >= _OPERATING_THRESHOLD).astype(int)

    roc_auc = float(roc_auc_score(y_np, y_prob))
    ap = float(average_precision_score(y_np, y_prob))
    brier = float(brier_score_loss(y_np, y_prob))
    ece = _compute_ece(y_np, y_prob, n_bins=_CALIBRATION_BINS)

    f1_05 = float(f1_score(y_np, y_pred_05, zero_division=0))

    f1_040 = float(f1_score(y_np, y_pred_040, zero_division=0))
    prec_040 = float(precision_score(y_np, y_pred_040, zero_division=0))
    rec_040 = float(recall_score(y_np, y_pred_040, zero_division=0))

    precision_arr, recall_arr, _ = precision_recall_curve(y_np, y_prob)
    meets_recall = recall_arr >= _TARGET_RECALL
    p_at_r80 = float(np.max(precision_arr[meets_recall])) if meets_recall.any() else 0.0

    metrics: MetricsDict = {
        "val_roc_auc": roc_auc,
        "val_ap": ap,
        "val_pr_auc": ap,
        "val_brier_score": brier,
        "val_f1": f1_05,
        "val_ece": ece,
        "val_precision_at_80pct_recall": p_at_r80,
        "val_f1_at_040": f1_040,
        "val_precision_at_040": prec_040,
        "val_recall_at_040": rec_040,
    }

    logger.info(
        "Binary eval (is_delayed): roc_auc=%.4f ap=%.4f brier=%.4f "
        "f1@0.5=%.4f f1@0.40=%.4f prec@0.40=%.4f rec@0.40=%.4f "
        "ece=%.4f p@r80=%.4f",
        roc_auc, ap, brier,
        f1_05, f1_040, prec_040, rec_040,
        ece, p_at_r80,
    )

    return metrics


def _evaluate_regression(
    model: Any,
    X_np: np.ndarray,
    y_np: np.ndarray,
) -> MetricsDict:
    """Compute regression metrics for the delay_minutes task.

    Args:
        model: Fitted regressor with predict().
        X_np: Preprocessed float64 feature matrix, shape (n_samples, n_features).
        y_np: Continuous target values (delay minutes, or log1p-transformed
            if the model was trained on a log-transformed target).

    Returns:
        MetricsDict with val_mae, val_rmse, val_r2.
    """
    y_pred = model.predict(X_np)

    mae = float(mean_absolute_error(y_np, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_np, y_pred)))
    r2 = float(r2_score(y_np, y_pred))

    metrics: MetricsDict = {
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2,
    }

    logger.info(
        "Regression eval (delay_minutes): mae=%.4f rmse=%.4f r2=%.4f",
        mae, rmse, r2,
    )

    return metrics


def _evaluate_multiclass(
    model: Any,
    X_np: np.ndarray,
    y_np: np.ndarray,
) -> MetricsDict:
    """Compute multi-class metrics for delay_category and delay_root_cause.

    Args:
        model: Fitted classifier with predict(), optionally predict_proba().
        X_np: Preprocessed float64 feature matrix, shape (n_samples, n_features).
        y_np: Categorical ground-truth labels.

    Returns:
        MetricsDict with val_weighted_f1, val_macro_f1, and optionally
        val_roc_auc if predict_proba() is available and the split contains
        all trained classes.
    """
    y_pred = model.predict(X_np)

    weighted_f1 = float(f1_score(y_np, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_np, y_pred, average="macro", zero_division=0))

    metrics: MetricsDict = {
        "val_weighted_f1": weighted_f1,
        "val_macro_f1": macro_f1,
    }

    if hasattr(model, "predict_proba"):
        try:
            y_prob_matrix = model.predict_proba(X_np)
            roc_auc = float(
                roc_auc_score(
                    y_np,
                    y_prob_matrix,
                    multi_class="ovr",
                    average="weighted",
                    labels=model.classes_,
                )
            )
            metrics["val_roc_auc"] = roc_auc
        except ValueError as exc:
            # Raised when the split is missing one or more trained classes.
            logger.warning(
                "Multi-class roc_auc_score skipped — %s. "
                "This can occur when a class is absent from the evaluation split.",
                exc,
            )

    roc_display = (
        f"{metrics['val_roc_auc']:.4f}" if "val_roc_auc" in metrics else "N/A"
    )
    logger.info(
        "Multi-class eval: weighted_f1=%.4f macro_f1=%.4f roc_auc=%s",
        weighted_f1, macro_f1, roc_display,
    )

    return metrics


# ===========================================================================
# Public API
# ===========================================================================


def precision_at_recall(
    model: Any,
    X_transformed: Union[np.ndarray, pd.DataFrame],
    y: Union[pd.Series, np.ndarray],
    target_recall: float = _TARGET_RECALL,
) -> float:
    """Return the precision at the highest threshold that achieves target_recall.

    Computes the full precision-recall curve and returns the maximum precision
    over all operating points where recall >= target_recall.  The result is the
    operational precision when the classifier is tuned to catch at least
    target_recall of all delayed orders.

    Typical usage: the MPC alert system requires recall >= 0.80 to surface most
    delays.  Precision at that operating point determines the false-alert rate.
    The production acceptance threshold is precision >= 0.50 at recall = 0.80
    (Doc 04 §Day 5).

    Args:
        model: Fitted binary classifier implementing predict_proba().  The
            positive class must be the higher-value class (index 1 in
            model.classes_), which is 1 (= delayed) for is_delayed.
        X_transformed: Preprocessed feature matrix, shape (n_samples, n_features).
            Must be the output of the fitted preprocessing pipeline — not raw input.
        y: Binary ground-truth labels (0 = on-time, 1 = delayed),
            shape (n_samples,).
        target_recall: Minimum required recall.  The function returns the maximum
            precision at or above this recall level.  Default: 0.80.

    Returns:
        Maximum precision at or above target_recall.  Returns 0.0 if no
        operating point achieves the target recall.

    Raises:
        ValueError: If target_recall is not in [0, 1].
        AttributeError: If model does not implement predict_proba().

    Examples:
        >>> preprocessor = full_pipeline.named_steps["preprocessor"]
        >>> X_val_t = preprocessor.transform(X_val)
        >>> p = precision_at_recall(model, X_val_t, y_val, target_recall=0.80)
        >>> print(f"Precision at 80% recall: {p:.3f}")
    """
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError(
            f"target_recall must be in [0, 1], got {target_recall!r}."
        )

    X_np = _coerce_to_float64(X_transformed)
    y_np = np.asarray(y)

    y_prob = model.predict_proba(X_np)[:, 1]
    precision_arr, recall_arr, _ = precision_recall_curve(y_np, y_prob)

    meets_recall = recall_arr >= target_recall
    if not meets_recall.any():
        logger.warning(
            "precision_at_recall: no operating point achieves recall >= %.2f. "
            "Returning 0.0.",
            target_recall,
        )
        return 0.0

    return float(np.max(precision_arr[meets_recall]))


def calibration_report(
    model: Any,
    X_transformed: Union[np.ndarray, pd.DataFrame],
    y: Union[pd.Series, np.ndarray],
    n_bins: int = _CALIBRATION_BINS,
) -> CalibrationResult:
    """Compute the reliability curve and Expected Calibration Error for a binary model.

    A well-calibrated model produces probability outputs that match the observed
    positive-class frequency: among orders predicted with 60% delay probability,
    approximately 60% should actually be delayed.

    The returned CalibrationResult supports two uses:

    1. Plotting a reliability diagram: ``mean_predicted_value`` (x) vs
       ``fraction_of_positives`` (y).  A diagonal line represents perfect
       calibration.
    2. Quantifying miscalibration via ``ece``.  The MPC acceptance threshold
       is ECE < 0.05 for the binary delay classifier (Doc 04 §Day 5).

    Args:
        model: Fitted binary classifier implementing predict_proba().
        X_transformed: Preprocessed feature matrix, shape (n_samples, n_features).
        y: Binary ground-truth labels (0 = on-time, 1 = delayed),
            shape (n_samples,).
        n_bins: Number of equal-width calibration bins in [0, 1].  Default: 10.

    Returns:
        CalibrationResult with fraction_of_positives, mean_predicted_value,
        ece, and n_bins.

    Raises:
        ValueError: If n_bins < 2.
        AttributeError: If model does not implement predict_proba().

    Examples:
        >>> result = calibration_report(model, X_val_t, y_val)
        >>> print(f"ECE: {result.ece:.4f}")
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result.mean_predicted_value, result.fraction_of_positives)
        >>> plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins!r}.")

    X_np = _coerce_to_float64(X_transformed)
    y_np = np.asarray(y)

    y_prob = model.predict_proba(X_np)[:, 1]

    frac_pos, mean_pred = _sklearn_calibration_curve(
        y_np,
        y_prob,
        n_bins=n_bins,
        strategy="uniform",
    )
    ece = _compute_ece(y_np, y_prob, n_bins=n_bins)

    logger.debug(
        "calibration_report: n_bins=%d ece=%.4f n_samples=%d",
        n_bins, ece, len(y_np),
    )

    return CalibrationResult(
        fraction_of_positives=frac_pos,
        mean_predicted_value=mean_pred,
        ece=ece,
        n_bins=n_bins,
    )


def confusion_matrix_annotated(
    model: Any,
    X_transformed: Union[np.ndarray, pd.DataFrame],
    y: Union[pd.Series, np.ndarray],
    threshold: float = _OPERATING_THRESHOLD,
) -> pd.DataFrame:
    """Return a confusion matrix DataFrame annotated with per-class precision and recall.

    For binary classifiers (len(model.classes_) == 2), predictions are made at
    the given probability threshold (default 0.40, the operational alert
    threshold from Doc 04 §Day 6).  For multi-class and ordinal classifiers,
    threshold is ignored and model.predict() is used directly.

    The returned DataFrame has:

    * Index: true class labels from model.classes_; index name is ``"true_label"``.
    * Columns: predicted class labels from model.classes_, then ``"precision"``,
      then ``"recall"``.
    * Confusion matrix cells: raw integer prediction counts.
    * ``precision`` column: per-class TP / (TP + FP); 0.0 on zero division.
    * ``recall`` column: per-class TP / (TP + FN); 0.0 on zero division.

    For binary classification, the row for class 1 (delayed) contains the
    precision and recall most relevant to operational alerting.

    Args:
        model: Fitted sklearn-compatible classifier with model.classes_.
        X_transformed: Preprocessed feature matrix, shape (n_samples, n_features).
        y: Ground-truth class labels, shape (n_samples,).
        threshold: Probability threshold for binary classification.  Ignored for
            models with more than 2 classes.  Default: 0.40.

    Returns:
        DataFrame of shape (n_classes, n_classes + 2).

    Raises:
        ValueError: If threshold is not in (0, 1) for a binary classifier.

    Examples:
        >>> cm_df = confusion_matrix_annotated(model, X_val_t, y_val)
        >>> print(cm_df)
        #              0    1  precision    recall
        # true_label
        # 0          427   41      0.918     0.912
        # 1           38  194      0.826     0.836
    """
    X_np = _coerce_to_float64(X_transformed)
    y_np = np.asarray(y)
    classes = model.classes_

    is_binary = len(classes) == 2 and hasattr(model, "predict_proba")

    if is_binary:
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"threshold must be in (0, 1) for binary classification, "
                f"got {threshold!r}."
            )
        y_prob = model.predict_proba(X_np)[:, 1]
        # np.where preserves the original class dtype (int, float, or str).
        y_pred = np.where(y_prob >= threshold, classes[1], classes[0])
    else:
        y_pred = model.predict(X_np)

    cm = confusion_matrix(y_np, y_pred, labels=classes)
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df.index.name = "true_label"

    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)

    precision_per_class = np.where(
        col_sums > 0,
        np.diag(cm) / np.where(col_sums > 0, col_sums, 1),
        0.0,
    )
    recall_per_class = np.where(
        row_sums > 0,
        np.diag(cm) / np.where(row_sums > 0, row_sums, 1),
        0.0,
    )

    df["precision"] = precision_per_class
    df["recall"] = recall_per_class

    logger.debug(
        "confusion_matrix_annotated: n_classes=%d is_binary=%s threshold=%.2f",
        len(classes), is_binary, threshold,
    )

    return df


def evaluate_model(
    model: Any,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    task: str,
) -> MetricsDict:
    """Compute the full evaluation metric set for a given prediction task.

    Transforms X using the fitted preprocessing pipeline, generates model
    predictions, and computes all metrics defined for the task in the MPC
    benchmarking specification (Doc 04 §Day 6).

    Task dispatch:

    *Binary (is_delayed):*
        ROC-AUC, Average Precision, PR-AUC, Brier score, F1 @ 0.5,
        F1 / precision / recall @ 0.40 operating threshold, precision at 80%
        recall, Expected Calibration Error.

    *Regression (delay_minutes):*
        MAE, RMSE, R².  Pass y in the same scale as the model's training
        target — if trained on log1p(delay_minutes), pass log1p-transformed y.

    *Ordinal (delay_category, 5 classes) and multi-class (delay_root_cause, 7 classes):*
        Weighted F1, macro F1.  ROC-AUC (OVR, weighted) added when
        predict_proba() is available and all trained classes appear in y.

    Args:
        model: Fitted sklearn-compatible estimator with predict().  Must also
            implement predict_proba() for all classification tasks.
        pipeline: Fitted preprocessing-only sklearn Pipeline — the 3-step
            pipeline returned by build_pipeline() after fit().  Does NOT include
            the model step.  Used to transform X to the 41-column model input
            space.
        X: Raw 37-column feature DataFrame, shape (n_samples, 37).  Must not
            contain any TARGET_COLS.
        y: Ground-truth labels or values corresponding to task.
            Shape: (n_samples,).
        task: One of the four TARGET_COLS: ``"is_delayed"``,
            ``"delay_minutes"``, ``"delay_category"``, ``"delay_root_cause"``.

    Returns:
        MetricsDict with all keys for the given task populated.  See the
        MetricsDict TypedDict definition for the complete key inventory.

    Raises:
        ValueError: If task is not a recognised TARGET_COL.
        ValueError: If X contains any TARGET_COLS columns (target leakage).
        ValueError: If y contains fewer than 2 unique classes for binary tasks.
        AttributeError: If model is missing predict_proba() for a classification task.

    Examples:
        >>> full_pipeline = Pipeline([
        ...     ("preprocessor", build_pipeline()),
        ...     ("model", XGBClassifier(**best_params)),
        ... ])
        >>> full_pipeline.fit(X_train, y_train)
        >>> preprocessor = full_pipeline.named_steps["preprocessor"]
        >>>
        >>> metrics = evaluate_model(
        ...     model=full_pipeline.named_steps["model"],
        ...     pipeline=preprocessor,
        ...     X=X_val,
        ...     y=y_val,
        ...     task="is_delayed",
        ... )
        >>> print(f"val_roc_auc:   {metrics['val_roc_auc']:.4f}")
        >>> print(f"val_f1_at_040: {metrics['val_f1_at_040']:.4f}")
    """
    _validate_task(task)

    leaking_cols = [c for c in X.columns if c in TARGET_COLS]
    if leaking_cols:
        raise ValueError(
            f"X contains target column(s): {leaking_cols}. "
            "Drop TARGET_COLS from X before calling evaluate_model()."
        )

    X_transformed = pipeline.transform(X)
    X_np = _coerce_to_float64(X_transformed)
    y_np = np.asarray(y)

    if task == _TASK_BINARY:
        return _evaluate_binary(model, X_np, y_np)
    if task == _TASK_REGRESSION:
        return _evaluate_regression(model, X_np, y_np)
    # delay_category and delay_root_cause both use the multi-class evaluation path.
    return _evaluate_multiclass(model, X_np, y_np)
