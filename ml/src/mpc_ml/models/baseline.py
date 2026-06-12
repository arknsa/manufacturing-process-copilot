"""
ml/src/mpc_ml/models/baseline.py
=====================================
Baseline model factories for the Manufacturing Process Copilot prediction tasks.

Two public functions:

* ``get_baseline_models()`` — returns the 5 configured, unfitted binary
  classifiers (LogisticRegression, DecisionTree, RandomForest, XGBoost,
  LightGBM) used in the Week 1 Day 5 baseline comparison experiment (Doc 04
  §Day 5).

* ``get_task_models(task)`` — returns the appropriate configured, unfitted
  estimator(s) for one of the four TARGET_COLS prediction tasks, per the
  task-model assignment in Doc 04 §Day 6.

Both functions return a ``Dict[str, Any]`` mapping from a short estimator name
(used as the MLflow run tag and comparison table row label) to an unfitted
sklearn-compatible estimator.  No data is required — these are pure factories.

Estimator configuration
-----------------------
Sklearn classifiers use ``class_weight='balanced'`` for binary and multi-class
imbalance handling.  XGBoost and LightGBM binary classifiers use
``scale_pos_weight`` as specified in Doc 04 §Day 5.  Multi-class XGBoost and
LightGBM use ``class_weight='balanced'`` (LightGBM) and rely on the caller to
pass ``sample_weight`` at fit-time for XGBoost, since the XGBoost sklearn API
class-weight support varies by version.

Training workflow contract
--------------------------
The caller (``train.py`` or ``02_baseline.ipynb``) is responsible for:

1. Wrapping the returned estimator in a full Pipeline::

       full_pipeline = Pipeline([
           ("preprocessor", build_pipeline()),
           ("model", estimator),
       ])

2. Fitting the full pipeline on the training split.

3. Evaluating on the validation split via
   ``evaluate_model(model, preprocessor, X_val, y_val, task)``.

4. Logging all runs to MLflow under ``mpc/delay_prediction/baseline`` via
   ``log_standard_params()``, ``log_standard_metrics()``, and
   ``log_standard_artifacts()``.

For ``delay_minutes``, fit on ``np.log1p(y_train)`` and evaluate on
``np.log1p(y_val)`` to match the log-scale regression target specified in
Doc 04 §Day 6.

Architecture references
-----------------------
* Doc 04 §Day 5 — Baseline algorithm list, class-weight strategy, expected AUC
* Doc 04 §Day 6 — Task-model assignments and primary metrics
* Doc 05       — ``models/baseline.py`` public API contract
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from mpc_ml.features.constants import TARGET_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default random seed applied to all stochastic estimators.  Exported so
#: that training scripts and notebooks can use the same value when splitting
#: data or initialising Optuna studies.
RANDOM_STATE: int = 42

#: Negative-to-positive class ratio derived from the documented 63 / 37
#: training-set imbalance (Doc 04 §Day 5).  Used as the default
#: ``scale_pos_weight`` argument for XGBoost and LightGBM binary classifiers.
#: Pass ``(y_train == 0).sum() / (y_train == 1).sum()`` from the actual
#: training split for a data-derived value.
DEFAULT_SCALE_POS_WEIGHT: float = 63.0 / 37.0  # ≈ 1.703

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "RANDOM_STATE",
    "DEFAULT_SCALE_POS_WEIGHT",
    "get_baseline_models",
    "get_task_models",
]

# ---------------------------------------------------------------------------
# Private module constants
# ---------------------------------------------------------------------------

_TASK_BINARY: str = "is_delayed"
_TASK_REGRESSION: str = "delay_minutes"
_TASK_ORDINAL: str = "delay_category"
_TASK_MULTICLASS: str = "delay_root_cause"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_task(task: str) -> None:
    """Raise ValueError if task is not a recognised TARGET_COL.

    Args:
        task: Task identifier string to validate.

    Raises:
        ValueError: If task is not in TARGET_COLS.
    """
    if task not in TARGET_COLS:
        raise ValueError(
            f"Unknown task {task!r}. Must be one of {list(TARGET_COLS)}."
        )


def _validate_scale_pos_weight(value: float) -> None:
    """Raise ValueError if scale_pos_weight is non-positive or non-finite.

    Args:
        value: scale_pos_weight value to validate.

    Raises:
        ValueError: If value is not a positive finite number.
    """
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"scale_pos_weight must be a positive finite float, got {value!r}. "
            "Compute it as (y_train == 0).sum() / (y_train == 1).sum() from "
            "the actual training split."
        )


# ===========================================================================
# Public API
# ===========================================================================


def get_baseline_models(
    scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Return five configured, unfitted binary classifiers for the is_delayed task.

    Each estimator is configured per the MPC benchmarking specification
    (Doc 04 §Day 5).  No data is required — all estimators are unfitted and
    ready to be wrapped in a full Pipeline.

    Estimators returned
    -------------------
    ``logistic_regression``
        Linear baseline.  Uses ``class_weight='balanced'`` and
        ``max_iter=1000`` to ensure convergence on the 41-feature input space.
        Establishes the minimum performance bar; expected val AUC: 0.68–0.75.

    ``decision_tree``
        Depth-5 classification tree.  Human-interpretable and suitable for
        stakeholder demonstrations.  Expected val AUC: 0.65–0.72.

    ``random_forest``
        100-tree ensemble with ``class_weight='balanced'``.  Handles the
        mixed continuous / binary / ordinal feature space well.
        Expected val AUC: 0.78–0.84.

    ``xgboost``
        Primary model candidate (XGBClassifier).  Uses ``scale_pos_weight``
        to counter the 63 / 37 class imbalance.  ``tree_method='hist'``
        for fast training on the simulation dataset size.
        Expected val AUC: 0.82–0.88.

    ``lightgbm``
        Primary model candidate (LGBMClassifier).  Uses ``scale_pos_weight``
        to counter the 63 / 37 class imbalance.  ``verbose=-1`` suppresses
        training output in notebook and MLflow logging contexts.
        Expected val AUC: 0.82–0.88.

    If either XGBoost or LightGBM achieves val AUC below 0.75, check:
    (1) no target-column leakage in X_train, (2) temporal train/val split
    preserved, (3) no target encoding applied to training data (Doc 04 §Day 5).

    Args:
        scale_pos_weight: Negative-to-positive class ratio applied to XGBoost
            and LightGBM.  Default is ``DEFAULT_SCALE_POS_WEIGHT`` (≈ 1.703)
            from the documented simulation class distribution.  Override with
            ``(y_train == 0).sum() / (y_train == 1).sum()`` computed from
            the actual training split for a data-derived value.
        random_state: Integer seed for all stochastic estimators.  All five
            models use this seed to produce reproducible results given the
            same training data and split.

    Returns:
        Dict mapping estimator name to an unfitted sklearn-compatible
        estimator.  Keys:

        * ``"logistic_regression"`` → :class:`~sklearn.linear_model.LogisticRegression`
        * ``"decision_tree"`` → :class:`~sklearn.tree.DecisionTreeClassifier`
        * ``"random_forest"`` → :class:`~sklearn.ensemble.RandomForestClassifier`
        * ``"xgboost"`` → :class:`~xgboost.XGBClassifier`
        * ``"lightgbm"`` → :class:`~lightgbm.LGBMClassifier`

        Each value is a fresh, unfitted instance ready for ``Pipeline.fit()``.

    Raises:
        ValueError: If ``scale_pos_weight`` is not a positive finite number.

    Examples:
        >>> from sklearn.pipeline import Pipeline
        >>> from mpc_ml.features.pipeline import build_pipeline
        >>> from mpc_ml.models.baseline import get_baseline_models
        >>>
        >>> for name, estimator in get_baseline_models().items():
        ...     full_pipeline = Pipeline([
        ...         ("preprocessor", build_pipeline()),
        ...         ("model", estimator),
        ...     ])
        ...     full_pipeline.fit(X_train, y_train)
        ...     preprocessor = full_pipeline.named_steps["preprocessor"]
        ...     model = full_pipeline.named_steps["model"]
        ...     # evaluate_model(model, preprocessor, X_val, y_val, "is_delayed")
    """
    _validate_scale_pos_weight(scale_pos_weight)

    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        ),
    }

    logger.debug(
        "get_baseline_models(): returning %d binary classifiers "
        "(scale_pos_weight=%.4f, random_state=%d).",
        len(models),
        scale_pos_weight,
        random_state,
    )

    return models


def get_task_models(
    task: str,
    scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Return configured, unfitted estimators for the given prediction task.

    Dispatches to the appropriate model family per the MPC task-model
    assignment (Doc 04 §Day 6):

    =====================  ==============================  ================
    Task                   Estimator(s) returned           Primary metric
    =====================  ==============================  ================
    ``is_delayed``         5 baseline classifiers          ROC-AUC
    ``delay_minutes``      XGBRegressor, LGBMRegressor     MAE, RMSE
    ``delay_category``     LGBMClassifier, XGBClassifier   Weighted F1
    ``delay_root_cause``   LGBMClassifier, XGBClassifier   Macro F1
    =====================  ==============================  ================

    Binary (is_delayed)
        Delegates to ``get_baseline_models()`` and returns all 5 classifiers
        for a full comparison run.

    Regression (delay_minutes)
        Returns XGBRegressor and LGBMRegressor.  Fit on ``np.log1p(y_train)``
        to match the log-scale regression target (Doc 04 §Day 6).

    Ordinal (delay_category, 5 classes)
        Returns LGBMClassifier with ``objective='multiclass'`` and
        XGBClassifier with ``objective='multi:softprob'``.  The ordinal class
        structure is not explicitly encoded — LightGBM and XGBoost both
        effectively discover ordinal thresholds through tree splits.
        LGBMClassifier uses ``class_weight='balanced'`` for the 5-class
        imbalance.  For XGBClassifier, pass ``sample_weight`` computed from
        class frequencies at ``full_pipeline.fit()`` time if per-class
        weighting is required.

    Multi-class (delay_root_cause, 7 classes)
        Returns the same model pair as ordinal.  Macro F1 is the primary
        metric and is sensitive to the rare root-cause classes
        (``quality_failure_rework``, ``multiple_causes``).  Verify that the
        validation split contains all 7 classes before evaluating ROC-AUC.

    Args:
        task: One of the four TARGET_COLS: ``"is_delayed"``,
            ``"delay_minutes"``, ``"delay_category"``,
            ``"delay_root_cause"``.
        scale_pos_weight: Negative-to-positive class ratio, forwarded to
            ``get_baseline_models()`` for the binary task only.  Ignored for
            regression and multi-class tasks.  See ``get_baseline_models()``
            for details.
        random_state: Integer seed applied to all stochastic estimators.

    Returns:
        Dict mapping estimator name to an unfitted sklearn-compatible
        estimator.  Key set depends on ``task``:

        * ``"is_delayed"`` → ``{"logistic_regression", "decision_tree",
          "random_forest", "xgboost", "lightgbm"}``
        * ``"delay_minutes"`` → ``{"xgboost", "lightgbm"}``
        * ``"delay_category"`` → ``{"lightgbm", "xgboost"}``
        * ``"delay_root_cause"`` → ``{"lightgbm", "xgboost"}``

    Raises:
        ValueError: If ``task`` is not in TARGET_COLS.
        ValueError: If ``scale_pos_weight`` is not a positive finite number.

    Examples:
        >>> import numpy as np
        >>> from sklearn.pipeline import Pipeline
        >>> from mpc_ml.features.pipeline import build_pipeline
        >>> from mpc_ml.models.baseline import get_task_models
        >>>
        >>> # Regression — fit on log1p-transformed target
        >>> for name, estimator in get_task_models("delay_minutes").items():
        ...     full_pipeline = Pipeline([
        ...         ("preprocessor", build_pipeline()),
        ...         ("model", estimator),
        ...     ])
        ...     full_pipeline.fit(X_train, np.log1p(y_regression_train))
        >>>
        >>> # Multi-class root cause
        >>> for name, estimator in get_task_models("delay_root_cause").items():
        ...     full_pipeline = Pipeline([
        ...         ("preprocessor", build_pipeline()),
        ...         ("model", estimator),
        ...     ])
        ...     full_pipeline.fit(X_train, y_root_cause_train)
    """
    _validate_task(task)
    _validate_scale_pos_weight(scale_pos_weight)

    if task == _TASK_BINARY:
        return get_baseline_models(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
        )

    if task == _TASK_REGRESSION:
        models: Dict[str, Any] = {
            "xgboost": XGBRegressor(
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
                random_state=random_state,
            ),
            "lightgbm": LGBMRegressor(
                n_jobs=-1,
                random_state=random_state,
                verbose=-1,
            ),
        }
        logger.debug(
            "get_task_models(task=%r): returning %d regressors (random_state=%d).",
            task,
            len(models),
            random_state,
        )
        return models

    # delay_category (ordinal 5-class) and delay_root_cause (multi-class 7-class)
    # both dispatch to multi-class classifiers.  The num_class parameter is
    # inferred from the training labels by both LightGBM and XGBoost at fit time.
    models = {
        "lightgbm": LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            random_state=random_state,
        ),
    }
    logger.debug(
        "get_task_models(task=%r): returning %d multi-class classifiers "
        "(random_state=%d).",
        task,
        len(models),
        random_state,
    )
    return models
