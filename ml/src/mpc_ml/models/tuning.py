"""
ml/src/mpc_ml/models/tuning.py
=====================================
Optuna hyperparameter optimisation for the XGBoost and LightGBM model candidates.

Three public functions:

* ``build_optuna_objective()`` — factory that returns a closed-over Optuna
  objective callable.  The objective runs TimeSeriesSplit cross-validation
  on the provided training data, reporting intermediate fold scores to the
  pruner.

* ``run_study()`` — creates and runs an Optuna study with ``TPESampler`` and
  ``MedianPruner`` configured per Doc 04 §Day 7.

* ``best_params_to_model()`` — reconstructs the champion estimator from the
  best trial's sampled hyperparameters, combined with the production fixed
  params (``tree_method``, ``verbosity``, ``scale_pos_weight``, etc.).

Search spaces (Doc 04 §Day 7)
-------------------------------
**XGBoost** (8 parameters):
    ``n_estimators`` [200, 1000] int log, ``max_depth`` [3, 8] int,
    ``learning_rate`` [0.01, 0.30] float log, ``subsample`` [0.6, 1.0] float,
    ``colsample_bytree`` [0.6, 1.0] float, ``min_child_weight`` [1, 10] int,
    ``reg_alpha`` [1e-8, 1.0] float log, ``reg_lambda`` [1e-8, 1.0] float log.

**LightGBM** (8 parameters):
    ``n_estimators`` [200, 1000] int log, ``num_leaves`` [20, 200] int,
    ``learning_rate`` [0.01, 0.30] float log, ``subsample`` [0.6, 1.0] float,
    ``colsample_bytree`` [0.6, 1.0] float, ``min_child_samples`` [5, 50] int,
    ``reg_alpha`` [1e-8, 1.0] float log, ``reg_lambda`` [1e-8, 1.0] float log.

Objective metrics per task
---------------------------
=========================================  ==============================
Task                                       Metric (always maximised)
=========================================  ==============================
``is_delayed`` (binary)                    ROC-AUC
``delay_minutes`` (regression)             −MAE (negative for direction)
``delay_category`` (ordinal 5-class)       Weighted F1
``delay_root_cause`` (multi-class 7-class) Macro F1
=========================================  ==============================

CV design
---------
Inside the objective, ``build_pipeline()`` is rebuilt from scratch for each
fold.  This ensures the ``StandardScaler`` mean/std and ``ColumnSelector``
cold-start defaults are learned exclusively from the fold's training rows,
preventing the subtle bias that occurs when a single preprocessor is fitted
on the entire training set before CV.

Training workflow contract
--------------------------
The caller (``train.py`` or ``03_tuning.ipynb``) is responsible for:

1. Splitting data temporally (train / val / test).
2. Building the objective with ``build_optuna_objective(X_train, y_train, ...)``.
3. Running the study with ``run_study(objective, n_trials=N_TRIALS)``.
4. Reconstructing the champion estimator with
   ``best_params_to_model(study, model_type=..., task=...)``.
5. Wrapping the estimator in a full Pipeline, fitting on the entire training
   set, evaluating on the validation split via ``evaluate_model()``, and
   logging to MLflow via ``log_model_with_signature()`` and
   ``log_standard_params(study.best_params)``.

The held-out test split must not be touched until ``scripts/evaluate.py``
is run after the champion is selected.

Architecture references
-----------------------
* Doc 04 §Day 7 — search spaces, study settings, expected post-tuning AUC
* Doc 05       — ``models/tuning.py`` public API contract
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from mpc_ml.features.constants import DELAY_CATEGORY_ORDER, ROOT_CAUSE_CLASSES, TARGET_COLS
from mpc_ml.features.pipeline import build_pipeline
from mpc_ml.models.baseline import DEFAULT_SCALE_POS_WEIGHT, RANDOM_STATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default number of Optuna trials per study.  Set in Doc 04 §Day 7.
N_TRIALS: int = 100

#: Default number of TimeSeriesSplit folds used inside the CV objective.
N_CV_SPLITS: int = 5

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "N_TRIALS",
    "N_CV_SPLITS",
    "build_optuna_objective",
    "run_study",
    "best_params_to_model",
    "encode_multiclass_labels_for_xgboost",
]

# ---------------------------------------------------------------------------
# Private string constants
# ---------------------------------------------------------------------------

_MODEL_XGBOOST: str = "xgboost"
_MODEL_LIGHTGBM: str = "lightgbm"
_VALID_MODEL_TYPES: tuple = (_MODEL_XGBOOST, _MODEL_LIGHTGBM)

_TASK_BINARY: str = "is_delayed"
_TASK_REGRESSION: str = "delay_minutes"
_TASK_ORDINAL: str = "delay_category"
_TASK_MULTICLASS: str = "delay_root_cause"


# ===========================================================================
# Private helpers
# ===========================================================================


def _validate_model_type(model_type: str) -> None:
    """Raise ValueError if model_type is not 'xgboost' or 'lightgbm'.

    Args:
        model_type: Framework identifier string to validate.

    Raises:
        ValueError: If model_type is not in the valid set.
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Must be one of {list(_VALID_MODEL_TYPES)}."
        )


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


def encode_multiclass_labels_for_xgboost(y: np.ndarray, task: str) -> np.ndarray:
    """Encode string multi-class labels to XGBoost-compatible integers.

    ``XGBClassifier`` with ``objective='multi:softprob'`` requires labels in
    the range ``[0, num_class-1]`` as integers.  LightGBM handles string
    labels natively; this function is only needed for the XGBoost path.

    Use this in the final training step after ``run_study()`` whenever
    ``model_type="xgboost"`` and ``task`` is ``"delay_category"`` or
    ``"delay_root_cause"``::

        y_encoded = encode_multiclass_labels_for_xgboost(y_train, "delay_root_cause")
        full_pipeline.fit(X_train, y_encoded)

    Canonical label orders per constants.py:
    - ``delay_category``   → ``DELAY_CATEGORY_ORDER`` (0=on_time, …, 4=critical_delay)
    - ``delay_root_cause`` → ``ROOT_CAUSE_CLASSES`` (0=machine_breakdown, …, 6=setup_overrun)

    Args:
        y: Ground-truth label array with string values, shape ``(n,)``.
        task: ``"delay_category"`` or ``"delay_root_cause"``.

    Returns:
        Integer-encoded array, same length as ``y``, dtype int64.

    Raises:
        KeyError: If any label in ``y`` is not in the canonical label set.
        ValueError: If ``task`` is not ``"delay_category"`` or
            ``"delay_root_cause"``.
    """
    if task == _TASK_ORDINAL:
        label_map: Dict[str, int] = {
            label: idx for idx, label in enumerate(DELAY_CATEGORY_ORDER)
        }
    elif task == _TASK_MULTICLASS:
        label_map = {
            label: idx for idx, label in enumerate(ROOT_CAUSE_CLASSES)
        }
    else:
        raise ValueError(
            f"encode_multiclass_labels_for_xgboost() is only valid for "
            f"'delay_category' and 'delay_root_cause', got {task!r}."
        )
    return np.array([label_map[str(v)] for v in y], dtype=np.int64)


def _sample_params(trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
    """Sample hyperparameters from the Doc 04 search space for model_type.

    All parameter names match the sklearn API parameter names accepted by
    both the estimator constructors and ``model.set_params()``.

    Args:
        trial: Active Optuna trial.  Calls ``suggest_*`` methods to sample
            values and record them in the trial's param store.
        model_type: ``"xgboost"`` or ``"lightgbm"``.

    Returns:
        Dict of sampled hyperparameter name → value.  Does not include
        fixed production params (``tree_method``, ``verbosity``,
        ``scale_pos_weight``, ``n_jobs``, ``random_state``).
    """
    if model_type == _MODEL_XGBOOST:
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
    # LightGBM
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }


def _build_estimator(
    params: Dict[str, Any],
    model_type: str,
    task: str,
    scale_pos_weight: float,
    random_state: int,
) -> Any:
    """Build a configured, unfitted estimator from sampled params.

    Combines the tunable parameters from ``params`` with the fixed production
    params (``tree_method``, ``verbosity``, ``n_jobs``, class-weight strategy,
    ``random_state``) to produce a complete estimator configuration.

    Args:
        params: Tunable hyperparameters sampled by ``_sample_params()``.
        model_type: ``"xgboost"`` or ``"lightgbm"``.
        task: One of the four TARGET_COLS; determines estimator class
            (classifier vs regressor) and class-imbalance strategy.
        scale_pos_weight: Negative-to-positive ratio for binary classifiers.
            Forwarded to ``XGBClassifier.scale_pos_weight`` and
            ``LGBMClassifier.scale_pos_weight``.
        random_state: Integer seed for reproducibility.

    Returns:
        Unfitted sklearn-compatible estimator with all parameters applied.
    """
    is_regression = (task == _TASK_REGRESSION)
    is_multiclass = (task in (_TASK_ORDINAL, _TASK_MULTICLASS))

    if model_type == _MODEL_XGBOOST:
        if is_regression:
            return XGBRegressor(
                **params,
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
                random_state=random_state,
            )
        if is_multiclass:
            return XGBClassifier(
                **params,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
                random_state=random_state,
            )
        # Binary
        return XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            random_state=random_state,
        )

    # LightGBM
    if is_regression:
        return LGBMRegressor(
            **params,
            subsample_freq=1,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    if is_multiclass:
        return LGBMClassifier(
            **params,
            objective="multiclass",
            class_weight="balanced",
            subsample_freq=1,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    # Binary
    return LGBMClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        subsample_freq=1,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )


def _compute_fold_score(
    full_pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    task: str,
) -> float:
    """Compute the optimization metric for the given task on a single fold.

    The metric is always oriented for maximisation:
    - Binary: ROC-AUC (higher is better).
    - Regression: negative MAE (higher = smaller error = better).
    - Ordinal: weighted F1 (higher is better).
    - Multi-class: macro F1 (higher is better).

    Args:
        full_pipeline: Fitted Pipeline([preprocessor, model]).
        X_val: Raw 37-column validation DataFrame.  Transformed internally
            by ``full_pipeline.predict*``.
        y_val: Ground-truth labels or values, shape (n_val,).
        task: One of the four TARGET_COLS.

    Returns:
        Scalar score oriented for maximisation.  Returns 0.0 on any scoring
        failure (e.g., single-class val split for ROC-AUC).
    """
    try:
        if task == _TASK_BINARY:
            y_prob = full_pipeline.predict_proba(X_val)[:, 1]
            return float(roc_auc_score(y_val, y_prob))

        if task == _TASK_REGRESSION:
            y_pred = full_pipeline.predict(X_val)
            return float(-mean_absolute_error(y_val, y_pred))

        if task == _TASK_ORDINAL:
            y_pred = full_pipeline.predict(X_val)
            return float(f1_score(y_val, y_pred, average="weighted", zero_division=0))

        # delay_root_cause — macro F1 is sensitive to rare classes
        y_pred = full_pipeline.predict(X_val)
        return float(f1_score(y_val, y_pred, average="macro", zero_division=0))

    except ValueError as exc:
        # Most common cause: single-class val split for ROC-AUC.
        # Return 0.0 so the trial is not aborted but scores poorly.
        logger.warning("Fold scoring failed (%s). Returning 0.0.", exc)
        return 0.0


# ===========================================================================
# Public API
# ===========================================================================


def build_optuna_objective(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    *,
    model_type: str,
    task: str,
    n_splits: int = N_CV_SPLITS,
    scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT,
    random_state: int = RANDOM_STATE,
) -> Callable[[optuna.Trial], float]:
    """Build a closed-over Optuna objective for TimeSeriesSplit cross-validation.

    Returns a callable ``objective(trial) → float`` that:

    1. Samples hyperparameters from the Doc 04 §Day 7 search space.
    2. Builds a fresh full Pipeline (``build_pipeline()`` + tuned estimator)
       for each CV fold — preventing preprocessor leakage from validation rows.
    3. Reports each fold's score to Optuna's pruner via
       ``trial.report(score, step=fold)``.
    4. Raises ``optuna.TrialPruned`` if ``trial.should_prune()`` returns
       ``True`` after any fold.
    5. Returns the mean score across all completed folds.

    The preprocessing pipeline is rebuilt from scratch inside each fold loop,
    not fitted once on the full training set.  This ensures that
    ``StandardScaler`` statistics and ``ColumnSelector.cold_start_defaults_``
    are learned only from the fold's training rows, which is the correct CV
    protocol for pipelines that learn statistics from data.

    Args:
        X_train: Raw 37-column feature DataFrame, shape (n_train, 37).  Must
            not contain any TARGET_COLS.  Rows must be in temporal order
            (earliest first) so that ``TimeSeriesSplit`` creates valid
            train-before-val splits.
        y_train: Ground-truth labels or values, shape (n_train,).  For
            ``delay_minutes``, pass ``np.log1p(raw_delay_minutes)`` if the
            model will be trained on a log-scale target.
        model_type: Framework identifier — ``"xgboost"`` or ``"lightgbm"``.
            Determines the hyperparameter search space and estimator class.
        task: One of the four TARGET_COLS.  Determines the CV objective metric
            and estimator class (classifier vs regressor).
        n_splits: Number of TimeSeriesSplit folds.  Each fold uses an
            expanding training window.  Default: ``N_CV_SPLITS`` (5).
        scale_pos_weight: Negative-to-positive class ratio for binary
            classification tasks.  Ignored for regression and multi-class.
            Default: ``DEFAULT_SCALE_POS_WEIGHT`` (≈ 1.703 from Doc 04).
        random_state: Seed applied to all stochastic estimators inside the
            objective.  Fixed across all trials for reproducible fold
            training.

    Returns:
        A callable ``objective(trial: optuna.Trial) → float`` suitable for
        ``study.optimize(objective, n_trials=N)``.  The objective returns the
        mean CV score across completed folds, oriented for maximisation (a
        higher value is always better).

    Raises:
        ValueError: If ``model_type`` is not ``"xgboost"`` or ``"lightgbm"``.
        ValueError: If ``task`` is not in TARGET_COLS.
        ValueError: If ``n_splits`` < 2.

    Examples:
        >>> import numpy as np
        >>> from mpc_ml.models.tuning import build_optuna_objective, run_study
        >>>
        >>> # Binary delay classifier — XGBoost
        >>> spw = (y_train == 0).sum() / (y_train == 1).sum()
        >>> objective = build_optuna_objective(
        ...     X_train, y_train,
        ...     model_type="xgboost",
        ...     task="is_delayed",
        ...     scale_pos_weight=spw,
        ... )
        >>> study = run_study(objective, n_trials=100)
        >>>
        >>> # Regression — LightGBM on log1p-transformed target
        >>> objective_reg = build_optuna_objective(
        ...     X_train, np.log1p(y_minutes_train),
        ...     model_type="lightgbm",
        ...     task="delay_minutes",
        ... )
    """
    _validate_model_type(model_type)
    _validate_task(task)
    _validate_scale_pos_weight(scale_pos_weight)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits!r}.")
    leaking_cols = [c for c in X_train.columns if c in TARGET_COLS]
    if leaking_cols:
        raise ValueError(
            f"X_train contains target column(s): {leaking_cols}. "
            "Drop TARGET_COLS from X_train before calling build_optuna_objective()."
        )

    y_np: np.ndarray = np.asarray(y_train)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # XGBoost multi:softprob requires integer labels; precompute a flag so the
    # per-fold encode call is skipped for all other (model_type, task) combos.
    _needs_xgb_label_encode: bool = (
        model_type == _MODEL_XGBOOST
        and task in (_TASK_ORDINAL, _TASK_MULTICLASS)
    )

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, model_type)
        base_estimator = _build_estimator(
            params, model_type, task, scale_pos_weight, random_state
        )

        fold_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train: pd.DataFrame = X_train.iloc[train_idx]
            y_fold_train: np.ndarray = y_np[train_idx]
            X_fold_val: pd.DataFrame = X_train.iloc[val_idx]
            y_fold_val: np.ndarray = y_np[val_idx]

            if _needs_xgb_label_encode:
                y_fold_train = encode_multiclass_labels_for_xgboost(y_fold_train, task)
                y_fold_val = encode_multiclass_labels_for_xgboost(y_fold_val, task)

            fold_estimator = clone(base_estimator)
            full_pipeline = Pipeline([
                ("preprocessor", build_pipeline()),
                ("model", fold_estimator),
            ])

            try:
                full_pipeline.fit(X_fold_train, y_fold_train)
            except Exception as exc:
                logger.warning(
                    "Trial %d fold %d fit failed: %s. Pruning trial.",
                    trial.number, fold, exc,
                )
                raise optuna.TrialPruned() from exc

            fold_score = _compute_fold_score(
                full_pipeline, X_fold_val, y_fold_val, task
            )
            fold_scores.append(fold_score)

            trial.report(fold_score, step=fold)
            if trial.should_prune():
                logger.debug(
                    "Trial %d pruned at fold %d (score=%.4f).",
                    trial.number, fold, fold_score,
                )
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        logger.debug(
            "Trial %d complete: model_type=%r task=%r mean_score=%.4f "
            "fold_scores=%s",
            trial.number,
            model_type,
            task,
            mean_score,
            [f"{s:.4f}" for s in fold_scores],
        )
        return mean_score

    return objective


def run_study(
    objective: Callable[[optuna.Trial], float],
    *,
    n_trials: int = N_TRIALS,
    study_name: Optional[str] = None,
    seed: int = RANDOM_STATE,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Create and run an Optuna study with MedianPruner and TPESampler.

    Study configuration per Doc 04 §Day 7:

    * **Direction:** ``"maximize"`` — all objective metrics are oriented
      for maximisation (including regression, which returns −MAE).
    * **Sampler:** ``TPESampler(seed=seed)`` — Tree-structured Parzen
      Estimator; Bayesian optimisation that models the distribution of
      good hyperparameters from completed trials.
    * **Pruner:** ``MedianPruner(n_startup_trials=5, n_warmup_steps=0)`` —
      discards trials whose intermediate fold score falls below the median
      of all previous trials at the same fold step.  The 5-trial warmup
      ensures the pruner has enough data before making pruning decisions.

    Optuna's per-trial INFO logs are suppressed internally (set to WARNING)
    to keep notebook and training-script output clean.  The summary
    statistics (best value, complete/pruned counts) are logged at INFO level
    via the module logger.

    Args:
        objective: Callable returned by ``build_optuna_objective()``.
            Signature: ``(trial: optuna.Trial) → float``.
        n_trials: Total number of trials to run.  Includes both completed
            and pruned trials.  Default: ``N_TRIALS`` (100).
        study_name: Optional display name for the study in the Optuna
            dashboard.  If ``None``, Optuna assigns an auto-generated name.
        seed: Random seed for the TPE sampler.  Fixed for reproducibility.
            Default: ``RANDOM_STATE`` (42).
        show_progress_bar: Display a ``tqdm`` progress bar during
            optimisation.  Requires ``tqdm`` to be installed.  Default:
            ``False`` (clean for MLflow and CI contexts).

    Returns:
        Completed :class:`optuna.Study`.  Inspect results via:

        * ``study.best_value`` — best objective score achieved.
        * ``study.best_params`` — hyperparameters of the best trial.
        * ``study.best_trial.number`` — trial index of the champion.
        * ``study.trials_dataframe()`` — full trial history as DataFrame.

    Examples:
        >>> objective = build_optuna_objective(
        ...     X_train, y_train,
        ...     model_type="xgboost",
        ...     task="is_delayed",
        ... )
        >>> study = run_study(objective, n_trials=100)
        >>> print(f"Best val AUC: {study.best_value:.4f}")
        >>> print(f"Best params: {study.best_params}")
        >>>
        >>> # Pass best_params directly to MLflow:
        >>> from mpc_ml.tracking.mlflow_utils import log_standard_params
        >>> with start_run(experiment_name, "xgb_champion"):
        ...     log_standard_params(study.best_params)
    """
    # Silence Optuna's per-trial INFO output to keep training logs clean.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    logger.info(
        "Optuna study started: n_trials=%d study_name=%r seed=%d "
        "pruner=MedianPruner(n_startup_trials=5)",
        n_trials,
        study_name,
        seed,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    n_complete = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    )

    if n_complete == 0:
        logger.warning(
            "Optuna study %r: all %d trial(s) were pruned or failed — "
            "no completed trials.  best_params_to_model() will raise.  "
            "Increase n_trials or widen the search space.",
            study_name,
            len(study.trials),
        )
    else:
        logger.info(
            "Optuna study complete: best_value=%.4f best_trial=%d "
            "n_total=%d n_complete=%d n_pruned=%d",
            study.best_value,
            study.best_trial.number,
            len(study.trials),
            n_complete,
            n_pruned,
        )

    return study


def best_params_to_model(
    study: optuna.Study,
    *,
    model_type: str,
    task: str,
    scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT,
    random_state: int = RANDOM_STATE,
) -> Any:
    """Reconstruct the champion estimator from the best trial's parameters.

    Combines the tunable hyperparameters from ``study.best_params`` with the
    fixed production parameters (``tree_method``, ``verbosity``,
    ``scale_pos_weight``, ``n_jobs``, ``random_state``) to produce a fully
    configured, unfitted estimator ready for final training on the complete
    training set.

    The returned estimator should be wrapped in a full Pipeline and fitted
    on the entire training split before logging to MLflow::

        estimator = best_params_to_model(study, model_type="xgboost", task="is_delayed")
        full_pipeline = Pipeline([
            ("preprocessor", build_pipeline()),
            ("model", estimator),
        ])
        full_pipeline.fit(X_train, y_train)

    **XGBoost multi-class label encoding (delay_category / delay_root_cause):**
    When ``model_type="xgboost"`` and ``task`` is ``"delay_category"`` or
    ``"delay_root_cause"``, XGBoost requires integer labels ``[0, num_class-1]``.
    The tuning objective encodes labels automatically, but the *final* training
    call must also encode ``y_train`` before ``full_pipeline.fit()``.  Use
    ``encode_multiclass_labels_for_xgboost(y_train, task)`` or a ``sklearn.LabelEncoder``
    fitted on ``DELAY_CATEGORY_ORDER`` / ``ROOT_CAUSE_CLASSES`` respectively.
    LightGBM does not require this encoding.

    Args:
        study: Completed Optuna study returned by ``run_study()``.  Must
            contain at least one successfully completed trial.
        model_type: Framework identifier — ``"xgboost"`` or ``"lightgbm"``.
            Must match the ``model_type`` argument used in
            ``build_optuna_objective()`` for the same study.
        task: One of the four TARGET_COLS.  Determines estimator class
            (classifier vs regressor) and class-imbalance strategy.  Must
            match the ``task`` argument used in ``build_optuna_objective()``.
        scale_pos_weight: Negative-to-positive class ratio for binary
            classifiers.  Pass the same value used in
            ``build_optuna_objective()`` so that the champion estimator
            uses identical class-weight settings during final training.
            Default: ``DEFAULT_SCALE_POS_WEIGHT`` (≈ 1.703).
        random_state: Seed for the returned estimator.  Default:
            ``RANDOM_STATE`` (42).

    Returns:
        Unfitted sklearn-compatible estimator with ``study.best_params``
        and all production fixed params applied.  The type is determined by
        ``(model_type, task)``:

        * ``("xgboost", "is_delayed")`` → :class:`~xgboost.XGBClassifier`
        * ``("xgboost", "delay_minutes")`` → :class:`~xgboost.XGBRegressor`
        * ``("xgboost", "delay_category"/"delay_root_cause")`` →
          :class:`~xgboost.XGBClassifier` with ``objective='multi:softprob'``
        * ``("lightgbm", "is_delayed")`` → :class:`~lightgbm.LGBMClassifier`
        * ``("lightgbm", "delay_minutes")`` → :class:`~lightgbm.LGBMRegressor`
        * ``("lightgbm", "delay_category"/"delay_root_cause")`` →
          :class:`~lightgbm.LGBMClassifier` with ``objective='multiclass'``

    Raises:
        ValueError: If ``model_type`` is not ``"xgboost"`` or ``"lightgbm"``.
        ValueError: If ``task`` is not in TARGET_COLS.
        ValueError: If the study has no successfully completed trials.

    Examples:
        >>> study = run_study(
        ...     build_optuna_objective(
        ...         X_train, y_train,
        ...         model_type="lightgbm",
        ...         task="is_delayed",
        ...     ),
        ...     n_trials=100,
        ... )
        >>> estimator = best_params_to_model(
        ...     study,
        ...     model_type="lightgbm",
        ...     task="is_delayed",
        ... )
        >>> from sklearn.pipeline import Pipeline
        >>> from mpc_ml.features.pipeline import build_pipeline
        >>> from mpc_ml.tracking.mlflow_utils import (
        ...     start_run, get_experiment_name,
        ...     log_standard_params, log_model_with_signature,
        ... )
        >>>
        >>> full_pipeline = Pipeline([
        ...     ("preprocessor", build_pipeline()),
        ...     ("model", estimator),
        ... ])
        >>> full_pipeline.fit(X_train, y_train)
        >>> preprocessor = full_pipeline.named_steps["preprocessor"]
        >>> X_transformed = preprocessor.transform(X_train)
        >>>
        >>> with start_run(
        ...     get_experiment_name("is_delayed"),
        ...     "lgbm_champion",
        ...     tags={"phase": "champion", "model_type": "LightGBM"},
        ... ) as run:
        ...     log_standard_params(study.best_params)
        ...     log_model_with_signature(
        ...         full_pipeline, X_train, X_transformed,
        ...         registered_model_name="delay_classifier",
        ...     )
        ...     champion_run_id = run.info.run_id
    """
    _validate_model_type(model_type)
    _validate_task(task)
    _validate_scale_pos_weight(scale_pos_weight)

    n_complete = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    if n_complete == 0:
        raise ValueError(
            "Study has no successfully completed trials. "
            "Run study.optimize(objective, n_trials=N) with a sufficient "
            "n_trials value so that at least one trial completes without being "
            "pruned or raising an exception."
        )

    best_params = study.best_params.copy()

    logger.info(
        "Reconstructing champion estimator: model_type=%r task=%r "
        "trial=%d score=%.4f params=%s",
        model_type,
        task,
        study.best_trial.number,
        study.best_value,
        best_params,
    )

    return _build_estimator(best_params, model_type, task, scale_pos_weight, random_state)
