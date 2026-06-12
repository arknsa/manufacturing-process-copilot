"""
ml/src/mpc_ml/tracking/mlflow_utils.py
=========================================
Centralised MLflow logging utilities for the Manufacturing Process Copilot
ML layer.

Every interaction with MLflow from within the ``mpc_ml`` package flows through
this module.  No other module in ``mpc_ml`` calls ``mlflow.*`` directly.  This
centralisation provides four guarantees:

1. **Consistent artifact schema.**  Every training run produces identical
   artifact names, enabling the backend serving layer to load artifacts by
   canonical path without guessing.

2. **Reliable SHAP background generation.**  ``shap_background_sample.npy``
   is always produced alongside the champion model, enabling
   ``DelayExplainer`` in the backend to initialise a stable
   ``shap.TreeExplainer`` at service startup.

3. **Auditability.**  All model promotion decisions pass through one code
   path with structured log messages.

4. **Testability.**  A test suite can configure a local file-based tracking
   URI via ``mlflow.set_tracking_uri("file:///tmp/mlruns")`` and all calls
   in this module are affected uniformly.

Canonical artifact paths
-------------------------
The backend serving layer (``backend/app/services/ml/``) loads these by
name.  They must never be renamed without a matching change in
``registry.py`` and ``explainability.py``.

==================================  ==========================================
Artifact name                       Consumer
==================================  ==========================================
``pipeline``                        ``registry.py`` — primary serving model
``preprocessing_pipeline``          ``explainability.py`` — SHAP transforms
``feature_names.json``              ``explainability.py`` — SHAP name mapping
``cold_start_defaults.json``        Audit / debugging (embedded in pkl)
``shap_background/shap_background_sample.npy``  ``explainability.py`` — TreeExplainer init
``classification_report.txt``       MLflow UI + offline analysis
``confusion_matrix.png``            MLflow UI
``calibration_curve.png``           MLflow UI
``shap_beeswarm.png``               MLflow UI + portfolio notebook
``feature_importance.csv``          MLflow UI + portfolio notebook
==================================  ==========================================

Experiment naming convention (Doc 04 §Day 10)
----------------------------------------------
======================  =============================
TARGET_COL              Experiment path
======================  =============================
``is_delayed``          ``mpc/delay_prediction``
``delay_minutes``       ``mpc/delay_regression``
``delay_category``      ``mpc/delay_category``
``delay_root_cause``    ``mpc/root_cause``
======================  =============================

Architecture references
-----------------------
* Doc 04 §Day 10 — Experiment structure, mandatory artifact list
* Doc 05 — ``tracking/mlflow_utils.py`` function contract
* Doc 08 §11 — MLflow serialisation strategy (all artifact specifications)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from mpc_ml.features.constants import TARGET_COLS
from mpc_ml.features.pipeline import get_feature_names

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------
__all__: List[str] = [
    "get_experiment_name",
    "start_run",
    "log_pipeline",
    "log_model_with_signature",
    "log_standard_params",
    "log_standard_metrics",
    "log_standard_artifacts",
    "promote_to_production",
]

# ---------------------------------------------------------------------------
# Canonical experiment paths (Doc 04 §Day 10)
# Keyed by TARGET_COL name; values are the MLflow experiment paths.
# ---------------------------------------------------------------------------
_EXPERIMENT_NAMES: Dict[str, str] = {
    "is_delayed":       "mpc/delay_prediction",
    "delay_minutes":    "mpc/delay_regression",
    "delay_category":   "mpc/delay_category",
    "delay_root_cause": "mpc/root_cause",
}

# ---------------------------------------------------------------------------
# Canonical artifact path strings
# These names are the on-disk identifiers inside the MLflow run artifact
# directory.  Any rename requires a matching change in the backend serving layer.
# ---------------------------------------------------------------------------
_ARTIFACT_PATH_PIPELINE:        str = "pipeline"
_ARTIFACT_PATH_PREPROCESSOR:    str = "preprocessing_pipeline"
_ARTIFACT_FEATURE_NAMES:        str = "feature_names.json"
_ARTIFACT_COLD_START:           str = "cold_start_defaults.json"
_ARTIFACT_SHAP_BG:              str = "shap_background_sample.npy"
# F-03: subdirectory under which shap_background_sample.npy is stored.
# Canonical load path: runs:/{run_id}/shap_background/shap_background_sample.npy
_ARTIFACT_SHAP_BG_DIR:          str = "shap_background"
_ARTIFACT_CLASS_REPORT:         str = "classification_report.txt"
_ARTIFACT_CONFUSION_MATRIX:     str = "confusion_matrix.png"
_ARTIFACT_CALIBRATION_CURVE:    str = "calibration_curve.png"
_ARTIFACT_SHAP_BEESWARM:        str = "shap_beeswarm.png"
_ARTIFACT_FEATURE_IMPORTANCE:   str = "feature_importance.csv"


# ===========================================================================
# Private helpers
# ===========================================================================

def _assert_active_run() -> None:
    """Raise ``RuntimeError`` when no MLflow run is active.

    All public logging functions call this guard first so that callers receive
    an actionable error message rather than an opaque ``MlflowException`` from
    deep inside the MLflow client code.

    Raises
    ------
    RuntimeError
        If ``mlflow.active_run()`` returns ``None``.
    """
    if mlflow.active_run() is None:
        raise RuntimeError(
            "No active MLflow run. Wrap all logging calls inside the "
            "'start_run()' context manager:\n\n"
            "    with start_run(experiment_name, run_name) as run:\n"
            "        log_standard_metrics(metrics)\n"
            "        log_pipeline(full_pipeline)\n"
        )


def _to_numpy(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Coerce a DataFrame or array to a 2-D float64 numpy array.

    The preprocessing pipeline is configured with
    ``set_output(transform='pandas')`` (sklearn ≥ 1.2), so
    ``preprocessor.transform()`` returns a ``pd.DataFrame``.  SHAP and numpy
    file operations require a raw array.  This helper normalises the type once.

    Parameters
    ----------
    X:
        Either a ``np.ndarray`` or a ``pd.DataFrame`` with numeric values.

    Returns
    -------
    np.ndarray
        2-D float64 array.  Shape is preserved; no copy is made if the input
        is already a float64 array.

    Raises
    ------
    TypeError
        If ``X`` is neither an ndarray nor a DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64)
    if isinstance(X, np.ndarray):
        return X.astype(np.float64, copy=False)
    raise TypeError(
        f"Expected np.ndarray or pd.DataFrame, got {type(X).__name__!r}. "
        "Pass the output of "
        "full_pipeline.named_steps['preprocessor'].transform(X_train)."
    )


def _sample_background(
    X_transformed: np.ndarray,
    n_samples: int,
    random_state: int,
) -> np.ndarray:
    """Draw a reproducible stratified random sample from the transformed matrix.

    Parameters
    ----------
    X_transformed:
        2-D float64 array with shape ``(n_train, 41)`` — the full preprocessor
        output for the training split.
    n_samples:
        Target number of rows.  Silently clamped to ``len(X_transformed)``
        when the training set has fewer than ``n_samples`` rows.
    random_state:
        Seed for ``numpy.random.default_rng``.  Fixed at 42 by default so
        the same background sample is generated on identical inputs across
        re-runs.

    Returns
    -------
    np.ndarray
        Array with shape ``(min(n_samples, n_train), 41)``.
    """
    n = min(n_samples, len(X_transformed))
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_transformed), size=n, replace=False)
    return X_transformed[indices]


# ===========================================================================
# Public API
# ===========================================================================

def get_experiment_name(task_name: str) -> str:
    """Return the canonical MLflow experiment path for a TARGET_COL task.

    The canonical paths are defined in Doc 04 §Day 10.  All training scripts
    must use this function rather than hardcoding experiment paths so that runs
    are consistently grouped in the MLflow UI and queryable via the REST API.

    Parameters
    ----------
    task_name:
        One of the four values in ``mpc_ml.features.constants.TARGET_COLS``:
        ``"is_delayed"``, ``"delay_minutes"``, ``"delay_category"``, or
        ``"delay_root_cause"``.

    Returns
    -------
    str
        Canonical MLflow experiment path, e.g. ``"mpc/delay_prediction"``.

    Raises
    ------
    ValueError
        If ``task_name`` is not a recognised TARGET_COL name.

    Examples
    --------
    >>> get_experiment_name("is_delayed")
    'mpc/delay_prediction'
    >>> get_experiment_name("delay_root_cause")
    'mpc/root_cause'
    >>> get_experiment_name("unknown_task")
    ValueError: Unknown task_name 'unknown_task'. Must be one of ...
    """
    if task_name not in _EXPERIMENT_NAMES:
        raise ValueError(
            f"Unknown task_name {task_name!r}. "
            f"Must be one of {list(_EXPERIMENT_NAMES)}. "
            f"Valid task names are the TARGET_COLS defined in "
            f"mpc_ml.features.constants: {TARGET_COLS}."
        )
    return _EXPERIMENT_NAMES[task_name]


@contextmanager
def start_run(
    experiment_name: str,
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager: create (if needed) and set the experiment, then start a run.

    All MLflow logging calls executed within the ``with`` block are
    automatically associated with this run.  The run is ended when the block
    exits, whether normally or via an exception.

    Experiments are created automatically on first use via
    ``mlflow.set_experiment()``.  This is idempotent — calling it when the
    experiment already exists is a no-op.

    Parameters
    ----------
    experiment_name:
        MLflow experiment path, e.g. ``"mpc/delay_prediction"``.  Use
        ``get_experiment_name(task_name)`` to obtain canonical paths.
        Paths are created automatically if they do not exist.
    run_name:
        Human-readable run identifier shown in the MLflow UI.  Recommended
        format: ``"<algorithm>_<phase>"`` — for example, ``"xgb_baseline"``,
        ``"lgbm_optuna_trial_042"``, ``"xgb_champion"``.
    tags:
        Optional dict of ``{str: str}`` key-value pairs logged as MLflow run
        tags.  Tags are queryable in the MLflow experiments API.  Recommended
        keys: ``"model_type"`` (``"XGBoost"``/``"LightGBM"``), ``"task"``
        (TARGET_COL name), ``"phase"`` (``"baseline"``/``"tuning"``/``"champion"``).

    Yields
    ------
    mlflow.ActiveRun
        The active run context object.  Its ``info.run_id`` attribute is the
        unique identifier used for artifact loading and model promotion.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the MLflow tracking server (configured via
        ``mlflow.set_tracking_uri()``) is unreachable.

    Examples
    --------
    >>> with start_run(
    ...     get_experiment_name("is_delayed"),
    ...     "xgb_baseline",
    ...     tags={"model_type": "XGBoost", "phase": "baseline"},
    ... ) as run:
    ...     log_standard_metrics({"val_roc_auc": 0.84, "val_f1": 0.71})
    ...     log_pipeline(full_pipeline)
    ...     saved_run_id = run.info.run_id
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        logger.info(
            "MLflow run started: experiment=%r run_name=%r run_id=%s",
            experiment_name,
            run_name,
            run.info.run_id,
        )
        try:
            yield run
        except Exception:
            logger.exception(
                "MLflow run %s failed (experiment=%r run_name=%r).",
                run.info.run_id,
                experiment_name,
                run_name,
            )
            raise
        else:
            logger.info(
                "MLflow run %s finished successfully (experiment=%r run_name=%r).",
                run.info.run_id,
                experiment_name,
                run_name,
            )


def log_pipeline(
    full_pipeline: Pipeline,
    *,
    is_champion: bool = False,
) -> None:
    """Log the full pipeline and the preprocessor-only as sklearn MLflow models.

    Logs two separate versioned sklearn model artifacts to the active run:

    ``pipeline``
        The complete ``Pipeline([('preprocessor', ...), ('model', ...)])``.
        This is the primary serving artifact loaded by
        ``backend/app/services/ml/registry.py``.  Calling
        ``pipeline.predict_proba(X_raw)`` on a 37-column raw input DataFrame
        is all the serving layer needs for inference.

    ``preprocessing_pipeline``
        Only the 3-step preprocessing pipeline returned by
        ``build_pipeline()`` (extracted as
        ``full_pipeline.named_steps['preprocessor']``).  Used by
        ``backend/app/services/ml/explainability.py`` to transform raw inputs
        into the 41-column model input space before SHAP value computation.

    Unlike ``log_model_with_signature``, this function does not infer an
    MLflow model signature and does not generate the auxiliary artifacts
    (``feature_names.json``, ``shap_background_sample.npy``).  **This
    function is intended for quick baseline runs only.**  For champion models
    that will be promoted to Production, use ``log_model_with_signature()``
    which logs the complete artifact set required by the serving layer.

    Parameters
    ----------
    full_pipeline:
        A fitted sklearn ``Pipeline`` with exactly two named steps:
        ``'preprocessor'`` (the 3-step pipeline from ``build_pipeline()``)
        and ``'model'`` (any sklearn-compatible estimator).  Must be fitted
        before calling this function.
    is_champion:
        Safety guard.  Set to ``True`` to explicitly signal that the caller
        intends to log a champion model — this always raises ``RuntimeError``
        with a message directing the caller to ``log_model_with_signature()``.
        Default: ``False``.

        This parameter exists to prevent the silent failure mode where a
        training script calls ``log_pipeline()`` instead of
        ``log_model_with_signature()`` for the promoted model, causing the
        backend serving layer to fail at startup when it cannot find the
        required ``shap_background/shap_background_sample.npy`` artifact.

    Raises
    ------
    RuntimeError
        If called outside a ``start_run()`` context (no active MLflow run).
    RuntimeError
        If ``is_champion=True``.  Champion models must be logged with
        ``log_model_with_signature()`` so that all serving artifacts
        (``feature_names.json``, ``shap_background/shap_background_sample.npy``,
        ``cold_start_defaults.json``) are present in the MLflow run.
    KeyError
        If ``full_pipeline`` does not have a ``'preprocessor'`` named step,
        indicating the pipeline was not assembled with the expected pattern:
        ``Pipeline([('preprocessor', build_pipeline()), ('model', estimator)])``.

    Examples
    --------
    >>> full_pipeline = Pipeline([
    ...     ("preprocessor", build_pipeline()),
    ...     ("model", XGBClassifier()),
    ... ])
    >>> full_pipeline.fit(X_train, y_train)
    >>> with start_run(get_experiment_name("is_delayed"), "xgb_baseline"):
    ...     log_pipeline(full_pipeline)          # baseline run — OK
    ...     log_standard_metrics({"val_roc_auc": 0.84})
    >>>
    >>> # For the champion model, use log_model_with_signature() instead:
    >>> # with start_run(..., "xgb_champion"):
    >>> #     log_model_with_signature(full_pipeline, X_train, X_transformed, ...)
    """
    # F-04: hard stop for callers who accidentally use this function for the
    # champion model instead of log_model_with_signature().
    if is_champion:
        raise RuntimeError(
            "log_pipeline() must not be used for champion models. "
            "Call log_model_with_signature() instead, which logs the complete "
            "serving artifact set required by the backend:\n"
            "  - feature_names.json\n"
            "  - shap_background/shap_background_sample.npy\n"
            "  - cold_start_defaults.json\n"
            "  - MLflow model signature\n\n"
            "log_pipeline() is for quick baseline runs only.  Missing these "
            "artifacts will cause the backend serving layer to fail at startup."
        )

    _assert_active_run()

    if "preprocessor" not in full_pipeline.named_steps:
        raise KeyError(
            "full_pipeline does not have a 'preprocessor' named step. "
            "Assemble the full pipeline as: "
            "Pipeline([('preprocessor', build_pipeline()), ('model', estimator)]). "
            f"Actual named steps: {list(full_pipeline.named_steps)}."
        )

    preprocessor = full_pipeline.named_steps["preprocessor"]

    mlflow.sklearn.log_model(
        sk_model=preprocessor,
        artifact_path=_ARTIFACT_PATH_PREPROCESSOR,
    )
    mlflow.sklearn.log_model(
        sk_model=full_pipeline,
        artifact_path=_ARTIFACT_PATH_PIPELINE,
    )

    logger.debug(
        "Logged sklearn artifacts: %r and %r",
        _ARTIFACT_PATH_PIPELINE,
        _ARTIFACT_PATH_PREPROCESSOR,
    )


def log_model_with_signature(
    full_pipeline: Pipeline,
    X_train_raw: pd.DataFrame,
    X_train_transformed: Union[np.ndarray, pd.DataFrame],
    *,
    registered_model_name: Optional[str] = None,
    n_background_samples: int = 200,
    background_random_state: int = 42,
) -> None:
    """Log the champion model with MLflow signature and all required serving artifacts.

    This is the primary logging function for a tuned champion model.  It logs
    all five artifacts required by the backend serving layer (Doc 08 §11):

    ``pipeline``
        The complete ``Pipeline([preprocessor, model])`` logged with an
        inferred MLflow model signature.  The signature's input schema is
        derived from the 37-column raw training DataFrame; the output schema
        from the model's ``predict_proba`` (or ``predict`` for regressors).
        Optionally registered in the MLflow Model Registry.

    ``preprocessing_pipeline``
        The preprocessing pipeline only, logged without a signature.  Used by
        ``explainability.py`` to transform raw 37-column inputs to the
        41-column model input space before SHAP computation.

    ``feature_names.json``
        A JSON dict ``{"feature_names": [...]}`` containing the 41-element
        list of ColumnTransformer output names in their fixed output order.
        Associates SHAP value positions [0..40] with human-readable feature
        names.  Generated from ``mpc_ml.features.pipeline.get_feature_names()``.

    ``cold_start_defaults.json``
        A JSON dict ``{rolling_feature_name: population_mean}`` extracted from
        the fitted ``ColumnSelector.cold_start_defaults_``.  Documents the
        training-set population means used for cold-start NaN filling at
        inference time.  Provided for audit and debugging; the values are
        already embedded in the serialised preprocessing pipeline pickle.

    ``shap_background_sample.npy``
        A numpy array of shape ``(min(n_background_samples, n_train), 41)``
        containing a reproducible random sample of the transformed training
        data.  Loaded by ``explainability.py`` at startup to initialise
        ``shap.TreeExplainer(model, data=background)`` with a stable expected-
        value baseline that matches the training distribution.

    Parameters
    ----------
    full_pipeline:
        A *fitted* sklearn ``Pipeline`` with named steps ``'preprocessor'``
        and ``'model'``.  Must be fitted before calling this function —
        ``cold_start_defaults_`` is read from the fitted ColumnSelector state.
    X_train_raw:
        The raw 37-column training DataFrame passed to ``full_pipeline.fit()``.
        Used to infer the MLflow model signature (input schema).  A 5-row
        sample is used internally; the full array is never stored as an
        artifact.  Must not contain any ``TARGET_COLS``.
    X_train_transformed:
        The 41-column output of
        ``full_pipeline.named_steps['preprocessor'].transform(X_train_raw)``.
        Accepted as either a numpy array or a pandas DataFrame (the pipeline
        uses ``set_output(transform='pandas')``).  Used exclusively to generate
        ``shap_background_sample.npy``.
    registered_model_name:
        If provided, registers the logged model in the MLflow Model Registry
        under this name.  This is required before ``promote_to_production()``
        can be called.  Recommended values: ``"delay_classifier"``,
        ``"delay_regressor"``, ``"delay_category_classifier"``,
        ``"root_cause_classifier"``.  If ``None``, the model is logged as an
        artifact only (not registered).
    n_background_samples:
        Target number of rows to sample from ``X_train_transformed`` for the
        SHAP background array.  Clamped to ``len(X_train_transformed)`` when
        the training set is smaller.  Default: 200 (per Doc 08 §11).
    background_random_state:
        Random seed for ``numpy.random.default_rng`` used in the background
        sample draw.  Fixed at 42 to ensure reproducibility across re-runs on
        identical data.

    Raises
    ------
    RuntimeError
        If called outside a ``start_run()`` context (no active MLflow run).
    RuntimeError
        If ``cold_start_defaults_`` cannot be extracted from the fitted
        ColumnSelector, indicating that ``full_pipeline.fit()`` has not been
        called.
    KeyError
        If ``full_pipeline`` does not have a ``'preprocessor'`` or the
        preprocessor does not have a ``'column_selector'`` step.
    TypeError
        If ``X_train_transformed`` is neither a numpy array nor a DataFrame.
    ValueError
        If ``X_train_raw`` contains columns listed in ``TARGET_COLS``,
        indicating target leakage in the input passed to signature inference.

    Examples
    --------
    >>> full_pipeline = Pipeline([
    ...     ("preprocessor", build_pipeline()),
    ...     ("model", XGBClassifier(**best_params)),
    ... ])
    >>> full_pipeline.fit(X_train, y_train)
    >>> preprocessor = full_pipeline.named_steps["preprocessor"]
    >>> X_transformed = preprocessor.transform(X_train)
    >>>
    >>> with start_run(
    ...     get_experiment_name("is_delayed"),
    ...     "xgb_champion",
    ...     tags={"phase": "champion", "model_type": "XGBoost"},
    ... ) as run:
    ...     log_model_with_signature(
    ...         full_pipeline,
    ...         X_train,
    ...         X_transformed,
    ...         registered_model_name="delay_classifier",
    ...     )
    ...     champion_run_id = run.info.run_id
    """
    _assert_active_run()

    if "preprocessor" not in full_pipeline.named_steps:
        raise KeyError(
            "full_pipeline does not have a 'preprocessor' named step. "
            "Assemble as: "
            "Pipeline([('preprocessor', build_pipeline()), ('model', estimator)]). "
            f"Actual named steps: {list(full_pipeline.named_steps)}."
        )

    # Guard against target leakage in signature input
    leaking_cols = [c for c in X_train_raw.columns if c in TARGET_COLS]
    if leaking_cols:
        raise ValueError(
            f"X_train_raw contains target columns: {leaking_cols}. "
            "Pass only the feature DataFrame — drop TARGET_COLS before "
            "calling log_model_with_signature()."
        )

    preprocessor = full_pipeline.named_steps["preprocessor"]

    # ------------------------------------------------------------------
    # 1. Infer MLflow model signature from a small sample of training data
    # ------------------------------------------------------------------
    X_sample = X_train_raw.iloc[:5]
    try:
        predictions_sample = full_pipeline.predict_proba(X_sample)
    except AttributeError:
        # Regressor: XGBRegressor / LGBMRegressor have no predict_proba
        predictions_sample = full_pipeline.predict(X_sample)

    signature = infer_signature(X_sample, predictions_sample)

    # ------------------------------------------------------------------
    # 2. Log full pipeline with signature (optionally register)
    # ------------------------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=full_pipeline,
        artifact_path=_ARTIFACT_PATH_PIPELINE,
        signature=signature,
        registered_model_name=registered_model_name,
    )

    # ------------------------------------------------------------------
    # 3. Log preprocessor only (no signature — SHAP access pattern)
    # ------------------------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=preprocessor,
        artifact_path=_ARTIFACT_PATH_PREPROCESSOR,
    )

    # ------------------------------------------------------------------
    # 4. Log feature_names.json
    # ------------------------------------------------------------------
    mlflow.log_dict(
        {"feature_names": get_feature_names()},
        artifact_file=_ARTIFACT_FEATURE_NAMES,
    )

    # ------------------------------------------------------------------
    # 5. Log cold_start_defaults.json
    # ------------------------------------------------------------------
    try:
        column_selector = preprocessor.named_steps["column_selector"]
        cold_start_defaults: Dict[str, float] = {
            k: float(v) for k, v in column_selector.cold_start_defaults_.items()
        }
    except AttributeError as exc:
        raise RuntimeError(
            "Cannot read cold_start_defaults_ from the ColumnSelector step. "
            "Ensure full_pipeline.fit(X_train, y_train) has been called "
            "before logging."
        ) from exc
    except KeyError as exc:
        raise KeyError(
            "Preprocessor does not have a 'column_selector' step. "
            "Verify that build_pipeline() was used to construct the preprocessor."
        ) from exc

    mlflow.log_dict(cold_start_defaults, artifact_file=_ARTIFACT_COLD_START)

    # ------------------------------------------------------------------
    # 6. Generate and log shap_background_sample.npy
    # ------------------------------------------------------------------
    X_transformed_np = _to_numpy(X_train_transformed)

    # F-07: validate column count before sampling so the error surfaces at
    # training time rather than at serve time (when TreeExplainer would fail
    # with a shape mismatch against the 41-column model input space).
    expected_n_features = len(get_feature_names())
    actual_n_features = X_transformed_np.shape[1]
    if actual_n_features != expected_n_features:
        raise ValueError(
            f"X_train_transformed has {actual_n_features} columns but "
            f"the pipeline produces {expected_n_features} output features. "
            "Pass the output of "
            "full_pipeline.named_steps['preprocessor'].transform(X_train_raw) "
            f"(shape: (n_train, {expected_n_features})), not the raw "
            f"X_train_raw DataFrame (shape: (n_train, {len(get_feature_names()) - 4}))."
        )

    background = _sample_background(
        X_transformed_np, n_background_samples, background_random_state
    )

    # F-03: log to a named subdirectory so the canonical load path is
    # runs:/{run_id}/shap_background/shap_background_sample.npy, consistent
    # with all other model artifacts that land in named subdirectories.
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_path = Path(tmpdir) / _ARTIFACT_SHAP_BG
        np.save(str(bg_path), background)
        mlflow.log_artifact(str(bg_path), artifact_path=_ARTIFACT_SHAP_BG_DIR)

    run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
    logger.info(
        "Champion model artifacts logged: run_id=%s background_shape=%s "
        "registered_model_name=%r",
        run_id,
        background.shape,
        registered_model_name,
    )


def log_standard_metrics(
    metrics: Dict[str, float],
    *,
    step: Optional[int] = None,
) -> None:
    """Log a flat dict of float metrics to the active MLflow run.

    This is the single call site for all metric logging in the modeling layer.
    Callers pass the ``EvaluationReport.metrics`` dict directly — no key
    manipulation is needed.  All keys are logged as-is, making them queryable
    in the MLflow experiments UI and via ``mlflow.search_runs()``.

    Parameters
    ----------
    metrics:
        Dict mapping metric name to float value.  All values must be finite —
        MLflow rejects NaN or infinite values with an opaque server-side error.
        Common keys (binary task): ``"val_roc_auc"``, ``"val_f1"``,
        ``"val_precision"``, ``"val_recall"``, ``"val_brier_score"``,
        ``"train_roc_auc"``.  Regression keys: ``"val_mae"``, ``"val_rmse"``,
        ``"val_r2"``.
    step:
        Optional integer step for time-series metrics, e.g. the Optuna trial
        index when logging intermediate cross-validation values from inside
        ``OptunaObjective.__call__()``.  When ``None``, the metric is logged
        without a step dimension.

    Raises
    ------
    RuntimeError
        If called outside a ``start_run()`` context.
    ValueError
        If any metric value is NaN or infinite.  Validated before the MLflow
        call so the error message names the offending keys.

    Examples
    --------
    >>> with start_run(get_experiment_name("is_delayed"), "xgb_run_01"):
    ...     log_standard_metrics({
    ...         "val_roc_auc": 0.872,
    ...         "val_f1": 0.714,
    ...         "val_precision_at_80pct_recall": 0.631,
    ...         "val_brier_score": 0.112,
    ...     })
    >>>
    >>> # Inside OptunaObjective — report per-trial intermediate value:
    >>> with start_run(get_experiment_name("is_delayed"), "optuna_trial_042"):
    ...     for fold, fold_auc in enumerate(fold_aucs):
    ...         log_standard_metrics({"cv_roc_auc": fold_auc}, step=fold)
    """
    _assert_active_run()

    bad_keys = [k for k, v in metrics.items() if not np.isfinite(v)]
    if bad_keys:
        raise ValueError(
            f"Non-finite metric values detected: {bad_keys}. "
            "Verify that the model produced valid predictions before evaluation. "
            "Common causes: untrained model, all-zero predictions, empty split."
        )

    mlflow.log_metrics(metrics, step=step)
    logger.debug("Logged %d metric(s): %s", len(metrics), sorted(metrics))


def log_standard_params(params: Dict[str, Union[str, int, float, bool]]) -> None:
    """Log a flat dict of hyperparameters to the active MLflow run.

    This is the single call site for all hyperparameter logging in the
    modeling layer.  Centralising param logging here enforces:

    * **Consistent serialisation** — MLflow params are stored as strings;
      non-serialisable types (numpy arrays, sklearn objects, nested dicts)
      are detected and rejected before the MLflow call produces an opaque
      server-side error.
    * **Auditability** — all params flow through one code path with a
      structured log message, matching the module-level contract.

    Parameters
    ----------
    params:
        Dict mapping hyperparameter name to its value.  Supported value
        types are ``str``, ``int``, ``float``, and ``bool``; all other
        types raise ``TypeError``.

        Typical callers pass ``model.get_params()`` for sklearn-compatible
        estimators (XGBoost, LightGBM, sklearn):

        .. code-block:: python

            log_standard_params(model.get_params())

        For Optuna best-trial params, pass the trial's
        ``best_params`` dict directly:

        .. code-block:: python

            log_standard_params(study.best_params)

    Raises
    ------
    RuntimeError
        If called outside a ``start_run()`` context.
    TypeError
        If any param value is not one of ``str``, ``int``, ``float``, or
        ``bool``.  The error message lists all offending keys and their
        actual types so the caller can convert or exclude them.
    ValueError
        If ``params`` is empty.  An empty param dict indicates a likely
        caller error (e.g., calling ``model.get_params()`` on an
        unconfigured estimator); logging nothing silently would hide this.

    Examples
    --------
    >>> xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
    >>> with start_run(get_experiment_name("is_delayed"), "xgb_baseline"):
    ...     log_standard_params(xgb.get_params())
    ...     log_standard_metrics({"val_roc_auc": 0.84})
    ...     log_pipeline(full_pipeline)
    >>>
    >>> # Optuna champion — pass best_params directly:
    >>> with start_run(get_experiment_name("is_delayed"), "xgb_champion"):
    ...     log_standard_params(study.best_params)
    ...     log_model_with_signature(full_pipeline, X_train, X_transformed,
    ...                              registered_model_name="delay_classifier")
    """
    _assert_active_run()

    if not params:
        raise ValueError(
            "params dict is empty.  Pass model.get_params() or study.best_params. "
            "Logging an empty param dict is almost always a caller error."
        )

    _ALLOWED_TYPES = (str, int, float, bool)
    bad_keys = {
        k: type(v).__name__
        for k, v in params.items()
        if not isinstance(v, _ALLOWED_TYPES)
    }
    if bad_keys:
        raise TypeError(
            f"Non-serialisable param value(s) detected: "
            f"{', '.join(f'{k!r}: {t}' for k, t in sorted(bad_keys.items()))}. "
            "MLflow params must be str, int, float, or bool. "
            "Convert arrays/objects to scalar values before calling "
            "log_standard_params(), or exclude them with: "
            "{k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}"
        )

    mlflow.log_params(params)
    logger.debug("Logged %d param(s): %s", len(params), sorted(params))


def log_standard_artifacts(
    *,
    classification_report: Optional[str] = None,
    confusion_matrix_path: Optional[Path] = None,
    calibration_curve_path: Optional[Path] = None,
    shap_beeswarm_path: Optional[Path] = None,
    feature_importance_csv_path: Optional[Path] = None,
) -> None:
    """Log the standard set of evaluation artifacts defined in Doc 04.

    All five parameters are optional.  Passing ``None`` for any artifact
    silently skips that artifact, allowing callers to log the subset relevant
    to their task type — for example, classification reports and confusion
    matrices do not apply to regression; SHAP beeswarms are not generated in
    quick baseline runs.

    Files are copied to a temporary directory with their canonical names
    before logging, ensuring artifact names in MLflow are always predictable
    regardless of what temporary filenames the caller used.

    Parameters
    ----------
    classification_report:
        String content of ``sklearn.metrics.classification_report()``.
        Written to a temporary file and logged as
        ``classification_report.txt``.  Not applicable to regression tasks.
    confusion_matrix_path:
        Path to the confusion matrix PNG file produced by
        ``evaluation.confusion_matrix_annotated()``.  Logged as
        ``confusion_matrix.png``.
    calibration_curve_path:
        Path to the reliability diagram PNG file produced by
        ``evaluation.calibration_report()``.  Logged as
        ``calibration_curve.png``.
    shap_beeswarm_path:
        Path to the SHAP beeswarm PNG file produced by
        ``evaluation.shap_beeswarm()``.  Logged as ``shap_beeswarm.png``.
    feature_importance_csv_path:
        Path to the feature importance CSV file produced by
        ``evaluation.feature_importance_report()``.  Logged as
        ``feature_importance.csv``.

    Raises
    ------
    RuntimeError
        If called outside a ``start_run()`` context.
    FileNotFoundError
        If any non-``None`` path argument points to a file that does not exist.
        All paths are validated before any artifact is uploaded so that
        partial-artifact runs do not occur.

    Examples
    --------
    >>> with start_run(get_experiment_name("is_delayed"), "xgb_champion"):
    ...     # ... training, tuning, evaluation ...
    ...     log_standard_artifacts(
    ...         classification_report=sklearn_report_str,
    ...         confusion_matrix_path=Path("/tmp/cm.png"),
    ...         shap_beeswarm_path=Path("/tmp/shap.png"),
    ...         feature_importance_csv_path=Path("/tmp/fi.csv"),
    ...     )
    """
    _assert_active_run()

    # Validate all provided paths before starting to upload
    # (prevents partial-artifact states from partial failures mid-upload)
    path_args: Dict[str, Optional[Path]] = {
        "confusion_matrix_path":       confusion_matrix_path,
        "calibration_curve_path":      calibration_curve_path,
        "shap_beeswarm_path":          shap_beeswarm_path,
        "feature_importance_csv_path": feature_importance_csv_path,
    }
    for arg_name, path in path_args.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(
                f"{arg_name}={path!r} does not exist. "
                "Generate the artifact before calling log_standard_artifacts()."
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Classification report: string content → canonical filename
        if classification_report is not None:
            report_path = tmp / _ARTIFACT_CLASS_REPORT
            report_path.write_text(classification_report, encoding="utf-8")
            mlflow.log_artifact(str(report_path))
            logger.debug("Logged artifact: %s", _ARTIFACT_CLASS_REPORT)

        # File artifacts: copy to canonical name, then log
        file_artifacts = [
            (confusion_matrix_path,       _ARTIFACT_CONFUSION_MATRIX),
            (calibration_curve_path,      _ARTIFACT_CALIBRATION_CURVE),
            (shap_beeswarm_path,          _ARTIFACT_SHAP_BEESWARM),
            (feature_importance_csv_path, _ARTIFACT_FEATURE_IMPORTANCE),
        ]
        for source_path, canonical_name in file_artifacts:
            if source_path is not None:
                dest = tmp / canonical_name
                shutil.copy2(str(source_path), str(dest))
                mlflow.log_artifact(str(dest))
                logger.debug(
                    "Logged artifact: %s (source: %s)", canonical_name, source_path
                )


def promote_to_production(
    model_name: str,
    run_id: str,
    *,
    archive_previous: bool = True,
) -> None:
    """Transition a registered model version to the Production stage.

    Finds the model version registered from ``run_id`` in the MLflow Model
    Registry and transitions it to ``"Production"``.  By default, all existing
    Production-tagged versions of the same model are simultaneously transitioned
    to ``"Archived"``, ensuring exactly one Production version exists at any
    given time.

    This function must be called *after* the champion model has been both
    registered (via ``registered_model_name`` in ``log_model_with_signature()``)
    and evaluated on the held-out test split (via ``scripts/evaluate.py``).
    Promotion is intentionally not automatic — test-set confirmation is required.

    Promotion protocol (Doc 04 §Day 10, architecture review precondition P-4):

    1. Run Optuna study → select champion by validation primary metric.
    2. Evaluate champion on test split once via ``scripts/evaluate.py``.
    3. Confirm test metrics meet the deployment threshold.
    4. Call ``promote_to_production(model_name, champion_run_id)``.
    5. Backend ``registry.py`` detects the new Production version on its
       60-second polling cycle and hot-reloads the serving model.

    Parameters
    ----------
    model_name:
        The registered model name in the MLflow Model Registry.  Must match
        the ``registered_model_name`` parameter used in
        ``log_model_with_signature()`` during training.
        Example values: ``"delay_classifier"``, ``"delay_regressor"``,
        ``"delay_category_classifier"``, ``"root_cause_classifier"``.
    run_id:
        The MLflow run ID (``run.info.run_id`` from ``start_run()``) that
        produced the champion model version to promote.
    archive_previous:
        If ``True`` (default), all existing ``"Production"`` versions of
        ``model_name`` are simultaneously transitioned to ``"Archived"``.
        This prevents multiple Production versions from coexisting and
        ensures ``registry.py`` loads a unique Production version.
        Set to ``False`` only when deliberately running an A/B production
        deployment with two Production versions.

    Raises
    ------
    ValueError
        If no registered version of ``model_name`` is found for ``run_id``.
        This occurs when ``registered_model_name`` was omitted during training,
        or when an incorrect ``run_id`` is provided.
    mlflow.exceptions.MlflowException
        If the MLflow tracking server is unreachable, or ``model_name`` does
        not exist in the registry at all.

    Examples
    --------
    >>> # During training:
    >>> with start_run(
    ...     get_experiment_name("is_delayed"), "xgb_champion"
    ... ) as run:
    ...     log_model_with_signature(
    ...         full_pipeline, X_train, X_transformed,
    ...         registered_model_name="delay_classifier",
    ...     )
    ...     champion_run_id = run.info.run_id
    >>>
    >>> # After test-set evaluation confirms the champion is deployable:
    >>> promote_to_production("delay_classifier", champion_run_id)
    """
    client = mlflow.MlflowClient()

    # Search for all versions registered from this run
    # MLflow search_model_versions uses SQL-like filter syntax
    all_versions = client.search_model_versions(f"run_id='{run_id}'")
    target_versions = [v for v in all_versions if v.name == model_name]

    if not target_versions:
        raise ValueError(
            f"No registered version of model {model_name!r} found for "
            f"run_id={run_id!r}. "
            "Ensure 'registered_model_name' was set in log_model_with_signature() "
            "during the training run, and that run_id matches the run that "
            "produced the model to promote."
        )

    # Select the most recently registered version for this run
    # (in practice exactly one version is registered per run)
    version_to_promote = max(target_versions, key=lambda v: int(v.version))

    client.transition_model_version_stage(
        name=model_name,
        version=version_to_promote.version,
        stage="Production",
        archive_existing_versions=archive_previous,
    )

    logger.info(
        "Model promoted to Production: name=%r version=%s run_id=%s "
        "archive_previous=%s",
        model_name,
        version_to_promote.version,
        run_id,
        archive_previous,
    )
