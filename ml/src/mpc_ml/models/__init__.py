"""
ml/src/mpc_ml/models/__init__.py
=================================
Public import facade for the ``mpc_ml.models`` modeling package.

This module re-exports every public symbol from the three modeling
sub-modules so that callers can write a single, stable import line::

    from mpc_ml.models import TASK_REGISTRY, get_baseline_models
    from mpc_ml.models import run_study, best_params_to_model
    from mpc_ml.models import evaluate_model, EvaluationReport

No logic lives here.  All implementations are in the sub-modules below.

Sub-module responsibilities
----------------------------
``baseline``
    Central task registry (``TASK_REGISTRY``) mapping each ``TARGET_COL``
    to its ``ModelConfig`` (task type, model class, objective, primary
    metric, label classes, target transform, and training-data filter).
    Also exposes multi-algorithm factories (``get_baseline_models``,
    ``get_task_models``) and target encoding helpers (``encode_target``,
    ``decode_predictions``).

``tuning``
    Optuna-based hyperparameter optimization.  Provides search space
    dataclasses (``XGBSearchSpace``, ``LGBMSearchSpace``), feature-gating
    parameters (``FeatureGatingParams``) for ``CANDIDATE_REMOVAL_FEATURES``,
    and the study runner (``build_optuna_objective``, ``run_study``,
    ``best_params_to_model``).

``evaluation``
    Task-aware metrics, diagnostic plots, and artifact generation.
    ``evaluate_model`` dispatches on ``TaskType`` to return the correct
    metric set.  Also provides ``precision_at_recall``,
    ``calibration_report``, ``confusion_matrix_annotated``,
    ``shap_beeswarm``, and ``feature_importance_report``.

Lazy loading strategy (PEP 562)
---------------------------------
Sub-modules are imported on first attribute access, not at package import
time.  This means ``import mpc_ml.models`` succeeds even when sub-modules
have not yet been implemented during incremental development.  An
``ImportError`` with a diagnostic message is raised only when a symbol
from an unimplemented sub-module is actually accessed.

Resolved symbols are cached in the module namespace so that subsequent
attribute accesses are O(1) dictionary lookups — ``__getattr__`` is not
called twice for the same name.

Serving boundary
-----------------
This package is a **training-time library only**.  It has no dependency on
FastAPI, SQLAlchemy, Celery, or any serving framework.  Production inference
is the responsibility of:

* ``backend.app.services.ml.registry`` — loads Production-stage artifacts
  from the MLflow Model Registry.
* ``backend.app.services.ml.service`` — orchestrates the two-stage
  prediction cascade (``is_delayed`` → ``delay_minutes``).
* ``backend.app.services.ml.explainability`` — initialises
  ``shap.TreeExplainer`` from the logged ``shap_background_sample.npy``
  artifact.

External dependencies
----------------------
``scikit-learn``, ``xgboost``, ``lightgbm``, ``optuna``, ``shap``,
``mlflow``, ``pandas``, ``numpy`` — all version-pinned in
``ml/pyproject.toml`` per Doc 08.

Architecture references
------------------------
* Doc 04 — Implementation roadmap (baseline benchmarking, Optuna schedule)
* Doc 05 — Repository structure (``models/``, ``tracking/`` layout)
* Doc 06 — Constants design (``TASK_REGISTRY`` design, training filters)
* Doc 08 — Pipeline design (artifact contracts, SHAP strategy)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Evaluated by type checkers and IDEs only — never executed at runtime.
    # Gives mypy / Pylance full visibility into types without triggering
    # lazy loading or failing when sub-modules are not yet implemented.
    from mpc_ml.models.baseline import (  # noqa: F401
        TASK_REGISTRY,
        ModelConfig,
        TaskType,
        decode_predictions,
        encode_target,
        get_baseline_models,
        get_task_models,
    )
    from mpc_ml.models.evaluation import (  # noqa: F401
        EvaluationReport,
        calibration_report,
        confusion_matrix_annotated,
        evaluate_model,
        feature_importance_report,
        precision_at_recall,
        shap_beeswarm,
    )
    from mpc_ml.models.tuning import (  # noqa: F401
        FeatureGatingParams,
        LGBMSearchSpace,
        OptunaObjective,
        XGBSearchSpace,
        best_params_to_model,
        build_optuna_objective,
        run_study,
    )


__all__: list[str] = [
    # --- baseline -----------------------------------------------------------
    "TASK_REGISTRY",
    "ModelConfig",
    "TaskType",
    "get_baseline_models",
    "get_task_models",
    "encode_target",
    "decode_predictions",
    # --- tuning -------------------------------------------------------------
    "XGBSearchSpace",
    "LGBMSearchSpace",
    "FeatureGatingParams",
    "OptunaObjective",
    "build_optuna_objective",
    "run_study",
    "best_params_to_model",
    # --- evaluation ---------------------------------------------------------
    "EvaluationReport",
    "evaluate_model",
    "precision_at_recall",
    "calibration_report",
    "confusion_matrix_annotated",
    "shap_beeswarm",
    "feature_importance_report",
]

# ---------------------------------------------------------------------------
# Lazy-import dispatch table
#
# Maps each public symbol name to the sub-module that owns it.  Used by
# __getattr__ to determine which file to import on first access.
# ---------------------------------------------------------------------------
_SYMBOL_TO_SUBMODULE: dict[str, str] = {
    # baseline
    "TASK_REGISTRY": "baseline",
    "ModelConfig": "baseline",
    "TaskType": "baseline",
    "get_baseline_models": "baseline",
    "get_task_models": "baseline",
    "encode_target": "baseline",
    "decode_predictions": "baseline",
    # tuning
    "XGBSearchSpace": "tuning",
    "LGBMSearchSpace": "tuning",
    "FeatureGatingParams": "tuning",
    "OptunaObjective": "tuning",
    "build_optuna_objective": "tuning",
    "run_study": "tuning",
    "best_params_to_model": "tuning",
    # evaluation
    "EvaluationReport": "evaluation",
    "evaluate_model": "evaluation",
    "precision_at_recall": "evaluation",
    "calibration_report": "evaluation",
    "confusion_matrix_annotated": "evaluation",
    "shap_beeswarm": "evaluation",
    "feature_importance_report": "evaluation",
}


def __getattr__(name: str) -> object:
    """Resolve lazy imports for symbols declared in ``__all__`` (PEP 562).

    Python calls this function when an attribute lookup on the
    ``mpc_ml.models`` module fails — i.e., when the symbol has not yet been
    cached in the module's ``__dict__``.  The function imports the owning
    sub-module, retrieves the symbol, caches it in the module namespace (so
    this function is only called once per symbol), and returns it.

    Parameters
    ----------
    name:
        The name of the attribute being accessed on ``mpc_ml.models``.

    Returns
    -------
    object
        The requested symbol from its owning sub-module.

    Raises
    ------
    AttributeError
        When ``name`` is not in ``__all__``.  Follows Python's standard
        convention: attribute access on a module that has no such attribute
        raises ``AttributeError``, not ``ImportError``.
    ImportError
        When ``name`` is in ``__all__`` but its owning sub-module has not
        been implemented yet.  The error message identifies both the symbol
        and the file that must be created, guiding incremental development.
    ImportError
        When the owning sub-module exists but does not export ``name``.
        This indicates a contract violation: the sub-module must define
        every symbol declared in this file's ``_SYMBOL_TO_SUBMODULE`` table.
    """
    if name not in _SYMBOL_TO_SUBMODULE:
        raise AttributeError(
            f"module 'mpc_ml.models' has no attribute {name!r}. "
            f"Public symbols are: {__all__}."
        )

    submodule_short = _SYMBOL_TO_SUBMODULE[name]
    qualified = f"mpc_ml.models.{submodule_short}"

    try:
        submodule = importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot import {name!r} from 'mpc_ml.models': the sub-module "
            f"'{qualified.replace('.', '/')}.py' has not been implemented. "
            f"Refer to the implementation roadmap: "
            f"baseline.py (Sprint 1), tuning.py (Sprint 2), "
            f"evaluation.py (Sprint 3)."
        ) from exc

    try:
        attr = getattr(submodule, name)
    except AttributeError as exc:
        raise ImportError(
            f"Sub-module '{qualified}' exists but does not export {name!r}. "
            f"Ensure {name!r} is defined in "
            f"'{qualified.replace('.', '/')}.py' and listed in its __all__."
        ) from exc

    # Cache in module namespace — bypasses __getattr__ on all future accesses.
    globals()[name] = attr
    return attr
