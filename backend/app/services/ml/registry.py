"""
backend/app/services/ml/registry.py
======================================
Loads champion models from MLflow at application startup and
initialises the DelayExplainer.  All models are cached in memory;
hot-reload is not implemented in this version.

Canonical artifact paths (from mpc_ml.tracking.mlflow_utils):
  pipeline                                     — full Pipeline([preprocessor, model])
  preprocessing_pipeline                       — preprocessor only (for SHAP)
  shap_background/shap_background_sample.npy  — TreeExplainer background
  feature_names.json                           — {"feature_names": [...]}
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np

from backend.app.services.ml.explainability import DelayExplainer

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    binary_run_id: str
    regression_run_id: str
    root_cause_run_id: str
    feature_count: int
    loaded_at: datetime


class MLflowModelRegistry:
    """Loads champion models from MLflow and provides a ready DelayExplainer."""

    def __init__(
        self,
        tracking_uri: str,
        binary_run_id: str,
        regression_run_id: str,
        root_cause_run_id: str,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._binary_run_id = binary_run_id
        self._regression_run_id = regression_run_id
        self._rc_run_id = root_cause_run_id
        self._explainer: Optional[DelayExplainer] = None
        self._model_info: Optional[ModelInfo] = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all champion models and initialise the DelayExplainer.

        Called once during FastAPI application lifespan startup.
        Loading takes ~5 seconds due to MLflow artifact downloads.
        """
        logger.info("Loading champion models from MLflow ...")
        tmp = tempfile.mkdtemp()

        # ── Binary classifier (41-feature pipeline, Day 7) ──────────────
        binary_full_pipe = mlflow.sklearn.load_model(
            f"runs:/{self._binary_run_id}/pipeline"
        )
        binary_preproc = mlflow.sklearn.load_model(
            f"runs:/{self._binary_run_id}/preprocessing_pipeline"
        )
        binary_model = binary_full_pipe.named_steps["model"]
        logger.info("Binary classifier loaded: %s", type(binary_model).__name__)

        # ── SHAP background + feature names (from binary run) ────────────
        bg_path = mlflow.artifacts.download_artifacts(
            artifact_uri=(
                f"runs:/{self._binary_run_id}"
                "/shap_background/shap_background_sample.npy"
            ),
            dst_path=tmp,
        )
        background_data: np.ndarray = np.load(bg_path)

        fn_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{self._binary_run_id}/feature_names.json",
            dst_path=tmp,
        )
        with open(fn_path) as fh:
            feature_names = json.load(fh)["feature_names"]

        logger.info(
            "SHAP background loaded: shape=%s  feature_count=%d",
            background_data.shape,
            len(feature_names),
        )

        # ── Regression model (delay_minutes, Day 7) ──────────────────────
        regr_full_pipe = mlflow.sklearn.load_model(
            f"runs:/{self._regression_run_id}/pipeline"
        )
        regr_model = regr_full_pipe.named_steps["model"]
        logger.info("Regression model loaded: %s", type(regr_model).__name__)

        # ── Root-cause classifier (44-feature pipeline, Day 8) ───────────
        rc_full_pipe = mlflow.sklearn.load_model(
            f"runs:/{self._rc_run_id}/pipeline"
        )
        rc_preproc = mlflow.sklearn.load_model(
            f"runs:/{self._rc_run_id}/preprocessing_pipeline"
        )
        rc_model = rc_full_pipe.named_steps["model"]
        logger.info(
            "Root-cause model loaded: %s  classes=%s",
            type(rc_model).__name__,
            list(rc_model.classes_),
        )

        # ── Assemble DelayExplainer ───────────────────────────────────────
        self._explainer = DelayExplainer(
            preprocessing_pipeline=binary_preproc,
            binary_model=binary_model,
            background_data=background_data,
            feature_names=feature_names,
            regressor=regr_model,
            root_cause_model=rc_model,
            root_cause_preprocessing_pipeline=rc_preproc,
        )
        self._model_info = ModelInfo(
            binary_run_id=self._binary_run_id,
            regression_run_id=self._regression_run_id,
            root_cause_run_id=self._rc_run_id,
            feature_count=len(feature_names),
            loaded_at=datetime.now(timezone.utc),
        )
        logger.info(
            "MLflowModelRegistry ready: binary_run_id=%s  feature_count=%d",
            self._binary_run_id,
            len(feature_names),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def explainer(self) -> DelayExplainer:
        if self._explainer is None:
            raise RuntimeError(
                "MLflowModelRegistry not loaded. Call load() before use."
            )
        return self._explainer

    @property
    def model_info(self) -> ModelInfo:
        if self._model_info is None:
            raise RuntimeError(
                "MLflowModelRegistry not loaded. Call load() before use."
            )
        return self._model_info
