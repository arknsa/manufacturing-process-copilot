"""
backend/app/services/ml/explainability.py
==========================================
SHAP-based explainability service for the Manufacturing Process Copilot.

Translates raw ML predictions into plain-English narratives that production
planners can act on.  Every delay prediction served by the backend must be
accompanied by an ExplanationResult.

Public API
----------
DelayExplainer
    Wraps a binary delay classifier, its preprocessing pipeline, a SHAP
    TreeExplainer, and optional regression / root-cause classifiers.
    ``explain_order(order_features)`` returns a full ExplanationResult.
    ``global_importance()`` returns a feature importance DataFrame.

ExplanationResult
    Structured explanation for a single order.  Contains probability,
    confidence tier, top risk / mitigating factors, narrative string, and
    the raw SHAP value array.

FactorExplanation
    A single contributing feature: human label, raw value, SHAP contribution,
    direction, and magnitude.

Architecture references
-----------------------
* Doc 04 §Week 2, Day 8 — SHAP Explainability specification
* ml/src/mpc_ml/tracking/mlflow_utils.py — canonical MLflow artifact paths
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable feature labels (Doc 04 §Day 8 table + extended coverage)
# ---------------------------------------------------------------------------

_FEATURE_LABELS: Dict[str, str] = {
    # Core labels specified in Doc 04
    "work_center_queue_depth_at_release": "Machine queue congestion",
    "release_lag_hours":                  "Late material release",
    "schedule_tightness_ratio":           "Schedule tightness",
    "material_availability_at_release":   "Material availability",
    "priority_encoded":                   "Order priority",
    "machine_oee_30d":                    "Machine performance",
    "product_delay_rate_90d":             "Product delay history",
    "operator_skill_tier_encoded":        "Operator experience level",
    "days_since_last_planned_maintenance":"Maintenance recency",
    "changeover_complexity_score":        "Changeover complexity",
    # Extended labels for remaining features
    "planned_lead_time_hours":            "Planned lead time",
    "estimated_total_hours":              "Estimated processing hours",
    "quantity":                           "Order quantity",
    "machine_unplanned_downtime_hours_30d": "Machine downtime (30d)",
    "operator_experience_months":         "Operator experience",
    "machine_avg_delay_minutes_90d":      "Machine average delay (90d)",
    "lag_as_pct_of_window":              "Release lag as % of window",
    "machine_utilization_at_release":     "Machine utilization",
    "machine_delay_rate_90d":             "Machine delay history",
    "operator_delay_rate_90d":            "Operator delay history",
    "product_x_machine_delay_rate_90d":   "Product-machine delay history",
    "product_first_pass_yield_90d":       "First-pass yield",
    "machine_setup_overrun_rate_90d":     "Setup overrun rate",
    "shift_delay_rate_30d":              "Shift delay rate",
    "tightness_x_queue":                 "Congestion risk score",
    "util_x_queue":                      "Utilization-queue interaction",
    "util_x_tight":                      "Utilization-tightness interaction",
    "is_expedited":                       "Expedited order",
    "is_month_end":                       "Month-end pressure",
    "is_quarter_end":                     "Quarter-end pressure",
    "maintenance_due_within_order_window":"Maintenance due during order",
    "changeover_required":               "Changeover required",
    "component_shortage_count":          "Component shortages",
    "schedule_revision_count":           "Schedule revisions",
    "product_complexity_score":          "Product complexity",
    "material_bom_complexity":           "BOM complexity",
    "operation_count":                   "Routing step count",
    "hours_into_shift_at_start":         "Hours into shift",
    "shift_type_encoded":                "Shift type",
    "planned_start_day_of_week":         "Day of week",
    "planned_start_hour":                "Start hour",
    "oee_x_maintenance_ratio":           "OEE-maintenance interaction",
    "log_experience_x_concurrent":       "Experience-concurrency interaction",
    "operator_concurrent_order_count":   "Concurrent order count",
}

# Probability thresholds for confidence tier
_HIGH_PROB_THRESHOLD: float = 0.65
_LOW_PROB_THRESHOLD: float = 0.35

# SHAP magnitude classification: fraction of max |SHAP| in this explanation
_HIGH_SHAP_FRACTION: float = 0.50
_MED_SHAP_FRACTION: float = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_label(feature_name: str) -> str:
    return _FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def _shap_magnitude(shap_val: float, max_abs_shap: float) -> str:
    if max_abs_shap == 0.0:
        return "low"
    ratio = abs(shap_val) / max_abs_shap
    if ratio >= _HIGH_SHAP_FRACTION:
        return "high"
    if ratio >= _MED_SHAP_FRACTION:
        return "medium"
    return "low"


def _to_numpy(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FactorExplanation:
    """One contributing feature in a delay explanation.

    Attributes
    ----------
    feature_name:
        Raw pipeline feature name (e.g. ``"release_lag_hours"``).
    human_label:
        Plain-English label for display (e.g. ``"Late material release"``).
    value:
        Raw feature value from the original order dict (before pipeline
        transformation).
    shap_contribution:
        SHAP value for this feature in this order.  Positive = increases
        delay risk; negative = reduces delay risk.
    direction:
        ``"increases_risk"`` when ``shap_contribution > 0``;
        ``"reduces_risk"`` otherwise.
    magnitude:
        ``"high"`` / ``"medium"`` / ``"low"`` relative to the largest
        |SHAP| value in this explanation.
    """
    feature_name: str
    human_label: str
    value: Any
    shap_contribution: float
    direction: str
    magnitude: str


@dataclass
class ExplanationResult:
    """Full delay explanation for a single manufacturing order.

    Attributes
    ----------
    predicted_delay_probability:
        P(is_delayed=1) from the binary classifier, in [0, 1].
    predicted_delay_minutes:
        Regression model estimate of delay duration (minutes).  ``None``
        when no regressor was provided to DelayExplainer.
    predicted_root_cause:
        Most-likely root cause label from the root-cause classifier.
        ``"unknown"`` when no root-cause model was provided.
    confidence:
        ``"high"`` when probability >= 0.65; ``"medium"`` when 0.35–0.65;
        ``"low"`` when < 0.35.
    top_risk_factors:
        Up to 3 features with the largest positive SHAP contributions,
        sorted descending by contribution.
    mitigating_factors:
        Up to 3 features with the largest (most negative) SHAP contributions,
        sorted ascending by contribution.
    narrative:
        Plain-English explanation string (4 sentences, Doc 04 template).
    shap_values:
        Raw 1-D SHAP value array, length = number of pipeline features.
        Exposed for downstream charting; not included in repr.
    """
    predicted_delay_probability: float
    predicted_delay_minutes: Optional[float]
    predicted_root_cause: str
    confidence: str
    top_risk_factors: List[FactorExplanation]
    mitigating_factors: List[FactorExplanation]
    narrative: str
    shap_values: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# DelayExplainer
# ---------------------------------------------------------------------------

class DelayExplainer:
    """SHAP-based explainer for the MPC delay prediction system.

    Wraps a binary delay classifier and its preprocessing pipeline.
    Optionally accepts a regression model (delay minutes) and a root-cause
    classifier for richer ExplanationResults.

    Parameters
    ----------
    preprocessing_pipeline:
        Fitted preprocessing-only pipeline (the ``'preprocessor'`` step of
        the full ``Pipeline([preprocessor, model])``).  Used to transform
        raw 38-column order DataFrames to the 44-column model input space.
    binary_model:
        Fitted binary LightGBM/XGBoost classifier for ``is_delayed``.
        Must implement ``predict_proba()``.
    background_data:
        2-D float64 array of shape ``(n_background, n_features)`` —
        a sample of the transformed training data.  Used to initialise
        ``shap.TreeExplainer`` with a stable expected-value baseline.
        Loaded from the MLflow artifact
        ``shap_background/shap_background_sample.npy``.
    feature_names:
        Ordered list of 44 feature names corresponding to columns of
        ``background_data``.  Loaded from the MLflow artifact
        ``feature_names.json``.
    regressor:
        Optional fitted regressor for ``delay_minutes`` (log1p-transformed
        target expected).  When provided, ``ExplanationResult.predicted_delay_minutes``
        is populated.
    root_cause_model:
        Optional fitted multi-class classifier for ``delay_root_cause``.
    root_cause_preprocessing_pipeline:
        Optional separate preprocessing pipeline for the root-cause
        classifier.  Defaults to ``preprocessing_pipeline`` when ``None``.
    """

    def __init__(
        self,
        preprocessing_pipeline: Pipeline,
        binary_model: Any,
        background_data: np.ndarray,
        feature_names: List[str],
        *,
        regressor: Optional[Any] = None,
        root_cause_model: Optional[Any] = None,
        root_cause_preprocessing_pipeline: Optional[Pipeline] = None,
    ) -> None:
        self._preproc = preprocessing_pipeline
        self._binary_model = binary_model
        self._feature_names = list(feature_names)
        self._background = background_data
        self._regressor = regressor
        self._rc_model = root_cause_model
        self._rc_preproc = root_cause_preprocessing_pipeline or preprocessing_pipeline
        # Lazy import: shap uses C extensions that require matching NumPy ABI.
        # Deferring to instantiation time keeps the module importable for tests.
        import shap  # noqa: PLC0415
        self._shap_explainer = shap.TreeExplainer(binary_model, data=background_data)
        logger.info(
            "DelayExplainer ready: n_features=%d background_n=%d "
            "has_regressor=%s has_rc_model=%s",
            len(feature_names), len(background_data),
            regressor is not None, root_cause_model is not None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_order(self, order_features: dict) -> ExplanationResult:
        """Explain the delay prediction for a single manufacturing order.

        Parameters
        ----------
        order_features:
            Raw feature dict with the 38-column order schema (no TARGET_COLS).
            Keys are feature names; values are scalars.

        Returns
        -------
        ExplanationResult
            Full explanation including probability, narrative, top risk /
            mitigating factors, and raw SHAP values.
        """
        X_raw = pd.DataFrame([order_features])
        X_t = self._preproc.transform(X_raw)
        X_np = _to_numpy(X_t)

        # ── Binary delay probability ───────────────────────────────────────
        prob = float(self._binary_model.predict_proba(X_np)[0, 1])

        # ── SHAP values for positive class ────────────────────────────────
        raw_shap = self._shap_explainer.shap_values(X_np, check_additivity=False)
        # LightGBM TreeExplainer may return list [class0, class1] or single array
        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            sv = raw_shap[1][0]   # shape (n_features,) for class 1
        elif isinstance(raw_shap, list):
            sv = raw_shap[0][0]
        else:
            sv = raw_shap[0]      # shape (n_features,)
        sv = np.asarray(sv, dtype=np.float64)

        # ── Delay minutes (optional regressor) ───────────────────────────
        delay_minutes: Optional[float] = None
        if self._regressor is not None:
            raw_pred = float(self._regressor.predict(X_np)[0])
            delay_minutes = max(0.0, float(np.expm1(raw_pred)))

        # ── Root cause (optional classifier) ─────────────────────────────
        root_cause = "unknown"
        if self._rc_model is not None:
            X_t_rc = self._rc_preproc.transform(X_raw)
            X_np_rc = _to_numpy(X_t_rc)
            root_cause = str(self._rc_model.predict(X_np_rc)[0])

        # ── Confidence tier ───────────────────────────────────────────────
        if prob >= _HIGH_PROB_THRESHOLD:
            confidence = "high"
        elif prob >= _LOW_PROB_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "low"

        # ── Build FactorExplanation objects ───────────────────────────────
        max_abs = float(np.max(np.abs(sv))) if sv.size > 0 else 1.0
        factors: List[FactorExplanation] = []
        for fname, sv_i in zip(self._feature_names, sv):
            raw_val: Any = order_features.get(fname, float("nan"))
            factors.append(FactorExplanation(
                feature_name=fname,
                human_label=_human_label(fname),
                value=raw_val,
                shap_contribution=float(sv_i),
                direction="increases_risk" if sv_i > 0 else "reduces_risk",
                magnitude=_shap_magnitude(sv_i, max_abs),
            ))

        risk_factors = sorted(
            [f for f in factors if f.shap_contribution > 0],
            key=lambda f: f.shap_contribution,
            reverse=True,
        )[:3]
        mitigating_factors = sorted(
            [f for f in factors if f.shap_contribution < 0],
            key=lambda f: f.shap_contribution,
        )[:3]

        narrative = self._build_narrative(
            prob, confidence, risk_factors, mitigating_factors,
            root_cause, delay_minutes,
        )

        return ExplanationResult(
            predicted_delay_probability=prob,
            predicted_delay_minutes=delay_minutes,
            predicted_root_cause=root_cause,
            confidence=confidence,
            top_risk_factors=risk_factors,
            mitigating_factors=mitigating_factors,
            narrative=narrative,
            shap_values=sv,
        )

    def global_importance(self) -> pd.DataFrame:
        """Return mean absolute SHAP importance ranked table.

        Computes SHAP values on the stored background dataset and returns
        a DataFrame with one row per feature, sorted by mean(|SHAP|).

        Returns
        -------
        pd.DataFrame
            Columns: ``feature_name``, ``mean_abs_shap``, ``rank``.
        """
        raw_shap = self._shap_explainer.shap_values(self._background, check_additivity=False)
        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            sv_matrix = raw_shap[1]
        elif isinstance(raw_shap, list):
            sv_matrix = raw_shap[0]
        else:
            sv_matrix = raw_shap

        mean_abs = np.abs(sv_matrix).mean(axis=0)
        df = pd.DataFrame({
            "feature_name": self._feature_names,
            "mean_abs_shap": mean_abs.astype(float),
        })
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_narrative(
        self,
        prob: float,
        confidence: str,
        risk_factors: List[FactorExplanation],
        mitigating_factors: List[FactorExplanation],
        root_cause: str,
        delay_minutes: Optional[float],
    ) -> str:
        """Generate a plain-English narrative from the Doc 04 template."""
        parts: List[str] = []

        # Sentence 1: Risk summary
        parts.append(
            f"This order has a {confidence} risk of delay ({prob:.0%} probability)."
        )

        # Sentence 2: Main risk factors
        if risk_factors:
            factor_strs = [
                f"{f.human_label} ({f.magnitude} impact)"
                for f in risk_factors[:2]
            ]
            parts.append(
                "The main risk factors are: " + ", ".join(factor_strs) + "."
            )
        else:
            parts.append("No significant risk factors were identified.")

        # Sentence 3: Mitigating factors
        if mitigating_factors:
            mit_strs = [f.human_label.lower() for f in mitigating_factors[:2]]
            parts.append(
                "Risk is partially offset by: " + ", ".join(mit_strs) + "."
            )
        else:
            parts.append("No significant risk mitigators were identified.")

        # Sentence 4: Root cause + delay estimate
        rc_label = root_cause.replace("_", " ")
        if delay_minutes is not None:
            parts.append(
                f"If delayed, the predicted root cause is {rc_label} "
                f"with an estimated delay of {delay_minutes:.0f} minutes."
            )
        else:
            parts.append(
                f"If delayed, the predicted root cause is {rc_label}."
            )

        return "\n\n".join(parts)
