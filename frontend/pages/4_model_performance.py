"""
frontend/pages/4_model_performance.py
========================================
MLops dashboard — active model info and global feature importance.

Calls:
  • GET /api/v1/models/current         → model metadata card
  • GET /api/v1/models/feature-importance → global mean |SHAP| chart
"""
from __future__ import annotations

import sys
import os

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import streamlit as st

from utils.bootstrap import ensure_session_state
from components.shap_chart import render_shap_chart
from components.metrics_table import render_metrics_comparison

st.set_page_config(
    page_title="Model Performance | MPC",
    page_icon="📊",
    layout="wide",
)

ensure_session_state()
client = st.session_state.api_client

st.title("📊 Model Performance")
st.caption(
    "Live data from GET /api/v1/models/current and "
    "GET /api/v1/models/feature-importance."
)
st.divider()

# ---------------------------------------------------------------------------
# Active model metadata
# ---------------------------------------------------------------------------

st.subheader("Active Model")

with st.spinner("Fetching model info…"):
    model_info = client.get_model_info()

if not model_info:
    st.warning(
        "Model metadata unavailable — is the backend running with a "
        "loaded MLflow model?"
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Binary AUC", model_info.get("binary_auc", "N/A"))
    c2.metric("Feature Count", str(model_info.get("feature_count", "N/A")))

    binary_run = model_info.get("binary_run_id", "") or ""
    regression_run = model_info.get("regression_run_id", "") or ""
    c3.metric(
        "Binary Run",
        binary_run[:8] + "…" if len(binary_run) > 8 else binary_run or "N/A",
    )
    loaded_at = model_info.get("loaded_at", "") or ""
    c4.metric("Loaded At", loaded_at[:10] if loaded_at else "N/A")

    # Detailed run IDs in an expander to keep the page clean
    with st.expander("Full model metadata", expanded=False):
        st.json(model_info)

st.divider()

# ---------------------------------------------------------------------------
# Global feature importance — mean |SHAP|
# ---------------------------------------------------------------------------

st.subheader("Global Feature Importance (Mean |SHAP|)")
st.caption(
    "Computed over recent predictions. Higher values indicate features "
    "that consistently drive predictions away from the base rate."
)

with st.spinner("Fetching feature importance…"):
    importance_raw: list[dict] = client.get_feature_importance()

if not importance_raw:
    st.info(
        "Feature importance data not available. "
        "It is computed when the model is loaded and predictions exist."
    )
else:
    # Normalise to the shape render_shap_chart expects
    normalised = [
        {
            "human_label": f.get("human_label") or f.get("feature_name", "Unknown"),
            "feature_name": f.get("feature_name", ""),
            "shap_contribution": abs(
                f.get("mean_abs_shap", f.get("importance", f.get("shap_contribution", 0)))
            ),
            "direction": "increases_risk",  # all global values are positive
            "magnitude": f.get("magnitude", ""),
        }
        for f in importance_raw
    ]

    render_shap_chart(
        normalised,
        title="Mean |SHAP| by Feature (top 20)",
        max_factors=20,
    )

    st.divider()
    with st.expander("Raw feature importance data", expanded=False):
        render_metrics_comparison(importance_raw)
