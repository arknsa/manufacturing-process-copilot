"""
frontend/components/shap_chart.py
====================================
Horizontal bar chart for SHAP factor contributions.

Accepts the `top_risk_factors` / `mitigating_factors` lists from a
DelayPrediction response.  Each element should match the
FactorExplanationResponse schema:
    {
        "feature_name": str,
        "human_label": str,
        "value": float,
        "shap_contribution": float,   # positive = increases risk
        "direction": "increases_risk" | "reduces_risk",
        "magnitude": "high" | "medium" | "low",
    }
"""
from __future__ import annotations

import sys
import os

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import plotly.graph_objects as go
import streamlit as st

_RISK_RED = "#e74c3c"
_MIT_GREEN = "#27ae60"
_NEUTRAL = "#7f8c8d"


def render_shap_chart(
    factors: list[dict],
    title: str = "Risk Factor Contributions",
    max_factors: int = 10,
) -> None:
    """Render a horizontal Plotly bar chart of SHAP contributions.

    Bars are coloured red (increases risk) or green (reduces risk).
    Sorted by |SHAP| descending; largest at the top of the chart.
    """
    if not factors:
        st.info("No SHAP factor data available.")
        return

    # Sort descending by absolute contribution, keep top N
    sorted_f = sorted(
        factors,
        key=lambda f: abs(f.get("shap_contribution", 0)),
        reverse=True,
    )[:max_factors]
    # Reverse so the highest-impact bar sits at the top of the chart
    sorted_f = list(reversed(sorted_f))

    labels = [
        f.get("human_label") or f.get("feature_name", "Unknown")
        for f in sorted_f
    ]
    values = [f.get("shap_contribution", 0.0) for f in sorted_f]
    directions = [f.get("direction", "") for f in sorted_f]
    magnitudes = [f.get("magnitude", "") for f in sorted_f]
    raw_values = [f.get("value", "") for f in sorted_f]

    colours = [
        _RISK_RED if d == "increases_risk" else (_MIT_GREEN if d == "reduces_risk" else _NEUTRAL)
        for d in directions
    ]

    hover = [
        f"<b>{lbl}</b><br>"
        f"Raw value: {rv}<br>"
        f"SHAP: {v:+.4f}<br>"
        f"Impact: {mag.title()}<br>"
        "<extra></extra>"
        for lbl, rv, v, mag in zip(labels, raw_values, values, magnitudes)
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            hovertemplate=hover,
        )
    )

    x_range = max(abs(v) for v in values) if values else 1.0
    padding = x_range * 0.35  # room for outside text labels

    fig.update_layout(
        title={"text": title, "font": {"size": 14}} if title else None,
        xaxis=dict(
            title="← reduces risk  |  increases risk →",
            range=[-(x_range + padding), x_range + padding],
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=1.5,
            gridcolor="#eeeeee",
        ),
        yaxis={"automargin": True},
        height=max(260, len(sorted_f) * 36 + 90),
        margin=dict(l=10, r=90, t=40 if title else 10, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif", "size": 12},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
