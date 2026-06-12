"""
frontend/components/risk_gauge.py
===================================
Plotly gauge chart for delay probability.

Three colour zones match the risk thresholds defined in the architecture:
  • Green  — probability < 0.40  (low risk)
  • Amber  — 0.40 ≤ probability < 0.65 (medium risk)
  • Red    — probability ≥ 0.65  (high risk)
"""
from __future__ import annotations

import sys
import os

# Ensure frontend/ is on sys.path when this module is imported from a page
_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import plotly.graph_objects as go
import streamlit as st

from utils.formatting import risk_colour, prob_to_risk_label


def render_risk_gauge(
    probability: float,
    title: str = "Delay Probability",
    height: int = 280,
) -> None:
    """Render a Plotly gauge for *probability* (0–1 scale).

    Parameters
    ----------
    probability:
        Float in [0, 1].
    title:
        Text shown above the gauge needle.
    height:
        Pixel height of the chart element.
    """
    pct = min(max(probability * 100, 0), 100)
    colour = risk_colour(probability)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": title, "font": {"size": 15, "color": "#333"}},
            number={"suffix": "%", "font": {"size": 30, "color": colour}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#aaa",
                    "tickvals": [0, 20, 40, 65, 80, 100],
                },
                "bar": {"color": colour, "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#ddd",
                "steps": [
                    {"range": [0, 40], "color": "#d5f5e3"},
                    {"range": [40, 65], "color": "#fef9e7"},
                    {"range": [65, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "#c0392b", "width": 3},
                    "thickness": 0.80,
                    "value": 65,
                },
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    st.plotly_chart(fig, use_container_width=True)
