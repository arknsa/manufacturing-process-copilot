"""
frontend/components/metrics_table.py
=======================================
Styled metrics comparison table using st.dataframe with ProgressColumn
formatting for standard ML metric columns.

Used by:
  • pages/4_model_performance.py  — model registry comparison
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# Metric columns that get a ProgressColumn (values in [0, 1])
_PROGRESS_KEYWORDS = ("auc", "f1", "precision", "recall", "accuracy", "ap")
# Metric columns that get NumberColumn formatting instead
_NUMBER_KEYWORDS = ("loss", "mse", "rmse", "mae", "error")


def render_metrics_comparison(
    data: list[dict],
    title: str = "",
) -> None:
    """Render *data* as a styled dataframe.

    Float columns whose names contain known metric keywords are displayed
    as progress bars (bounded [0, 1]).  Other float columns use numeric
    formatting.

    Parameters
    ----------
    data:
        List of dicts, one per model/experiment row.
    title:
        Optional heading rendered above the table.
    """
    if not data:
        st.info("No metrics data available.")
        return

    if title:
        st.subheader(title)

    df = pd.DataFrame(data)
    column_config: dict = {}

    for col in df.columns:
        col_lower = col.lower()

        if df[col].dtype not in (float, "float64", "float32"):
            continue

        if any(k in col_lower for k in _PROGRESS_KEYWORDS):
            column_config[col] = st.column_config.ProgressColumn(
                col.replace("_", " ").title(),
                min_value=0.0,
                max_value=1.0,
                format="%.4f",
            )
        elif any(k in col_lower for k in _NUMBER_KEYWORDS):
            column_config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.4f",
            )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config or None,
    )
