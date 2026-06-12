"""
frontend/pages/2_risk_board.py
================================
Live risk board — today's production orders colour-coded by delay probability.

Features:
  • Fetches GET /api/v1/orders/today on every page load
  • Summary metrics: total, high-risk count, in-progress, avg probability
  • st.dataframe with ProgressColumn for delay_probability (when available)
  • Order selectbox → switches to Order Detail page via st.switch_page
  • Optional 60-second auto-refresh toggle
"""
from __future__ import annotations

import sys
import os
import time

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import pandas as pd
import streamlit as st

from utils.bootstrap import ensure_session_state
from utils.formatting import format_probability, status_to_display

st.set_page_config(
    page_title="Risk Board | MPC",
    page_icon="📋",
    layout="wide",
)

ensure_session_state()
client = st.session_state.api_client

# ---------------------------------------------------------------------------
# Header row
# ---------------------------------------------------------------------------

st.title("📋 Today's Production Orders — Risk Board")

hdr_left, hdr_right = st.columns([3, 1])
with hdr_right:
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False, key="rb_autorefresh")
with hdr_left:
    st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

with st.spinner("Loading orders…"):
    orders: list[dict] = client.get_today_orders()

if not orders:
    st.warning(
        "No orders found for today, or the backend is unreachable. "
        "Check that the FastAPI service is running."
    )
    if auto_refresh:
        time.sleep(60)
        st.rerun()
    st.stop()

df = pd.DataFrame(orders)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

total = len(df)
in_progress_count = int((df["status"] == "in_progress").sum()) if "status" in df.columns else 0
pending_count = int((df["status"] == "pending").sum()) if "status" in df.columns else 0
completed_count = int((df["status"] == "completed").sum()) if "status" in df.columns else 0

has_probs = "delay_probability" in df.columns

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Orders", total)
m2.metric("Pending", pending_count)
m3.metric("In Progress", in_progress_count)
m4.metric("Completed", completed_count)

if has_probs:
    high_risk = int((df["delay_probability"] >= 0.65).sum())
    m5.metric(
        "High Risk ⚠️",
        high_risk,
        delta=f"{high_risk / total * 100:.0f}%",
        delta_color="inverse",
    )

st.divider()

# ---------------------------------------------------------------------------
# Build display DataFrame
# ---------------------------------------------------------------------------

DISPLAY_COLS: list[str] = []
COL_CONFIG: dict = {}

_add = DISPLAY_COLS.append

if "order_number" in df.columns:
    _add("order_number")
    COL_CONFIG["order_number"] = st.column_config.TextColumn("Order #", width="small")

if "status" in df.columns:
    df["status_display"] = df["status"].apply(status_to_display)
    _add("status_display")
    COL_CONFIG["status_display"] = st.column_config.TextColumn("Status", width="small")

if "priority" in df.columns:
    _add("priority")
    COL_CONFIG["priority"] = st.column_config.TextColumn("Priority", width="small")

if "planned_start" in df.columns:
    _add("planned_start")
    COL_CONFIG["planned_start"] = st.column_config.DatetimeColumn(
        "Planned Start", format="YYYY-MM-DD HH:mm", width="medium"
    )

if "planned_end" in df.columns:
    _add("planned_end")
    COL_CONFIG["planned_end"] = st.column_config.DatetimeColumn(
        "Planned End", format="YYYY-MM-DD HH:mm", width="medium"
    )

if "quantity" in df.columns:
    _add("quantity")
    COL_CONFIG["quantity"] = st.column_config.NumberColumn("Qty", width="small")

if "estimated_total_hours" in df.columns:
    _add("estimated_total_hours")
    COL_CONFIG["estimated_total_hours"] = st.column_config.NumberColumn(
        "Est. Hours", format="%.1f", width="small"
    )

if has_probs:
    # Sort by risk descending before displaying
    df = df.sort_values("delay_probability", ascending=False)
    _add("delay_probability")
    COL_CONFIG["delay_probability"] = st.column_config.ProgressColumn(
        "Delay Risk",
        min_value=0.0,
        max_value=1.0,
        format="%.1%",
        width="medium",
    )

# ---------------------------------------------------------------------------
# Render table
# ---------------------------------------------------------------------------

if DISPLAY_COLS:
    st.dataframe(
        df[DISPLAY_COLS],
        use_container_width=True,
        hide_index=True,
        column_config=COL_CONFIG,
        height=min(600, 55 + 35 * len(df)),
    )
else:
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Order detail navigation
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Open Order Detail")

# Build label → order dict mapping
order_labels: dict[str, dict] = {}
for o in orders:
    num = o.get("order_number", "")
    oid = str(o.get("id", ""))
    prob = o.get("delay_probability")
    risk_str = f" — {format_probability(prob)}" if prob is not None else ""
    label = f"{num or oid[:8]}{risk_str}"
    order_labels[label] = o

if order_labels:
    selected_label = st.selectbox(
        "Select an order",
        options=list(order_labels.keys()),
        key="rb_order_select",
    )
    if st.button("🔍 Open Order Detail", type="primary"):
        selected = order_labels[selected_label]
        st.session_state.selected_order_id = str(selected.get("id", ""))
        st.session_state.selected_order_data = selected
        st.switch_page("pages/3_order_detail.py")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(60)
    st.rerun()
