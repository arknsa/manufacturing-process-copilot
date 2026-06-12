"""
frontend/app.py
================
Streamlit entry point for the Manufacturing Process Copilot dashboard.

Run from the frontend/ directory:
    streamlit run app.py

Or from the project root:
    streamlit run frontend/app.py

This file:
  1. Adds frontend/ to sys.path so that all sub-packages are importable.
  2. Calls st.set_page_config (must be the very first Streamlit call).
  3. Initialises session state via ensure_session_state().
  4. Renders the sidebar connection status.
  5. Renders the home/overview page with navigation cards.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — makes `services`, `components`, `utils` importable
# from any page, regardless of CWD when streamlit is invoked.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st

st.set_page_config(
    page_title="Manufacturing Process Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.bootstrap import ensure_session_state  # noqa: E402
from utils.formatting import format_probability   # noqa: E402

ensure_session_state()

client = st.session_state.api_client

# ---------------------------------------------------------------------------
# Sidebar — connection badge + quick links
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏭 MPC")
    st.divider()

    healthy = client.health_check()
    if healthy:
        st.success("Backend connected", icon="✅")
    else:
        st.error("Backend unreachable", icon="❌")
        st.caption(f"Expecting API at `{st.session_state.api_base_url}`")

    st.divider()
    st.caption(
        f"Chat session: `{st.session_state.chat_session_token[:12]}…`"
    )

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

st.title("🏭 Manufacturing Process Copilot")
st.markdown(
    "ML-powered delay prediction · root-cause diagnosis · "
    "AI planning assistant"
)
st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 💬 Copilot Chat")
    st.caption("Ask the AI about orders, delays, and recommendations.")
    st.page_link("pages/1_copilot_chat.py", label="Open Chat →")

with col2:
    st.markdown("### 📋 Risk Board")
    st.caption("Live colour-coded table of today's orders by risk level.")
    st.page_link("pages/2_risk_board.py", label="Open Risk Board →")

with col3:
    st.markdown("### 🔍 Order Detail")
    st.caption("Deep-dive: SHAP chart, root-cause, and narrative.")
    st.page_link("pages/3_order_detail.py", label="Open Order Detail →")

with col4:
    st.markdown("### 📊 Model Performance")
    st.caption("Active model metrics and global feature importance.")
    st.page_link("pages/4_model_performance.py", label="Open Model Stats →")

st.divider()

# ---------------------------------------------------------------------------
# Live backend snapshot (only when healthy)
# ---------------------------------------------------------------------------

if not healthy:
    st.info(
        "Start the FastAPI backend (`uvicorn backend.app.main:app --reload`) "
        "to see live data."
    )
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.subheader("Active Model")
    model_info = client.get_model_info()
    if model_info:
        st.metric("Binary AUC", model_info.get("binary_auc", "N/A"))
        st.metric("Features", str(model_info.get("feature_count", "N/A")))
        loaded = model_info.get("loaded_at", "")
        st.caption(f"Loaded: {loaded[:19].replace('T', ' ') if loaded else 'unknown'}")
    else:
        st.warning("Model metadata unavailable.")

with right:
    st.subheader("Today's Orders")
    orders = client.get_today_orders()
    if orders:
        total = len(orders)
        statuses = [o.get("status", "") for o in orders]
        in_prog = statuses.count("in_progress")
        pending = statuses.count("pending")
        completed = statuses.count("completed")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", total)
        c2.metric("Pending", pending)
        c3.metric("In Progress", in_prog)
        c4.metric("Completed", completed)

        if any("delay_probability" in o for o in orders):
            probs = [
                o["delay_probability"]
                for o in orders
                if "delay_probability" in o
            ]
            high = sum(1 for p in probs if p >= 0.65)
            st.metric(
                "High-risk orders",
                high,
                delta=f"{high / len(probs) * 100:.0f}% of orders with predictions",
                delta_color="inverse",
            )
    else:
        st.info("No orders found for today.")
