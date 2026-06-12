"""
frontend/pages/3_order_detail.py
==================================
Deep-dive view for a single production order.

Receives an order from one of two sources (in priority order):
  1. st.session_state.selected_order_data  — set by 2_risk_board.py
  2. Manual order-ID text input on this page

Renders:
  • Order metadata cards
  • Plotly delay probability gauge
  • Prediction summary (root cause, confidence, estimated delay minutes)
  • Narrative explanation
  • Side-by-side SHAP charts: top risk factors and mitigating factors
  • "Ask Copilot" shortcut that pre-fills the chat with an order question
"""
from __future__ import annotations

import sys
import os

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import streamlit as st

from utils.bootstrap import ensure_session_state
from utils.formatting import (
    confidence_badge_colour,
    format_probability,
    minutes_to_display,
    prob_to_risk_label,
    risk_colour,
    root_cause_to_display,
    status_to_display,
)
from components.risk_gauge import render_risk_gauge
from components.shap_chart import render_shap_chart

st.set_page_config(
    page_title="Order Detail | MPC",
    page_icon="🔍",
    layout="wide",
)

ensure_session_state()
client = st.session_state.api_client

# ---------------------------------------------------------------------------
# Resolve which order to display
# ---------------------------------------------------------------------------

order_data: dict | None = st.session_state.get("selected_order_data")
order_id: str | None = st.session_state.get("selected_order_id")

if not order_data:
    st.title("🔍 Order Detail")
    st.info(
        "No order selected. Navigate here from the **Risk Board**, "
        "or enter an Order ID manually below."
    )
    manual_id = st.text_input("Order ID (UUID)", placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    if manual_id and st.button("Load", type="primary"):
        # Try to find in today's orders (cheapest route — no dedicated GET /orders/{id})
        all_orders = client.get_today_orders()
        match = next((o for o in all_orders if str(o.get("id", "")) == manual_id.strip()), None)
        if match:
            st.session_state.selected_order_id = manual_id.strip()
            st.session_state.selected_order_data = match
            st.rerun()
        else:
            st.error(f"Order `{manual_id}` not found in today's orders.")
    st.stop()

order_number: str = order_data.get("order_number") or (order_id or "")[:8]

# ---------------------------------------------------------------------------
# Page header + quick-action buttons
# ---------------------------------------------------------------------------

st.title(f"🔍 Order {order_number}")

btn_left, btn_right = st.columns([2, 3])
with btn_left:
    if st.button("← Back to Risk Board"):
        st.switch_page("pages/2_risk_board.py")
with btn_right:
    if st.button("💬 Ask Copilot about this order", type="primary"):
        prompt = (
            f"Tell me about order {order_number}: what is the delay risk, "
            "what are the main causes, and what actions should I take?"
        )
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.switch_page("pages/1_copilot_chat.py")

st.divider()

# ---------------------------------------------------------------------------
# Order metadata
# ---------------------------------------------------------------------------

st.subheader("Order Information")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Status", status_to_display(order_data.get("status", "N/A")))
m2.metric("Quantity", order_data.get("quantity", "N/A"))
m3.metric(
    "Est. Duration",
    f"{order_data.get('estimated_total_hours', 0):.1f}h",
)
priority = order_data.get("priority", "normal")
m4.metric("Priority", priority.title())

planned_start = order_data.get("planned_start", "")
planned_end = order_data.get("planned_end", "")
if planned_start and planned_end:
    st.caption(
        f"Planned window: {planned_start[:16].replace('T', ' ')} UTC  →  "
        f"{planned_end[:16].replace('T', ' ')} UTC"
    )

actual_start = order_data.get("actual_start")
actual_end = order_data.get("actual_end")
if actual_start or actual_end:
    st.caption(
        f"Actual: {(actual_start or '—')[:16].replace('T', ' ')}  →  "
        f"{(actual_end or 'ongoing')[:16].replace('T', ' ')}"
    )

st.divider()

# ---------------------------------------------------------------------------
# Extract prediction payload
# (future: dedicated GET /predictions/{order_id}; for now use inline fields)
# ---------------------------------------------------------------------------

prediction: dict | None = order_data.get("prediction")

# Some backend shapes embed prediction fields directly on the order object
if prediction is None and "delay_probability" in order_data:
    prediction = order_data

if prediction is None:
    st.info(
        "No prediction stored for this order yet. "
        "Predictions are generated automatically by n8n when an order is "
        "released, or you can trigger one via the `/webhooks/order-released` endpoint."
    )
    st.subheader("Raw Order Data")
    st.json(order_data)
    st.stop()

# ---------------------------------------------------------------------------
# Gauge + prediction summary
# ---------------------------------------------------------------------------

prob = float(prediction.get("delay_probability", 0.0))
risk_label = prob_to_risk_label(prob)

gauge_col, summary_col = st.columns([1, 2])

with gauge_col:
    render_risk_gauge(prob, title=f"{risk_label}")

with summary_col:
    st.subheader("Prediction Summary")

    root_cause = prediction.get("root_cause", "")
    confidence = prediction.get("confidence", "")
    delay_est = prediction.get("delay_minutes_estimate")

    if root_cause:
        st.markdown(f"**Root Cause:** {root_cause_to_display(root_cause)}")

    if confidence:
        badge_colour = confidence_badge_colour(confidence)
        st.markdown(
            f"**Confidence:** "
            f'<span style="background:{badge_colour};color:#fff;'
            f'padding:3px 10px;border-radius:12px;font-size:0.85rem;">'
            f"{confidence.title()}</span>",
            unsafe_allow_html=True,
        )

    if delay_est is not None:
        st.markdown(
            f"**Estimated Delay:** {minutes_to_display(int(delay_est))}"
        )

    st.divider()

    narrative = prediction.get("narrative", "").strip()
    if narrative:
        st.markdown("**Analysis:**")
        st.markdown(narrative)
    else:
        st.info("No narrative explanation available for this prediction.")

st.divider()

# ---------------------------------------------------------------------------
# SHAP charts — risk factors and mitigating factors side by side
# ---------------------------------------------------------------------------

risk_factors: list[dict] = prediction.get("top_risk_factors", [])
mitigating: list[dict] = prediction.get("mitigating_factors", [])

if risk_factors or mitigating:
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("🔺 Top Risk Factors")
        if risk_factors:
            render_shap_chart(risk_factors, title="", max_factors=8)
        else:
            st.info("No risk factor data.")

    with right_col:
        st.subheader("🔻 Mitigating Factors")
        if mitigating:
            render_shap_chart(mitigating, title="", max_factors=5)
        else:
            st.info("No mitigating factor data.")
else:
    st.info(
        "SHAP factor breakdown not available. "
        "Run the prediction endpoint to generate explanations."
    )
