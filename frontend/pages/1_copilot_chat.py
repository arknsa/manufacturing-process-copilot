"""
frontend/pages/1_copilot_chat.py
==================================
Full-screen streaming chat interface for the Manufacturing Copilot.

Architecture:
  • st.session_state.chat_messages  — persisted list of {role, content} dicts
  • st.session_state.chat_session_token — UUID sent to /api/v1/chat/message
  • MpcApiClient.stream_chat() yields text chunks from the backend SSE stream
  • Chunks are appended to a st.empty() placeholder to simulate live typing
  • The sidebar provides session management and quick-prompt injection

SSE wire format (backend streaming.py):
    data: {"content": "..."}\n\n
    data: [DONE]\n\n
"""
from __future__ import annotations

import sys
import os
import uuid

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)

import streamlit as st

from utils.bootstrap import ensure_session_state
from components.chat_window import render_chat_messages

st.set_page_config(
    page_title="Copilot Chat | MPC",
    page_icon="💬",
    layout="wide",
)

ensure_session_state()
client = st.session_state.api_client

# ---------------------------------------------------------------------------
# Sidebar — session controls + example prompts
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 💬 Chat Session")
    token_preview = st.session_state.chat_session_token
    st.caption(f"Token: `{token_preview[:8]}…{token_preview[-4:]}`")

    st.divider()

    if st.button("🆕 New Session", use_container_width=True, key="new_session_btn"):
        # Best-effort delete on backend; ignore failure
        client.delete_session(st.session_state.chat_session_token)
        st.session_state.chat_session_token = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.rerun()

    if st.button("🗑️ Clear Display", use_container_width=True, key="clear_btn"):
        # Clears local display only — session history stays on the backend
        st.session_state.chat_messages = []
        st.rerun()

    st.divider()
    st.markdown("**Quick prompts:**")

    example_prompts = [
        "What orders are at high risk right now?",
        "Which machine has the highest delay rate this month?",
        "Summarise today's shift performance.",
        "What actions can reduce the current bottleneck?",
        "Why is our on-time delivery rate below target?",
    ]

    for prompt in example_prompts:
        if st.button(
            prompt,
            use_container_width=True,
            key=f"qp_{hash(prompt)}",
            help=prompt,
        ):
            st.session_state["_pending_prompt"] = prompt
            st.rerun()

    st.divider()
    healthy = client.health_check()
    if healthy:
        st.success("Backend: connected", icon="✅")
    else:
        st.error("Backend: unreachable", icon="❌")

# ---------------------------------------------------------------------------
# Main chat area header
# ---------------------------------------------------------------------------

st.title("💬 Manufacturing Copilot")
st.caption(
    "Ask about production orders, delay risks, machine performance, "
    "and recommended actions. Responses stream in real time."
)

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------

render_chat_messages(st.session_state.chat_messages)

# ---------------------------------------------------------------------------
# Resolve user input (direct chat_input OR sidebar quick-prompt)
# ---------------------------------------------------------------------------

pending: str | None = st.session_state.pop("_pending_prompt", None)
user_input: str | None = st.chat_input(
    "Ask about your production orders…",
    key="main_chat_input",
)

# Quick-prompt takes precedence over empty chat_input
active_input = pending or user_input

# ---------------------------------------------------------------------------
# Handle user turn + stream assistant response
# ---------------------------------------------------------------------------

if active_input:
    active_input = active_input.strip()
    if not active_input:
        st.stop()

    # 1. Append and display user message
    st.session_state.chat_messages.append({"role": "user", "content": active_input})
    with st.chat_message("user"):
        st.markdown(active_input)

    # 2. Stream assistant response into a live placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in client.stream_chat(
                active_input,
                st.session_state.chat_session_token,
            ):
                full_response += chunk
                # Trailing cursor while streaming
                placeholder.markdown(full_response + "▌")

            # Final render without cursor
            if full_response:
                placeholder.markdown(full_response)
            else:
                placeholder.warning("No response received from the backend.")
                full_response = "_No response received._"

        except Exception as exc:  # noqa: BLE001
            full_response = f"_Error communicating with backend: {exc}_"
            placeholder.error(full_response)

    # 3. Persist assistant message
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": full_response}
    )
