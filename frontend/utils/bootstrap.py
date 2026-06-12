"""
frontend/utils/bootstrap.py
=============================
One-call session-state initialisation.

All Streamlit pages import and call `ensure_session_state()` at the top so
that any page can be visited directly (not just through app.py) and still
have a correctly initialised state.
"""
from __future__ import annotations

import os
import uuid

import streamlit as st


def ensure_session_state() -> None:
    """Idempotently initialise all required session-state keys."""
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = os.environ.get(
            "API_BASE_URL", "http://localhost:8000"
        )

    if "api_client" not in st.session_state:
        from services.api_client import MpcApiClient  # noqa: PLC0415

        st.session_state.api_client = MpcApiClient(st.session_state.api_base_url)

    if "chat_session_token" not in st.session_state:
        st.session_state.chat_session_token = str(uuid.uuid4())

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages: list[dict] = []

    if "selected_order_id" not in st.session_state:
        st.session_state.selected_order_id = None

    if "selected_order_data" not in st.session_state:
        st.session_state.selected_order_data = None
