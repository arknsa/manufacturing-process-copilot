"""
frontend/components/chat_window.py
=====================================
Reusable Streamlit chat message renderer.

render_chat_messages() is intentionally thin — it delegates to
st.chat_message() so Streamlit handles the avatar, alignment, and
background colour automatically.

Tool calls are collapsed into expanders so they don't clutter the
conversation flow but are still inspectable.
"""
from __future__ import annotations

import json

import streamlit as st


def render_chat_messages(messages: list[dict]) -> None:
    """Render a list of ``{role, content}`` dicts as chat bubbles.

    Recognised roles:
      • ``"user"``      — right-aligned bubble
      • ``"assistant"`` — left-aligned bubble with bot avatar
      • ``"tool"``      — collapsed JSON expander (tool call / result)

    Unknown roles fall back to ``"assistant"`` styling.
    """
    for msg in messages:
        role: str = msg.get("role", "assistant")
        content: str = msg.get("content", "")

        if role == "tool":
            _render_tool_message(content, msg.get("tool_name", "Tool"))
            continue

        display_role = role if role in ("user", "assistant") else "assistant"
        with st.chat_message(display_role):
            st.markdown(content)


def _render_tool_message(content: str, tool_name: str = "Tool") -> None:
    """Collapse a tool call/result into a labelled expander."""
    with st.expander(f"🔧 {tool_name} result", expanded=False):
        try:
            parsed = json.loads(content)
            st.json(parsed)
        except (json.JSONDecodeError, TypeError):
            st.code(content, language="text")
