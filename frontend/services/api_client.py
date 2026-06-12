"""
frontend/services/api_client.py
=================================
Typed wrapper around the Manufacturing Process Copilot REST API.

This is the ONLY file that constructs URLs or sets request headers.
All pages and components import and call methods on MpcApiClient.
Returns plain dicts/lists — no Pydantic in the frontend.

SSE wire format (from backend streaming.py):
    data: {"content": "some text chunk"}\\n\\n
    data: [DONE]\\n\\n
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8000"
_TIMEOUT = 10        # seconds — regular requests
_STREAM_TIMEOUT = 90  # seconds — SSE streaming (agent can take time to reason)


class MpcApiClient:
    """Single requests.Session shared across pages via st.session_state.

    All methods return None / empty list on any network or HTTP error so
    pages can check ``if result:`` instead of catching exceptions.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True when the backend is up and responding."""
        try:
            r = self._session.get(
                f"{self.base_url}/health", timeout=3
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    def is_ready(self) -> dict[str, Any] | None:
        try:
            r = self._session.get(f"{self.base_url}/ready", timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.warning("is_ready: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_today_orders(self) -> list[dict]:
        """GET /api/v1/orders/today — today's orders sorted by planned_start."""
        try:
            r = self._session.get(
                f"{self.base_url}/api/v1/orders/today", timeout=_TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.error("get_today_orders: %s", exc)
            return []

    def create_order(self, order: dict) -> dict | None:
        """POST /api/v1/orders/ — create a new production order."""
        try:
            r = self._session.post(
                f"{self.base_url}/api/v1/orders/",
                json=order,
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.error("create_order: %s", exc)
            return None

    def update_order_status(
        self, order_id: str, status: str, notes: str | None = None
    ) -> dict | None:
        """PATCH /api/v1/orders/{id}/status."""
        payload: dict[str, Any] = {"status": status}
        if notes is not None:
            payload["notes"] = notes
        try:
            r = self._session.patch(
                f"{self.base_url}/api/v1/orders/{order_id}/status",
                json=payload,
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.error("update_order_status: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict(self, features: dict) -> dict | None:
        """POST /api/v1/predictions/delay — single order DelayPrediction."""
        try:
            r = self._session.post(
                f"{self.base_url}/api/v1/predictions/delay",
                json=features,
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.error("predict: %s", exc)
            return None

    def predict_batch(self, orders: list[dict]) -> list[dict]:
        """POST /api/v1/predictions/delay/batch — up to 100 orders."""
        try:
            r = self._session.post(
                f"{self.base_url}/api/v1/predictions/delay/batch",
                json={"orders": orders},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("predictions", [])
        except requests.RequestException as exc:
            logger.error("predict_batch: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict | None:
        """GET /api/v1/models/current — active model metadata."""
        try:
            r = self._session.get(
                f"{self.base_url}/api/v1/models/current", timeout=_TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.error("get_model_info: %s", exc)
            return None

    def get_feature_importance(self) -> list[dict]:
        """GET /api/v1/models/feature-importance — global SHAP ranking."""
        try:
            r = self._session.get(
                f"{self.base_url}/api/v1/models/feature-importance",
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.error("get_feature_importance: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Chat — SSE streaming
    # ------------------------------------------------------------------

    def stream_chat(
        self,
        message: str,
        session_token: str,
    ) -> Iterator[str]:
        """POST /api/v1/chat/message with stream=True, yield text chunks.

        Parses the SSE format emitted by backend/services/llm/streaming.py:
            data: {"content": "..."}\n\n
            data: [DONE]\n\n

        Yields each non-empty content string.  On network failure yields a
        single error string so the caller always receives something renderable.
        """
        payload = {
            "message": message,
            "session_token": session_token,
            "stream": True,
        }
        try:
            with self._session.post(
                f"{self.base_url}/api/v1/chat/message",
                json=payload,
                stream=True,
                timeout=_STREAM_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line: str = raw if isinstance(raw, str) else raw.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]  # strip "data: " prefix
                    if payload_str.strip() == "[DONE]":
                        return
                    try:
                        data = json.loads(payload_str)
                        content: str = data.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass
        except requests.RequestException as exc:
            logger.error("stream_chat: %s", exc)
            yield f"\n\n_[Connection error: {exc}]_"

    def chat_sync(self, message: str, session_token: str) -> str:
        """Non-streaming fallback — returns full response as a string."""
        payload = {
            "message": message,
            "session_token": session_token,
            "stream": False,
        }
        try:
            r = self._session.post(
                f"{self.base_url}/api/v1/chat/message",
                json=payload,
                timeout=_STREAM_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("content", "")
        except requests.RequestException as exc:
            logger.error("chat_sync: %s", exc)
            return f"_[Error: {exc}]_"

    def get_session_history(self, session_token: str) -> dict | None:
        """GET /api/v1/chat/sessions/{token}."""
        try:
            r = self._session.get(
                f"{self.base_url}/api/v1/chat/sessions/{session_token}",
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            logger.error("get_session_history: %s", exc)
            return None

    def delete_session(self, session_token: str) -> bool:
        """DELETE /api/v1/chat/sessions/{token} — returns True on 204."""
        try:
            r = self._session.delete(
                f"{self.base_url}/api/v1/chat/sessions/{session_token}",
                timeout=_TIMEOUT,
            )
            return r.status_code == 204
        except requests.RequestException as exc:
            logger.error("delete_session: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._session.close()
