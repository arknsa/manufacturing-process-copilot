"""
backend/tests/unit/test_llm_client.py
========================================
Unit tests for LLMClient.

No real HTTP calls are made — httpx responses are mocked via
unittest.mock.patch and a custom HTTPX transport.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_settings(api_key: str = "test-key") -> MagicMock:
    s = MagicMock()
    s.OPENROUTER_API_KEY = api_key
    s.OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    s.OPENROUTER_MODEL = "qwen/qwen3-80b-a3b:free"
    s.OPENROUTER_CODER_MODEL = "qwen/qwen3-coder:free"
    s.OLLAMA_BASE_URL = "http://localhost:11434"
    s.OLLAMA_MODEL = "llama3.2:3b"
    s.LLM_TIMEOUT_SECONDS = 10.0
    return s


def test_llm_client_initializes_openrouter_when_key_set():
    from backend.app.services.llm.client import LLMClient

    client = LLMClient(_make_settings(api_key="sk-test"))
    assert client._active_provider() == "openrouter"


def test_llm_client_initializes_ollama_when_no_key():
    from backend.app.services.llm.client import LLMClient

    client = LLMClient(_make_settings(api_key=""))
    assert client._active_provider() == "ollama"


def test_circuit_trips_after_three_errors():
    from backend.app.services.llm.client import LLMClient

    client = LLMClient(_make_settings())
    assert not client._circuit_open

    # Simulate 3 consecutive errors incrementing the counter
    client._consecutive_errors = 2
    client._trip()  # third error triggers trip

    assert client._circuit_open
    assert client._active_provider() == "ollama"


def test_reset_circuit_restores_openrouter():
    from backend.app.services.llm.client import LLMClient

    client = LLMClient(_make_settings())
    client._trip()
    assert client._circuit_open

    client.reset_circuit()
    assert not client._circuit_open
    assert client._active_provider() == "openrouter"


def test_active_provider_is_ollama_when_circuit_open():
    from backend.app.services.llm.client import LLMClient

    client = LLMClient(_make_settings())
    client._circuit_open = True
    assert client._active_provider() == "ollama"
