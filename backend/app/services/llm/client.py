"""
backend/app/services/llm/client.py
=====================================
LLM routing layer with circuit-breaker failover.

Primary path:   OpenRouter API — qwen/qwen3-80b-a3b:free (reasoning)
                               or qwen/qwen3-coder:free  (structured output)
Fallback path:  Ollama local — llama3.2:3b

Circuit-breaker rule: ≥3 consecutive OpenRouter errors OR a single timeout
>= LLM_TIMEOUT_SECONDS → flip provider to Ollama for the remainder of the
process lifetime. Manual reset via reset_circuit().

Public API
----------
LLMClient.complete(messages, model_hint, stream) → str | AsyncGenerator[str]
LLMClient.reset_circuit()                        → None  (testing / forced reset)
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Literal

import httpx

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)

MessageRole = Literal["system", "user", "assistant", "tool"]


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider: Literal["openrouter", "ollama"] = (
            "ollama" if not settings.OPENROUTER_API_KEY else "openrouter"
        )
        self._consecutive_errors = 0
        self._circuit_open = False
        self._http = httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS + 5.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model_hint: Literal["reasoning", "coder"] = "reasoning",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Call the active provider. Returns a string or async generator if stream=True."""
        provider = self._active_provider()
        try:
            if provider == "openrouter":
                result = await self._openrouter(messages, model_hint=model_hint, stream=stream)
            else:
                result = await self._ollama(messages, stream=stream)
            self._consecutive_errors = 0
            return result
        except httpx.TimeoutException:
            logger.warning("LLM timeout via %s — tripping circuit", provider)
            self._trip()
            raise LLMError(f"LLM timeout (provider={provider})")
        except Exception as exc:
            logger.warning("LLM error via %s: %s", provider, exc)
            self._consecutive_errors += 1
            if self._consecutive_errors >= 3:
                self._trip()
            raise LLMError(str(exc)) from exc

    def reset_circuit(self) -> None:
        self._circuit_open = False
        self._consecutive_errors = 0
        self._provider = (
            "ollama" if not self._settings.OPENROUTER_API_KEY else "openrouter"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_provider(self) -> str:
        if self._circuit_open or not self._settings.OPENROUTER_API_KEY:
            return "ollama"
        return self._provider

    def _trip(self) -> None:
        if not self._circuit_open:
            logger.error("Circuit tripped — switching to Ollama for this session")
        self._circuit_open = True
        self._provider = "ollama"

    # ------------------------------------------------------------------
    # OpenRouter
    # ------------------------------------------------------------------

    async def _openrouter(
        self,
        messages: list[dict[str, str]],
        *,
        model_hint: str,
        stream: bool,
    ) -> str | AsyncGenerator[str, None]:
        model = (
            self._settings.OPENROUTER_CODER_MODEL
            if model_hint == "coder"
            else self._settings.OPENROUTER_MODEL
        )
        logger.info("[LLM] _openrouter called model=%s stream=%s msgs=%d",
                    model, stream, len(messages))
        headers = {
            "Authorization": f"Bearer {self._settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mpc",
            "X-Title": "Manufacturing Process Copilot",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        if stream:
            return self._openrouter_stream(headers, payload)

        logger.info("[LLM] POST start → %s/chat/completions", self._settings.OPENROUTER_BASE_URL)
        resp = await self._http.post(
            f"{self._settings.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        logger.info("[LLM] POST complete → status=%d content_length=%s",
                    resp.status_code,
                    resp.headers.get("content-length", "chunked"))
        logger.info("[LLM] resp.text preview: %s", resp.text[:200])
        resp.raise_for_status()
        logger.info("[LLM] raise_for_status passed")
        data = resp.json()
        logger.info("[LLM] resp.json() parsed — choices=%d", len(data.get("choices", [])))
        content = data["choices"][0]["message"]["content"]
        logger.info("[LLM] content extracted length=%s type=%s",
                    len(content) if content else 0, type(content).__name__)
        return content

    async def _openrouter_stream(
        self, headers: dict, payload: dict
    ) -> AsyncGenerator[str, None]:
        async with self._http.stream(
            "POST",
            f"{self._settings.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    async def _ollama(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
    ) -> str | AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "model": self._settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": 0.3, "num_predict": 1024},
        }

        if stream:
            return self._ollama_stream(payload)

        resp = await self._http.post(
            f"{self._settings.OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    async def _ollama_stream(self, payload: dict) -> AsyncGenerator[str, None]:
        async with self._http.stream(
            "POST",
            f"{self._settings.OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

    async def aclose(self) -> None:
        await self._http.aclose()
