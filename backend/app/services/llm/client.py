"""
backend/app/services/llm/client.py
=====================================
LLM routing layer with circuit-breaker failover.

Primary path:   OpenRouter API — OPENROUTER_MODEL      (reasoning)
                               or OPENROUTER_CODER_MODEL (structured output)
Model failover: On 429 from primary model, retries once with
                OPENROUTER_FALLBACK_MODEL before raising LLMRateLimitError.
                Rate limits do NOT trip the circuit breaker.
Fallback path:  Ollama local — OLLAMA_MODEL

Circuit-breaker rule: ≥3 consecutive circuit-eligible errors (auth failures,
connection errors, provider outages) OR a single timeout → flip provider to
Ollama for the remainder of the process lifetime.
HTTP 429 (rate limit) is NOT a circuit-trip condition.
Manual reset via reset_circuit().

Public API
----------
LLMClient.complete(messages, model_hint, stream) → str | AsyncGenerator[str]
LLMClient.reset_circuit()                        → None  (testing / forced reset)

Exception hierarchy
-------------------
LLMError               — base
  LLMRateLimitError    — HTTP 429 (both primary and fallback exhausted)
  LLMAuthError         — HTTP 401
  LLMPermissionError   — HTTP 403
  LLMModelNotFoundError— HTTP 404
  LLMTimeoutError      — httpx.TimeoutException
  LLMProviderError     — HTTP 5xx
  LLMConnectionError   — httpx.ConnectError / network unreachable
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Literal

import httpx

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)

MessageRole = Literal["system", "user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMAuthError(LLMError):
    pass


class LLMPermissionError(LLMError):
    pass


class LLMModelNotFoundError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


# HTTP status codes that increment the circuit-breaker error counter.
# 429 is intentionally excluded — rate limits are transient and should
# not permanently degrade the session to Ollama.
_CIRCUIT_ELIGIBLE_STATUSES = frozenset({401, 403, 404, 500, 502, 503, 504})


def _classify_http_error(exc: httpx.HTTPStatusError) -> LLMError:
    """Map an httpx HTTP error to a typed LLMError subclass."""
    status = exc.response.status_code
    if status == 429:
        return LLMRateLimitError(f"Rate limited by provider (HTTP 429): {exc}")
    if status == 401:
        return LLMAuthError(f"Invalid or expired API key (HTTP 401): {exc}")
    if status == 403:
        return LLMPermissionError(f"Access denied (HTTP 403): {exc}")
    if status == 404:
        return LLMModelNotFoundError(f"Model not found (HTTP 404): {exc}")
    if status >= 500:
        return LLMProviderError(f"Provider outage (HTTP {status}): {exc}")
    return LLMError(f"Unexpected HTTP error (HTTP {status}): {exc}")


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
            raise LLMTimeoutError(f"LLM timeout (provider={provider})")
        except httpx.ConnectError as exc:
            logger.warning("LLM connection error via %s: %s", provider, exc)
            self._count_circuit_error()
            raise LLMConnectionError(str(exc)) from exc
        except LLMRateLimitError:
            # Rate limits are not circuit-trip conditions — re-raise as-is.
            raise
        except LLMError:
            # Typed LLMError subclasses from _openrouter/_ollama — count and re-raise.
            self._count_circuit_error()
            raise
        except Exception as exc:
            logger.warning("LLM unexpected error via %s: %s", provider, exc)
            self._count_circuit_error()
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

    def _count_circuit_error(self) -> None:
        self._consecutive_errors += 1
        if self._consecutive_errors >= 3:
            self._trip()

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
        if model_hint == "coder":
            primary_model = self._settings.OPENROUTER_CODER_MODEL
        else:
            primary_model = self._settings.OPENROUTER_MODEL

        logger.info("[LLM] _openrouter called model=%s stream=%s msgs=%d",
                    primary_model, stream, len(messages))
        headers = {
            "Authorization": f"Bearer {self._settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mpc",
            "X-Title": "Manufacturing Process Copilot",
        }

        try:
            return await self._openrouter_call(headers, messages, primary_model, stream=stream)
        except LLMRateLimitError:
            fallback = self._settings.OPENROUTER_FALLBACK_MODEL
            logger.warning(
                "[LLM] primary model %s rate-limited — retrying with fallback %s",
                primary_model, fallback,
            )
            try:
                return await self._openrouter_call(headers, messages, fallback, stream=stream)
            except LLMRateLimitError:
                logger.error("[LLM] fallback model %s also rate-limited", fallback)
                raise

    async def _openrouter_call(
        self,
        headers: dict,
        messages: list[dict[str, str]],
        model: str,
        *,
        stream: bool,
    ) -> str | AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        if stream:
            return self._openrouter_stream(headers, payload)

        logger.info("[LLM] POST start → %s/chat/completions model=%s",
                    self._settings.OPENROUTER_BASE_URL, model)
        resp = await self._http.post(
            f"{self._settings.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        logger.info("[LLM] POST complete → status=%d content_length=%s",
                    resp.status_code,
                    resp.headers.get("content-length", "chunked"))
        logger.info("[LLM] resp.text preview: %s", resp.text[:200])
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _classify_http_error(exc) from exc
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
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise _classify_http_error(exc) from exc
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
