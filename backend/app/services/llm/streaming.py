"""
backend/app/services/llm/streaming.py
========================================
Server-Sent Events (SSE) formatting utilities for FastAPI streaming responses.

stream_llm_response() wraps any async string generator (from LLMClient.stream())
into properly formatted SSE chunks that Streamlit's event-source reader can parse.

SSE wire format per chunk:
    data: {"content": "..."}\n\n

Terminal sentinel:
    data: [DONE]\n\n
"""

from __future__ import annotations

import json
from typing import AsyncGenerator


async def stream_llm_response(
    generator: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Convert raw LLM text chunks into SSE-formatted data frames."""
    try:
        async for chunk in generator:
            if chunk:
                payload = json.dumps({"content": chunk})
                yield f"data: {payload}\n\n"
    except Exception as exc:
        error_payload = json.dumps({"error": str(exc)})
        yield f"data: {error_payload}\n\n"
    finally:
        yield "data: [DONE]\n\n"


async def stream_text(text: str, chunk_size: int = 20) -> AsyncGenerator[str, None]:
    """Simulate streaming for a pre-computed string (used in fallback / testing)."""
    words = text.split()
    buffer: list[str] = []
    for word in words:
        buffer.append(word)
        if len(buffer) >= chunk_size:
            yield " ".join(buffer) + " "
            buffer = []
    if buffer:
        yield " ".join(buffer)
