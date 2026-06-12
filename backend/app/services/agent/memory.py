"""
backend/app/services/agent/memory.py
=======================================
SessionMemory — reads and writes chat_sessions + chat_messages tables.

Public API
----------
load(session_token, max_messages)    → list[dict]   (role, content pairs)
save(session_token, role, content)   → None
compress(session_token)              → None         (LLM-assisted truncation)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.chat_message import ChatMessage
from backend.app.db.models.chat_session import ChatSession
from backend.app.services.llm.client import LLMClient

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # rough estimation heuristic
_KEEP_RECENT = 5       # messages retained after compression


class SessionMemory:
    def __init__(self, db: AsyncSession, llm: LLMClient) -> None:
        self._db = db
        self._llm = llm

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def load(
        self, session_token: str, max_messages: int = 10
    ) -> list[dict[str, str]]:
        """Return the last ``max_messages`` messages for the session as dicts."""
        session = await self._get_or_create_session(session_token)

        # Fetch newest N, then reverse for chronological order
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(max_messages)
        )
        result = await self._db.execute(stmt)
        messages = list(reversed(result.scalars().all()))

        history: list[dict[str, str]] = []
        # Prepend session summary if one exists
        if session.summary:
            history.append(
                {"role": "system", "content": f"[Earlier context summary]: {session.summary}"}
            )
        history.extend({"role": m.role, "content": m.content} for m in messages)
        return history

    async def save(
        self,
        session_token: str,
        role: str,
        content: str,
        *,
        tool_calls: list | None = None,
        tool_results: list | None = None,
        model_used: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Append a message row to the session."""
        session = await self._get_or_create_session(session_token)

        msg = ChatMessage(
            session_id=session.id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._db.add(msg)
        await self._db.flush()

        # Touch updated_at on the session
        session.updated_at = datetime.now(timezone.utc)
        await self._db.flush()

    async def compress(self, session_token: str) -> None:
        """Summarise old messages with the LLM and truncate them."""
        session = await self._get_or_create_session(session_token)

        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.created_at)
        )
        result = await self._db.execute(stmt)
        all_messages = result.scalars().all()

        estimated_tokens = sum(len(m.content) for m in all_messages) // _CHARS_PER_TOKEN
        if estimated_tokens <= session.token_budget:
            return  # No compression needed

        if len(all_messages) <= _KEEP_RECENT:
            return  # Not enough messages to compress

        to_summarize = all_messages[:-_KEEP_RECENT]
        convo = "\n".join(
            f"{m.role.upper()}: {m.content[:500]}" for m in to_summarize
        )

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Summarise the following factory-copilot conversation in 2–3 sentences. "
                    "Preserve key facts: order numbers, risk levels, decisions made."
                ),
            },
            {"role": "user", "content": convo},
        ]

        try:
            summary_text = await self._llm.complete(summary_messages)
        except Exception as exc:
            logger.warning("LLM compression failed: %s", exc)
            return  # Never crash the agent over compression failure

        # Update session summary
        session.summary = (
            (session.summary or "")
            + ("\n\n" if session.summary else "")
            + str(summary_text)
        )

        # Delete the summarised messages
        old_ids = [m.id for m in to_summarize]
        await self._db.execute(
            delete(ChatMessage).where(ChatMessage.id.in_(old_ids))
        )
        await self._db.flush()
        logger.info(
            "Compressed %d messages for session %s", len(old_ids), session_token
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_or_create_session(self, session_token: str) -> ChatSession:
        """Return existing session or create a new one."""
        stmt = select(ChatSession).where(ChatSession.session_token == session_token)
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            session = ChatSession(session_token=session_token)
            self._db.add(session)
            await self._db.flush()

        return session
