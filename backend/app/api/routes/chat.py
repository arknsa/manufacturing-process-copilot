"""
backend/app/api/routes/chat.py
================================
Copilot chat endpoints.

POST   /api/v1/chat/message             — send a message; response is SSE stream
                                          (or plain JSON when stream=False)
GET    /api/v1/chat/sessions/{token}    — full session history
DELETE /api/v1/chat/sessions/{token}    — clear session (204 No Content)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import get_agent, get_db
from backend.app.db.models.chat_message import ChatMessage
from backend.app.db.models.chat_session import ChatSession
from backend.app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
)
from backend.app.services.agent.agent import CopilotAgent
from backend.app.services.llm.streaming import stream_llm_response

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/message")
async def send_message(
    req: ChatMessageRequest,
    agent: CopilotAgent = Depends(get_agent),
):
    """Send a user message to the ReAct agent.

    With stream=True (default): returns SSE — data frames containing JSON
    ``{"content": "..."}`` chunks, terminated by ``data: [DONE]``.

    With stream=False: returns a plain JSON object.
    """
    if req.stream:
        async def _sse():
            async for chunk in stream_llm_response(
                agent.run(req.message, req.session_token)
            ):
                yield chunk

        return StreamingResponse(
            _sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming fallback — collect and return JSON
    chunks: list[str] = []
    async for text in agent.run(req.message, req.session_token):
        chunks.append(text)

    return {
        "session_token": req.session_token,
        "role": "assistant",
        "content": "".join(chunks),
    }


@router.get("/sessions/{session_token}", response_model=ChatSessionResponse)
async def get_session(
    session_token: str,
    db: AsyncSession = Depends(get_db),
) -> ChatSessionResponse:
    """Return the full message history for a chat session."""
    sess_result = await db.execute(
        select(ChatSession).where(ChatSession.session_token == session_token)
    )
    session = sess_result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at)
    )
    messages = msg_result.scalars().all()

    return ChatSessionResponse(
        session_token=session.session_token,
        messages=[
            ChatMessageResponse(
                session_token=session.session_token,
                role=m.role,
                content=m.content,
                tool_calls=None,
                model_used=m.model_used,
                input_tokens=m.input_tokens,
                output_tokens=m.output_tokens,
                created_at=m.created_at,
            )
            for m in messages
        ],
        summary=session.summary,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete("/sessions/{session_token}")
async def delete_session(
    session_token: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Delete a chat session and all its messages (cascade via FK)."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.session_token == session_token)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    return Response(status_code=204)
