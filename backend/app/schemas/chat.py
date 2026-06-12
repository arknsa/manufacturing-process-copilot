"""
backend/app/schemas/chat.py
==============================
Pydantic schemas for the chat / copilot resource.

ChatMessageRequest  — inbound user message with session token.
ChatMessageResponse — outbound SSE frame or single response.
ChatSessionResponse — full session history returned by GET /chat/sessions/{token}.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_token: str = Field(..., min_length=1, max_length=100)
    stream: bool = True


class ToolCallDetail(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[str] = None


class ChatMessageResponse(BaseModel):
    session_token: str
    role: str
    content: str
    tool_calls: Optional[List[ToolCallDetail]] = None
    model_used: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    session_token: str
    messages: List[ChatMessageResponse]
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
