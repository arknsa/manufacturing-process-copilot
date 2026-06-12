"""
backend/app/schemas/recommendations.py
=========================================
Pydantic schemas for agent-generated recommendations.

RecommendationResponse     — outbound representation of a recommendation.
RecommendationStatusUpdate — PATCH body for acknowledging or actioning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    id: uuid.UUID
    title: str
    description: str
    category: str
    urgency: str
    order_id: Optional[uuid.UUID]
    bottleneck_id: Optional[uuid.UUID]
    status: str
    actioned_by: Optional[str]
    actioned_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RecommendationStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(acknowledged|actioned|dismissed)$")
    actioned_by: Optional[str] = None
