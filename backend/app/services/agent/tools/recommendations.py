"""
backend/app/services/agent/tools/recommendations.py
=====================================================
Agent tools for creating and managing recommendations.

Tools
-----
create_recommendation          — write a new recommendation row
get_recommendations            — list recommendations filtered by status
update_recommendation_status   — move a recommendation through its lifecycle
"""

from __future__ import annotations

import logging
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.recommendation import Recommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

CREATE_RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "description": (
        "Create a new actionable recommendation. The agent calls this when it "
        "identifies a risk or opportunity that requires supervisor attention."
    ),
    "parameters": {
        "title": "str — short summary (≤ 200 chars)",
        "description": "str — detailed explanation and suggested action",
        "category": (
            "str — one of: schedule_change | resource_reallocation | "
            "maintenance | escalation | other"
        ),
        "urgency": "str (optional, default 'medium') — low | medium | high | critical",
        "order_id": "str (optional) — UUID of the related production order",
    },
}

GET_RECOMMENDATIONS_SCHEMA: dict[str, Any] = {
    "description": "List recommendations, optionally filtered by status.",
    "parameters": {
        "status": (
            "str (optional, default 'open') — open | acknowledged | actioned | dismissed | all"
        ),
        "limit": "int (optional, default 10)",
    },
}

UPDATE_RECOMMENDATION_STATUS_SCHEMA: dict[str, Any] = {
    "description": (
        "Update the status of a recommendation (acknowledge, action, or dismiss it). "
        "Use the recommendation UUID from get_recommendations."
    ),
    "parameters": {
        "recommendation_id": "str — UUID of the recommendation",
        "status": "str — acknowledged | actioned | dismissed",
        "actioned_by": "str (optional) — name of the person taking the action",
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rec_to_dict(r: Recommendation) -> dict[str, Any]:
    return {
        "id": str(r.id),
        "title": r.title,
        "description": r.description,
        "category": r.category,
        "urgency": r.urgency,
        "order_id": str(r.order_id) if r.order_id else None,
        "bottleneck_id": str(r.bottleneck_id) if r.bottleneck_id else None,
        "status": r.status,
        "actioned_by": r.actioned_by,
        "actioned_at": r.actioned_at.isoformat() if r.actioned_at else None,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def create_recommendation(
    db: AsyncSession,
    title: str,
    description: str,
    category: str,
    urgency: str = "medium",
    order_id: str | None = None,
) -> dict[str, Any]:
    """Insert a new recommendation and return its id."""
    valid_categories = {
        "schedule_change", "resource_reallocation",
        "maintenance", "escalation", "other",
    }
    if category not in valid_categories:
        return {"error": f"Invalid category '{category}'. Choose from: {valid_categories}"}

    valid_urgencies = {"low", "medium", "high", "critical"}
    if urgency not in valid_urgencies:
        urgency = "medium"

    order_uuid: _uuid_mod.UUID | None = None
    if order_id:
        try:
            order_uuid = _uuid_mod.UUID(order_id)
        except ValueError:
            return {"error": f"Invalid UUID for order_id: '{order_id}'"}

    rec = Recommendation(
        title=title[:200],
        description=description,
        category=category,
        urgency=urgency,
        order_id=order_uuid,
        status="open",
    )
    db.add(rec)
    await db.flush()  # Populate rec.id without committing (commit on request teardown)

    logger.info("Created recommendation %s: %s", rec.id, title)
    return {"created": True, "recommendation_id": str(rec.id), "title": title}


async def get_recommendations(
    db: AsyncSession,
    status: str = "open",
    limit: int = 10,
) -> dict[str, Any]:
    """Return recommendations, optionally filtered by status."""
    stmt = (
        select(Recommendation)
        .order_by(
            Recommendation.urgency.desc(),
            Recommendation.created_at.desc(),
        )
        .limit(limit)
    )
    if status != "all":
        stmt = stmt.where(Recommendation.status == status)

    result = await db.execute(stmt)
    recs = result.scalars().all()

    return {
        "recommendations": [_rec_to_dict(r) for r in recs],
        "count": len(recs),
        "status_filter": status,
    }


async def update_recommendation_status(
    db: AsyncSession,
    recommendation_id: str,
    status: str,
    actioned_by: str | None = None,
) -> dict[str, Any]:
    """Transition a recommendation's status."""
    valid_transitions = {"acknowledged", "actioned", "dismissed"}
    if status not in valid_transitions:
        return {
            "error": f"Invalid status '{status}'. Choose from: {valid_transitions}"
        }

    try:
        rec_uuid = _uuid_mod.UUID(recommendation_id)
    except ValueError:
        return {"error": f"Invalid UUID: '{recommendation_id}'"}

    result = await db.execute(
        select(Recommendation).where(Recommendation.id == rec_uuid)
    )
    rec = result.scalar_one_or_none()
    if rec is None:
        return {"error": f"Recommendation '{recommendation_id}' not found."}

    rec.status = status
    if actioned_by:
        rec.actioned_by = actioned_by
    if status == "actioned":
        rec.actioned_at = datetime.now(timezone.utc)

    await db.flush()
    return {
        "updated": True,
        "recommendation_id": recommendation_id,
        "new_status": status,
        "actioned_by": actioned_by,
    }
