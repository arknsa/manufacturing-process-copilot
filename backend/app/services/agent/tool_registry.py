"""
backend/app/services/agent/tool_registry.py
==============================================
Maps tool name strings → async callable functions.

The registry is the only code that executes tool functions.
The agent calls the registry; it never calls tools directly.

Usage
-----
    registry = ToolRegistry(db)
    registry.register("get_order", get_production_order, SCHEMA)
    result_json = await registry.dispatch("get_order", {"order_id": "ORD-001"})
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Type alias: every tool is an async function that receives `db` plus
# keyword arguments extracted from the LLM's JSON and returns a dict.
ToolFn = Callable[..., Awaitable[dict[str, Any]]]


class ToolRegistry:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._tools: dict[str, ToolFn] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, func: ToolFn, schema: dict[str, Any]) -> None:
        """Add a tool to the registry.

        Args:
            name:   Tool name the LLM will use in its Action field.
            func:   Async callable — must accept ``db`` as a keyword arg plus
                    any parameters listed in ``schema["parameters"]``.
            schema: Dict with at minimum a ``"description"`` key; optionally
                    a ``"parameters"`` dict mapping param name → description.
        """
        self._tools[name] = func
        self._schemas[name] = schema

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute ``tool_name`` with ``arguments``; return JSON string.

        Always returns a valid JSON string — never raises. Unknown tools and
        tool-internal exceptions produce ``{"error": "..."}`` payloads.
        """
        func = self._tools.get(tool_name)
        if func is None:
            logger.warning("Agent requested unknown tool: %s", tool_name)
            return json.dumps(
                {
                    "error": f"Unknown tool '{tool_name}'.",
                    "available_tools": self.tool_names,
                }
            )
        try:
            result = await func(db=self._db, **arguments)
            return json.dumps(result, default=str)
        except TypeError as exc:
            logger.error("Tool %s called with wrong arguments %s: %s", tool_name, arguments, exc)
            return json.dumps({"error": f"Invalid arguments for tool '{tool_name}': {exc}"})
        except Exception as exc:
            logger.error("Tool %s raised: %s", tool_name, exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def tool_descriptions(self) -> str:
        """Return a markdown-formatted tool listing for inclusion in SYSTEM_PROMPT."""
        if not self._schemas:
            return "(no tools registered)"
        lines: list[str] = []
        for name, schema in self._schemas.items():
            desc = schema.get("description", "")
            lines.append(f"- **{name}**: {desc}")
            for param_name, param_desc in schema.get("parameters", {}).items():
                lines.append(f"    - `{param_name}`: {param_desc}")
        return "\n".join(lines)
