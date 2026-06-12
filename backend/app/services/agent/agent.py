"""
backend/app/services/agent/agent.py
======================================
CopilotAgent — stateless ReAct (Reason + Act) loop.

Each call to run() is a single conversation turn:
  1. Load session memory (last N messages + summary).
  2. Save the user's message.
  3. Enter the ReAct loop (max MAX_ITERATIONS rounds):
       a. Build prompt with memory context and tool descriptions.
       b. Call the LLM — expect Thought / Action / Action Input.
       c. Parse the response.
       d. If Action == FINAL_ANSWER → save assistant reply, yield text, done.
       e. Otherwise dispatch the named tool, append the observation, repeat.
  4. If MAX_ITERATIONS exhausted without FINAL_ANSWER, yield a fallback.

The agent itself is stateless — all state lives in the database (SessionMemory).

Factory function
----------------
build_registry(db) → ToolRegistry   registers all 10 tools from tools/*
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.agent.memory import SessionMemory
from backend.app.services.agent.tool_registry import ToolRegistry
from backend.app.services.agent.tools import analytics, orders, predictions, recommendations
from backend.app.services.llm.client import LLMClient
from backend.app.services.llm.prompts import OBSERVATION_PROMPT, SYSTEM_PROMPT, TOOL_SELECTION_PROMPT

logger = logging.getLogger(__name__)

# ReAct response parser — lenient: allows Thought to span multiple lines
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(\S+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.DOTALL | re.IGNORECASE)

_FALLBACK = (
    "I wasn't able to retrieve the information needed to answer your question. "
    "Please try rephrasing or provide more specific details such as an order number."
)


# ---------------------------------------------------------------------------
# Registry factory — wires all 10 tools
# ---------------------------------------------------------------------------

def build_registry(db: AsyncSession) -> ToolRegistry:
    """Create and populate a ToolRegistry with all domain tools."""
    registry = ToolRegistry(db)

    # Orders
    registry.register("get_production_order", orders.get_production_order, orders.GET_PRODUCTION_ORDER_SCHEMA)
    registry.register("get_active_orders", orders.get_active_orders, orders.GET_ACTIVE_ORDERS_SCHEMA)
    registry.register("get_orders_at_risk", orders.get_orders_at_risk, orders.GET_ORDERS_AT_RISK_SCHEMA)

    # Predictions
    registry.register("get_delay_prediction", predictions.get_delay_prediction, predictions.GET_DELAY_PREDICTION_SCHEMA)
    registry.register("get_risk_summary", predictions.get_risk_summary, predictions.GET_RISK_SUMMARY_SCHEMA)
    registry.register("get_feature_explanation", predictions.get_feature_explanation, predictions.GET_FEATURE_EXPLANATION_SCHEMA)

    # Analytics
    registry.register("get_machine_history", analytics.get_machine_history, analytics.GET_MACHINE_HISTORY_SCHEMA)
    registry.register("get_bottlenecks", analytics.get_bottlenecks, analytics.GET_BOTTLENECKS_SCHEMA)
    registry.register("get_shift_summary", analytics.get_shift_summary, analytics.GET_SHIFT_SUMMARY_SCHEMA)
    registry.register("get_kpi_dashboard", analytics.get_kpi_dashboard, analytics.GET_KPI_DASHBOARD_SCHEMA)

    # Recommendations
    registry.register("create_recommendation", recommendations.create_recommendation, recommendations.CREATE_RECOMMENDATION_SCHEMA)
    registry.register("get_recommendations", recommendations.get_recommendations, recommendations.GET_RECOMMENDATIONS_SCHEMA)
    registry.register("update_recommendation_status", recommendations.update_recommendation_status, recommendations.UPDATE_RECOMMENDATION_STATUS_SCHEMA)

    return registry


# ---------------------------------------------------------------------------
# ReAct parser
# ---------------------------------------------------------------------------

def _parse_react(text: str) -> tuple[str, str, str]:
    """Extract (thought, action, action_input) from an LLM response.

    Falls back to treating the entire text as a FINAL_ANSWER when the
    expected format is not detected — preventing agent stalls.
    """
    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    input_m = _INPUT_RE.search(text)

    if action_m is None:
        return ("", "FINAL_ANSWER", text.strip())

    thought = thought_m.group(1).strip() if thought_m else ""
    action = action_m.group(1).strip()
    action_input = input_m.group(1).strip() if input_m else ""

    return thought, action, action_input


# ---------------------------------------------------------------------------
# CopilotAgent
# ---------------------------------------------------------------------------

class CopilotAgent:
    MAX_ITERATIONS = 5
    MAX_MEMORY_MESSAGES = 10

    def __init__(
        self,
        db: AsyncSession,
        llm: LLMClient,
        registry: ToolRegistry,
    ) -> None:
        self._db = db
        self._llm = llm
        self._registry = registry
        self._memory = SessionMemory(db, llm)

    async def run(
        self, message: str, session_token: str
    ) -> AsyncGenerator[str, None]:
        """Execute one turn of the ReAct loop; yield the final answer."""
        # 1. Load history & compress if needed
        logger.info("[AGENT] compress start for session %s", session_token[:8])
        await self._memory.compress(session_token)
        history = await self._memory.load(session_token, self.MAX_MEMORY_MESSAGES)
        logger.info("[AGENT] history loaded: %d messages", len(history))

        # 2. Persist the user message
        await self._memory.save(session_token, "user", message)

        # 3. Build the initial message list for the LLM
        llm_messages = self._build_messages(message, history)

        # 4. ReAct loop
        final_answer = _FALLBACK
        tool_calls_log: list[dict[str, Any]] = []

        for iteration in range(self.MAX_ITERATIONS):
            logger.info("[AGENT] iteration %d/%d starting", iteration + 1, self.MAX_ITERATIONS)

            try:
                response = await self._llm.complete(llm_messages)
            except Exception as exc:
                logger.error("LLM call failed on iteration %d: %s", iteration + 1, exc)
                final_answer = (
                    "I'm having trouble reaching the AI model right now. "
                    "Please try again in a moment."
                )
                break

            logger.info(
                "[AGENT] LLM responded on iteration %d, length=%s chars",
                iteration + 1,
                len(response) if isinstance(response, str) else "stream",
            )

            # Ensure response is a plain string (not an async generator)
            if not isinstance(response, str):
                try:
                    parts = []
                    async for chunk in response:
                        parts.append(chunk)
                    response = "".join(parts)
                except Exception:
                    response = ""

            thought, action, action_input = _parse_react(response)
            logger.info("[AGENT] parsed → action=%r input_preview=%s", action, action_input[:80])

            if action.upper() == "FINAL_ANSWER":
                logger.info("[AGENT] FINAL_ANSWER reached at iteration %d", iteration + 1)
                final_answer = action_input or response
                break

            # Dispatch tool
            try:
                arguments = json.loads(action_input) if action_input else {}
            except json.JSONDecodeError:
                arguments = {}

            logger.info("[AGENT] dispatching tool %r with args %s", action, list(arguments.keys()))
            tool_result = await self._registry.dispatch(action, arguments)
            logger.info("[AGENT] tool %r returned %d chars", action, len(tool_result))
            tool_calls_log.append(
                {"tool": action, "arguments": arguments, "result": tool_result}
            )

            # Append assistant's reasoning + observation to the message list
            llm_messages.append({"role": "assistant", "content": response})
            observation = OBSERVATION_PROMPT.format(
                tool_name=action, tool_result=tool_result
            )
            llm_messages.append({"role": "user", "content": observation})

        # 5. Persist the assistant response
        await self._memory.save(
            session_token,
            "assistant",
            final_answer,
            tool_calls=tool_calls_log if tool_calls_log else None,
        )

        # 6. Yield the final answer
        logger.info("[AGENT] yielding final answer (%d chars)", len(final_answer))
        yield final_answer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self, user_message: str, history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        system_content = SYSTEM_PROMPT.format(
            tool_descriptions=self._registry.tool_descriptions()
        )
        memory_context = self._format_history(history)
        react_content = TOOL_SELECTION_PROMPT.format(
            user_message=user_message,
            memory_length=len([h for h in history if h["role"] != "system"]),
            memory_context=memory_context,
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": react_content},
        ]

    @staticmethod
    def _format_history(history: list[dict[str, str]]) -> str:
        if not history:
            return "(no previous messages in this session)"
        lines = []
        for msg in history:
            role = msg["role"].upper()
            # Truncate very long messages to keep the prompt manageable
            content = msg["content"][:300]
            if len(msg["content"]) > 300:
                content += "…"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
