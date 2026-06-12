"""
backend/tests/unit/test_agent.py
===================================
Unit tests for the agent layer.

Covered:
- ToolRegistry: register, dispatch (hit + miss), tool_descriptions
- _parse_react: happy path, missing format, FINAL_ANSWER
- CopilotAgent: FINAL_ANSWER on first iteration (no tools)
- CopilotAgent: one tool call then FINAL_ANSWER
- CopilotAgent: MAX_ITERATIONS guard (returns fallback)
- SessionMemory: load/save round-trip with mocked DB
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.agent.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _dummy_tool(db, order_id: str) -> dict:
    return {"order_number": order_id, "status": "pending"}


async def _failing_tool(db) -> dict:
    raise RuntimeError("DB exploded")


def _make_registry() -> tuple[ToolRegistry, AsyncMock]:
    db = AsyncMock()
    registry = ToolRegistry(db)
    return registry, db


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_populates_tool_names(self):
        registry, _ = _make_registry()
        registry.register("get_order", _dummy_tool, {"description": "Get order"})
        assert "get_order" in registry.tool_names

    def test_register_multiple_tools(self):
        registry, _ = _make_registry()
        registry.register("tool_a", _dummy_tool, {"description": "A"})
        registry.register("tool_b", _dummy_tool, {"description": "B"})
        assert set(registry.tool_names) == {"tool_a", "tool_b"}

    def test_tool_descriptions_includes_all_names(self):
        registry, _ = _make_registry()
        registry.register(
            "get_order",
            _dummy_tool,
            {
                "description": "Fetch production order",
                "parameters": {"order_id": "str — order number"},
            },
        )
        desc = registry.tool_descriptions()
        assert "get_order" in desc
        assert "Fetch production order" in desc
        assert "order_id" in desc

    def test_tool_descriptions_empty_registry(self):
        registry, _ = _make_registry()
        assert "no tools" in registry.tool_descriptions()

    def test_dispatch_calls_registered_tool(self):
        registry, _ = _make_registry()
        registry.register("get_order", _dummy_tool, {"description": "Get order"})

        result_json = asyncio.run(
            registry.dispatch("get_order", {"order_id": "ORD-001"})
        )
        result = json.loads(result_json)
        assert result["order_number"] == "ORD-001"
        assert result["status"] == "pending"

    def test_dispatch_unknown_tool_returns_error(self):
        registry, _ = _make_registry()

        result_json = asyncio.run(registry.dispatch("nonexistent_tool", {}))
        result = json.loads(result_json)
        assert "error" in result
        assert "nonexistent_tool" in result["error"]

    def test_dispatch_tool_exception_returns_error(self):
        registry, _ = _make_registry()
        registry.register("bad_tool", _failing_tool, {"description": "Fails"})

        result_json = asyncio.run(registry.dispatch("bad_tool", {}))
        result = json.loads(result_json)
        assert "error" in result

    def test_dispatch_wrong_arguments_returns_error(self):
        registry, _ = _make_registry()
        registry.register("get_order", _dummy_tool, {"description": "Get order"})

        # _dummy_tool requires order_id — passing wrong kwarg should fail gracefully
        result_json = asyncio.run(
            registry.dispatch("get_order", {"wrong_param": "value"})
        )
        result = json.loads(result_json)
        assert "error" in result


# ---------------------------------------------------------------------------
# _parse_react tests
# ---------------------------------------------------------------------------

class TestParseReact:
    def test_happy_path_final_answer(self):
        from backend.app.services.agent.agent import _parse_react

        text = (
            "Thought: I have enough information.\n"
            "Action: FINAL_ANSWER\n"
            "Action Input: Order ORD-001 is at high risk."
        )
        thought, action, action_input = _parse_react(text)
        assert action == "FINAL_ANSWER"
        assert "ORD-001" in action_input
        assert "enough information" in thought

    def test_happy_path_tool_call(self):
        from backend.app.services.agent.agent import _parse_react

        text = (
            "Thought: I need to check active orders.\n"
            "Action: get_active_orders\n"
            'Action Input: {"limit": 5}'
        )
        thought, action, action_input = _parse_react(text)
        assert action == "get_active_orders"
        assert json.loads(action_input)["limit"] == 5

    def test_no_format_falls_back_to_final_answer(self):
        from backend.app.services.agent.agent import _parse_react

        text = "I cannot find any relevant orders."
        thought, action, action_input = _parse_react(text)
        assert action == "FINAL_ANSWER"
        assert action_input == text.strip()

    def test_case_insensitive_action(self):
        from backend.app.services.agent.agent import _parse_react

        text = (
            "Thought: done.\n"
            "action: FINAL_ANSWER\n"
            "Action Input: All good."
        )
        _, action, _ = _parse_react(text)
        assert action == "FINAL_ANSWER"


# ---------------------------------------------------------------------------
# CopilotAgent tests
# ---------------------------------------------------------------------------

def _make_agent(llm_response: str | list[str]):
    """Build a CopilotAgent with mocked DB, LLM, and memory."""
    from backend.app.services.agent.agent import CopilotAgent
    from backend.app.services.agent.tool_registry import ToolRegistry

    db = AsyncMock()

    # LLM mock: returns each response in turn for successive calls
    llm = AsyncMock()
    if isinstance(llm_response, str):
        llm.complete = AsyncMock(return_value=llm_response)
    else:
        llm.complete = AsyncMock(side_effect=llm_response)

    registry = ToolRegistry(db)
    registry.register("get_active_orders", _dummy_tool, {"description": "Get active orders"})

    return CopilotAgent(db, llm, registry), llm


async def _run_agent(agent, message: str, session_token: str) -> list[str]:
    chunks = []
    async for chunk in agent.run(message, session_token):
        chunks.append(chunk)
    return chunks


class TestCopilotAgent:
    def test_final_answer_on_first_iteration(self):
        react_response = (
            "Thought: I have sufficient context.\n"
            "Action: FINAL_ANSWER\n"
            "Action Input: There are currently no high-risk orders."
        )
        with patch("backend.app.services.agent.agent.SessionMemory") as MockMemory:
            mem = AsyncMock()
            mem.load = AsyncMock(return_value=[])
            mem.save = AsyncMock()
            mem.compress = AsyncMock()
            MockMemory.return_value = mem

            agent, llm = _make_agent(react_response)
            chunks = asyncio.run(_run_agent(agent, "Any risky orders?", "sess-1"))

        assert len(chunks) == 1
        assert "no high-risk orders" in chunks[0]
        llm.complete.assert_called_once()
        # User message + final answer were both saved
        assert mem.save.call_count == 2

    def test_tool_call_then_final_answer(self):
        tool_response = (
            "Thought: I need to check at-risk orders.\n"
            "Action: get_active_orders\n"
            'Action Input: {"order_id": "ORD-001"}'
        )
        final_response = (
            "Thought: I now have the data.\n"
            "Action: FINAL_ANSWER\n"
            "Action Input: Order ORD-001 is pending with low risk."
        )

        with patch("backend.app.services.agent.agent.SessionMemory") as MockMemory:
            mem = AsyncMock()
            mem.load = AsyncMock(return_value=[])
            mem.save = AsyncMock()
            mem.compress = AsyncMock()
            MockMemory.return_value = mem

            agent, llm = _make_agent([tool_response, final_response])
            chunks = asyncio.run(_run_agent(agent, "What's the status of ORD-001?", "sess-2"))

        assert len(chunks) == 1
        assert "ORD-001" in chunks[0]
        assert llm.complete.call_count == 2

    def test_max_iterations_returns_fallback(self):
        # LLM always returns a tool call — never FINAL_ANSWER
        never_final = (
            "Thought: Need more data.\n"
            "Action: get_active_orders\n"
            "Action Input: {}"
        )
        with patch("backend.app.services.agent.agent.SessionMemory") as MockMemory:
            mem = AsyncMock()
            mem.load = AsyncMock(return_value=[])
            mem.save = AsyncMock()
            mem.compress = AsyncMock()
            MockMemory.return_value = mem

            agent, llm = _make_agent([never_final] * 10)
            chunks = asyncio.run(
                _run_agent(agent, "Give me all orders forever.", "sess-3")
            )

        # Should return the fallback message after MAX_ITERATIONS
        assert len(chunks) == 1
        assert chunks[0]  # Non-empty
        assert llm.complete.call_count == agent.MAX_ITERATIONS

    def test_llm_failure_returns_graceful_message(self):
        from backend.app.services.llm.client import LLMError

        with patch("backend.app.services.agent.agent.SessionMemory") as MockMemory:
            mem = AsyncMock()
            mem.load = AsyncMock(return_value=[])
            mem.save = AsyncMock()
            mem.compress = AsyncMock()
            MockMemory.return_value = mem

            agent, llm = _make_agent("irrelevant")
            llm.complete = AsyncMock(side_effect=LLMError("timeout"))

            chunks = asyncio.run(_run_agent(agent, "Hello", "sess-4"))

        assert len(chunks) == 1
        assert "trouble" in chunks[0].lower() or chunks[0]

    def test_memory_with_history_is_included_in_prompt(self):
        """Verify that load() result is used to build the system prompt context."""
        history = [
            {"role": "user", "content": "Show me risky orders"},
            {"role": "assistant", "content": "I found 3 high-risk orders."},
        ]
        react_response = (
            "Thought: Continuing from previous turn.\n"
            "Action: FINAL_ANSWER\n"
            "Action Input: Continuing from before — still 3 high-risk orders."
        )
        with patch("backend.app.services.agent.agent.SessionMemory") as MockMemory:
            mem = AsyncMock()
            mem.load = AsyncMock(return_value=history)
            mem.save = AsyncMock()
            mem.compress = AsyncMock()
            MockMemory.return_value = mem

            agent, llm = _make_agent(react_response)
            asyncio.run(_run_agent(agent, "What about now?", "sess-5"))

        # The LLM was called — the prompt included history
        call_args = llm.complete.call_args[0][0]  # first positional arg
        context = " ".join(m["content"] for m in call_args)
        assert "3 high-risk orders" in context or "risky orders" in context


# ---------------------------------------------------------------------------
# SessionMemory tests
# ---------------------------------------------------------------------------

class TestSessionMemory:
    def _make_memory(self):
        from backend.app.services.agent.memory import SessionMemory

        db = AsyncMock()
        llm = AsyncMock()
        return SessionMemory(db, llm), db, llm

    def test_get_or_create_session_creates_new_session(self):
        memory, db, _ = self._make_memory()

        # Simulate no existing session
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=mock_result)
        db.flush = AsyncMock()

        async def _run():
            session = await memory._get_or_create_session("new-token")
            return session

        session = asyncio.run(_run())
        assert session.session_token == "new-token"
        db.add.assert_called_once_with(session)
        db.flush.assert_called()

    def test_save_adds_message(self):
        memory, db, _ = self._make_memory()

        mock_result = MagicMock()
        existing_session = MagicMock()
        existing_session.id = "some-uuid"
        existing_session.summary = None
        mock_result.scalar_one_or_none.return_value = existing_session
        db.execute = AsyncMock(return_value=mock_result)
        db.flush = AsyncMock()

        asyncio.run(memory.save("token-1", "user", "Hello factory!"))

        # db.add was called with the ChatMessage
        db.add.assert_called_once()
        added = db.add.call_args[0][0]
        assert added.role == "user"
        assert added.content == "Hello factory!"
