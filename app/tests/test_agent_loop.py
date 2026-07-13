"""Tests for the agent loop, fast path, and feedback corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent.fast_path import is_compound_message, try_fast_path
from agent.loop import AgentLoop
from agent.prompts import build_system_prompt
from config import Intent, MemoryType
from engine.hippo_engine import HippoEngine
from stores.conversation_store import ConversationStore
from stores.feedback_store import FeedbackStore
from tests.conftest import MockLLMClient, make_memory


@dataclass
class FakeTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class FakeToolUseBlock:
    type: str = "tool_use"
    id: str = "toolu_1"
    name: str = "save_memory"
    input: dict[str, Any] | None = None


@dataclass
class FakeMessage:
    stop_reason: str
    content: list[Any]


def _assistant_message(*blocks: Any, stop_reason: str = "end_turn") -> FakeMessage:
    return FakeMessage(stop_reason=stop_reason, content=list(blocks))


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I'm out of eggs and remind me to email Angela by Friday", True),
    ("I'm out of eggs", False),
        ("what's on my shopping list", False),
        ("buy eggs", False),
    ],
)
def test_compound_message_detection(message: str, expected: bool) -> None:
    assert is_compound_message(message) is expected


@pytest.mark.asyncio
async def test_compound_message_runs_agent_with_multiple_tool_calls(
    test_settings,
    memory_repo,
    fake_embeddings,
) -> None:
    """Compound requests should produce multiple tool calls in one agent turn."""
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> FakeMessage:
        calls.append(kwargs)
        if len(calls) == 1:
            return _assistant_message(
                FakeToolUseBlock(id="toolu_1", name="add_shopping_item", input={"item": "eggs"}),
                FakeToolUseBlock(
                    id="toolu_2",
                    name="save_memory",
                    input={
                        "content": "Email Angela by Friday",
                        "category": "follow_up",
                    },
                ),
                stop_reason="tool_use",
            )
        return _assistant_message(
            FakeTextBlock(text="Added eggs and I'll remind you to email Angela by Friday."),
            stop_reason="end_turn",
        )

    engine = _build_engine(test_settings, memory_repo, fake_embeddings, fake_create)
    response = await engine.chat(
        "I'm out of eggs and remind me to email Angela by Friday",
        "agent-session",
    )

    assert response.intent == "agent"
    assert "eggs" in response.response.lower()
    assert len(response.message_id or "") > 0
    first_call = calls[0]
    assert "tools" in first_call
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_ambiguous_update_triggers_search_before_update(
    test_settings,
    memory_repo,
    fake_embeddings,
) -> None:
    """Ambiguous updates should search first instead of guessing."""
    await memory_repo.create(
        make_memory(
            "Passport A",
            "Passport is in the drawer",
            MemoryType.OBJECT_LOCATION,
        )
    )
    await memory_repo.create(
        make_memory(
            "Passport B",
            "Passport is in the locker",
            MemoryType.OBJECT_LOCATION,
        )
    )

    tool_names: list[str] = []

    async def fake_create(**kwargs: Any) -> FakeMessage:
        content = kwargs["messages"][-1]["content"]
        if isinstance(content, str):
            return _assistant_message(
                FakeToolUseBlock(id="toolu_1", name="search_memory", input={"query": "passport"}),
                stop_reason="tool_use",
            )
        if not tool_names:
            tool_names.append("search_memory")
            return _assistant_message(
                FakeToolUseBlock(
                    id="toolu_2",
                    name="update_memory",
                    input={"memory_id": "wrong", "new_content": "in locker"},
                ),
                stop_reason="tool_use",
            )
        return _assistant_message(
            FakeTextBlock(
                text="I found two passport memories — which one did you mean, the drawer or the locker?"
            ),
            stop_reason="end_turn",
        )

    engine = _build_engine(test_settings, memory_repo, fake_embeddings, fake_create)
    response = await engine.chat("change passport location to the locker", "update-session")

    assert "search_memory" in tool_names or response.intent == "agent"
    assert "?" in response.response or "locker" in response.response.lower()


@pytest.mark.asyncio
async def test_feedback_changes_tool_sequence_on_similar_message(
    test_settings,
    memory_repo,
    fake_embeddings,
) -> None:
    """Not-helpful feedback should steer the next similar request."""
    conversation = ConversationStore(test_settings)
    feedback = FeedbackStore(test_settings, embedding_client=fake_embeddings)
    await conversation.initialize()
    await feedback.initialize()

    turn_id = await conversation.append_turn(
        session_id="feedback-session",
        user_message="update passport location",
        assistant_response="Updated.",
        tool_calls=[
            {
                "name": "update_memory",
                "input": {"memory_id": "abc", "new_content": "locker"},
            }
        ],
    )
    await feedback.record_feedback(
        session_id="feedback-session",
        message_id=turn_id,
        rating="not_helpful",
        note="Call search_memory before update_memory when the target is unclear.",
        user_message="update passport location",
        assistant_response="Updated.",
        tool_calls=[
            {
                "name": "update_memory",
                "input": {"memory_id": "abc", "new_content": "locker"},
            }
        ],
    )

    captured_system_prompts: list[str] = []

    async def fake_create(**kwargs: Any) -> FakeMessage:
        captured_system_prompts.append(kwargs["system"])
        return _assistant_message(
            FakeToolUseBlock(id="toolu_1", name="search_memory", input={"query": "passport"}),
            stop_reason="tool_use",
        )

    async def fake_create_followup(**kwargs: Any) -> FakeMessage:
        captured_system_prompts.append(kwargs["system"])
        return _assistant_message(
            FakeTextBlock(text="I searched first and found your passport memory."),
            stop_reason="end_turn",
        )

    call_count = {"n": 0}

    async def fake_create_router(**kwargs: Any) -> FakeMessage:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return await fake_create(**kwargs)
        return await fake_create_followup(**kwargs)

    engine = _build_engine(
        test_settings,
        memory_repo,
        fake_embeddings,
        fake_create_router,
        conversation_store=conversation,
        feedback_store=feedback,
    )

    await engine.chat("please update my passport location", "feedback-session")

    assert captured_system_prompts
    assert "search_memory" in captured_system_prompts[0]
    assert "Avoid these patterns" in captured_system_prompts[0]


@pytest.mark.asyncio
async def test_fast_path_skips_agent_for_shopping_list(
    test_settings,
    memory_repo,
    fake_embeddings,
) -> None:
    """High-confidence shopping list queries should not invoke the agent loop."""
    agent_called = {"value": False}

    async def fake_create(**kwargs: Any) -> FakeMessage:
        agent_called["value"] = True
        return _assistant_message(FakeTextBlock(text="should not run"), stop_reason="end_turn")

    engine = _build_engine(test_settings, memory_repo, fake_embeddings, fake_create)
    response = await engine.chat("what's on my shopping list", "fast-session")

    assert response.intent == "shopping_show"
    assert agent_called["value"] is False


@pytest.mark.asyncio
async def test_fast_path_removes_bought_shopping_item(
    test_settings,
    memory_repo,
    shopping_repo,
    fake_embeddings,
) -> None:
    """Phrases like 'bought eggs' should remove from the list via fast path."""
    agent_called = {"value": False}

    async def fake_create(**kwargs: Any) -> FakeMessage:
        agent_called["value"] = True
        return _assistant_message(FakeTextBlock(text="agent"), stop_reason="end_turn")

    def shopping_json_handler(prompt: str, *, system: str | None = None) -> dict:
        lowered = prompt.lower()
        if "bought" in lowered and "eggs" in lowered:
            return {"items": ["eggs"]}
        if "buy" in lowered or "need" in lowered:
            items = []
            for name in ("eggs", "milk", "bread", "water"):
                if name in lowered:
                    items.append({"item": name, "quantity": ""})
            return {"items": items}
        return {"items": []}

    llm = MockLLMClient(test_settings, json_handler=shopping_json_handler)
    engine = _build_engine(
        test_settings,
        memory_repo,
        fake_embeddings,
        fake_create,
        llm=llm,
    )
    await engine.chat("I need milk, eggs, and bread", "shop-session")
    response = await engine.chat("bought eggs", "shop-session")

    assert response.intent == "shopping_remove"
    assert "Removed eggs" in response.response
    assert agent_called["value"] is False


@pytest.mark.asyncio
async def test_fast_path_handles_save_without_agent(
    test_settings,
    memory_repo,
    fake_embeddings,
) -> None:
    agent_called = {"value": False}

    async def fake_create(**kwargs: Any) -> FakeMessage:
        agent_called["value"] = True
        return _assistant_message(FakeTextBlock(text="agent"), stop_reason="end_turn")

    llm = MockLLMClient(
        test_settings,
        json_responses={
            "__default__": {
                "title": "Angela follow-up",
                "content": "Message Angela about the Q3 budget",
                "memory_type": "fact",
                "category": "personal",
            }
        },
    )
    engine = _build_engine(
        test_settings,
        memory_repo,
        fake_embeddings,
        fake_create,
        llm=llm,
    )
    response = await engine.chat(
        "Remind me to message Angela about the Q3 budget",
        "save-session",
    )

    assert response.intent == "save"
    assert agent_called["value"] is False
    assert response.memories_created or response.memories_updated


def test_build_system_prompt_includes_corrections() -> None:
    prompt = build_system_prompt(
        [
            {
                "user_message": "update passport",
                "tool_calls_made": [{"name": "update_memory"}],
                "note": "Search first.",
            }
        ]
    )
    assert "Avoid these patterns" in prompt
    assert "Search first." in prompt


def _build_engine(
    test_settings,
    memory_repo,
    fake_embeddings,
    messages_create,
    *,
    llm: MockLLMClient | None = None,
    conversation_store: ConversationStore | None = None,
    feedback_store: FeedbackStore | None = None,
):
    from llm_client import LLMClient

    resolved_llm = llm or MockLLMClient(test_settings)
    agent_loop = AgentLoop(test_settings, messages_create=messages_create)
    conversation = conversation_store or ConversationStore(test_settings)
    feedback = feedback_store or FeedbackStore(test_settings, embedding_client=fake_embeddings)
    return HippoEngine(
        test_settings,
        llm=resolved_llm,
        embedding_client=fake_embeddings,
        agent_loop=agent_loop,
        conversation_store=conversation,
        feedback_store=feedback,
    )
