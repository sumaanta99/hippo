"""Tests for HippoEngine public API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from classifier import ClassificationResult
from config import Intent, MemoryType
from engine.hippo_engine import HippoEngine
from models.operations import MemoryServiceResult
from prompts import SAVE_CONFIRM_RESPONSE
from tests.conftest import make_memory


@pytest.mark.asyncio
async def test_chat_returns_structured_response(test_settings) -> None:
    """chat() should return a ChatResponse with session scoping."""
    engine = HippoEngine(test_settings)
    await engine.initialize()
    engine._classifier.classify_intent = AsyncMock(
        return_value=ClassificationResult(
            intent=Intent.SAVE_MEMORY,
            confidence=0.95,
            reasoning="save",
        )
    )
    engine._session("user-a").memory_service.save_memory = AsyncMock(
        return_value=MemoryServiceResult(text=SAVE_CONFIRM_RESPONSE)
    )
    result = await engine.chat("hair clip on shelf", "user-a")
    assert result.response == SAVE_CONFIRM_RESPONSE
    assert result.session_id == "user-a"
    assert result.intent == "save"


@pytest.mark.asyncio
async def test_sessions_are_isolated(test_settings, fake_embeddings) -> None:
    """Memories saved in one session should not appear in another."""
    engine = HippoEngine(test_settings, embedding_client=fake_embeddings)
    await engine.initialize()

    repo_a = engine._session("session-a").memory_repository
    await repo_a.create(
        make_memory("Secret A", "Only in session A.", MemoryType.FACT)
    )

    memories_b = await engine.get_memories("session-b")
    assert memories_b == []

    memories_a = await engine.get_memories("session-a")
    assert len(memories_a) == 1
    assert memories_a[0].session_id == "session-a"


@pytest.mark.asyncio
async def test_get_stats(test_settings, fake_embeddings) -> None:
    """get_stats should return counts for the session."""
    engine = HippoEngine(test_settings, embedding_client=fake_embeddings)
    await engine.initialize()

    repo = engine._session("stats-user").memory_repository
    await repo.create(make_memory("One", "First memory."))
    stats = await engine.get_stats("stats-user")
    assert stats.session_id == "stats-user"
    assert stats.memory_count == 1
    assert stats.shopping_item_count == 0


@pytest.mark.asyncio
async def test_clear_session(test_settings, fake_embeddings) -> None:
    """clear_session should remove all data for a session."""
    engine = HippoEngine(test_settings, embedding_client=fake_embeddings)
    await engine.initialize()

    repo = engine._session("clear-me").memory_repository
    created = await repo.create(make_memory("Temp", "Temporary."))
    await engine.clear_session("clear-me")
    assert await engine.get_memories("clear-me") == []
    assert await engine.delete_memory(created.id, "clear-me") is False


@pytest.mark.asyncio
async def test_delete_memory_by_id(test_settings, fake_embeddings) -> None:
    """delete_memory should archive a specific memory."""
    engine = HippoEngine(test_settings, embedding_client=fake_embeddings)
    await engine.initialize()

    repo = engine._session("del-user").memory_repository
    created = await repo.create(make_memory("To delete", "Gone soon."))
    assert await engine.delete_memory(created.id, "del-user") is True
    assert await engine.get_memories("del-user") == []
