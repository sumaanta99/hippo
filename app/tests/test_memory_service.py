"""Tests for memory service business logic."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from config import MemoryType
from prompts import NO_MATCH_RESPONSE, SAVE_CONFIRM_RESPONSE, UPDATE_CONFIRM_RESPONSE
from repositories.memory_repository import MemoryRepository
from retriever import MemoryRetriever
from services.memory_service import MemoryService
from services.hippo_service import HippoService
from tests.conftest import MockLLMClient, make_memory


def _memory_json_handler(prompt: str, *, system: str | None = None) -> dict:
    """Return canned memory extraction payloads."""
    lowered = prompt.lower()
    if "delete" in lowered or "forget" in lowered:
        return {"memory_id": ""}
    if "update" in lowered or "now in" in lowered or "drawer" in lowered:
        return {
            "memory_id": "",
            "title": "Hair clip",
            "content": "Hair clip is in the drawer.",
            "memory_type": "object_location",
            "category": "household",
        }
    if "passport" in lowered:
        return {
            "title": "Passport location",
            "content": "Passport is in the locker.",
            "memory_type": "object_location",
            "category": "personal",
        }
    return {
        "title": "Hair clip",
        "content": "Hair clip is on the bookshelf.",
        "memory_type": "object_location",
        "category": "household",
    }


@pytest.fixture
def memory_service(test_settings, memory_repo) -> MemoryService:
    """Provide a memory service with a mock LLM and retriever."""
    llm = MockLLMClient(
        test_settings,
        json_handler=_memory_json_handler,
        text_response="Found it. Hair clip is on the bookshelf.",
    )
    retriever = MemoryRetriever(memory_repo.store, test_settings)
    return MemoryService(memory_repo, retriever, llm, test_settings)


@pytest.mark.asyncio
async def test_save_extracts_and_stores(memory_service: MemoryService) -> None:
    """Saving should extract structured fields and persist the memory."""
    response = await memory_service.save_memory(
        "hair clip on the bookshelf",
        "test_user",
    )
    assert response.text == SAVE_CONFIRM_RESPONSE
    results = await memory_service._repository.search("hair clip", limit=1)
    assert results
    assert "bookshelf" in results[0].content.lower()


@pytest.mark.asyncio
async def test_query_calls_status_callback(memory_service: MemoryService, memory_repo) -> None:
    """Querying should notify via on_status before retrieval."""
    await memory_repo.create(
        make_memory("Hair clip", "Hair clip is on the bookshelf.", MemoryType.OBJECT_LOCATION)
    )
    memory_service._retriever.retrieve_and_rerank = AsyncMock(
        return_value=await memory_repo.search("hair clip", limit=1)
    )
    memory_service._retriever.answer_query = AsyncMock(
        return_value="Found it. On the bookshelf."
    )
    statuses: list[str] = []
    await memory_service.query_memory(
        "where's my hair clip",
        "test_user",
        on_status=statuses.append,
    )
    assert statuses == ["Hippo is asking the ducks..."]


@pytest.mark.asyncio
async def test_query_returns_answer(memory_service: MemoryService, memory_repo) -> None:
    """Querying should return a natural-language answer."""
    await memory_repo.create(
        make_memory("Hair clip", "Hair clip is on the bookshelf.", MemoryType.OBJECT_LOCATION)
    )
    memory_service._retriever.retrieve_and_rerank = AsyncMock(
        return_value=await memory_repo.search("hair clip", limit=1)
    )
    response = await memory_service.query_memory("where's my hair clip", "test_user")
    assert "bookshelf" in response.text.lower()


@pytest.mark.asyncio
async def test_update_existing_memory(memory_service: MemoryService, memory_repo) -> None:
    """Updating should modify an existing memory instead of duplicating it."""
    created = await memory_repo.create(
        make_memory("Hair clip", "Hair clip is on the bookshelf.", MemoryType.OBJECT_LOCATION)
    )
    memory_service._retriever.retrieve_and_rerank = AsyncMock(return_value=[])
    response = await memory_service.update_memory(
        "hair clip is in the drawer now",
        "test_user",
    )
    assert response.text == UPDATE_CONFIRM_RESPONSE
    updated = await memory_repo.get_by_id(created.id)
    assert updated is not None
    assert updated.version_number == 2
    assert "drawer" in updated.content.lower()


@pytest.mark.asyncio
async def test_delete_archives_memory(memory_service: MemoryService, memory_repo) -> None:
    """Deleting should archive the selected memory."""
    created = await memory_repo.create(
        make_memory("Hair clip", "Hair clip is on the bookshelf.", MemoryType.OBJECT_LOCATION)
    )

    async def delete_handler(prompt: str, *, system: str | None = None) -> dict:
        if "delete" in prompt.lower() or "forget" in prompt.lower():
            return {"memory_id": created.id}
        return _memory_json_handler(prompt, system=system)

    memory_service._llm = MockLLMClient(
        memory_service._settings,
        json_handler=delete_handler,
    )
    response = await memory_service.delete_memory("forget hair clip", "test_user")
    assert "removed" in response.text.lower()
    assert await memory_repo.get_by_id(created.id) is None


@pytest.mark.asyncio
async def test_query_no_match(memory_service: MemoryService) -> None:
    """Queries with no matches should return the standard no-match response."""
    memory_service._retriever.retrieve_and_rerank = AsyncMock(return_value=[])
    memory_service._retriever.answer_query = AsyncMock(return_value=NO_MATCH_RESPONSE)
    response = await memory_service.query_memory("missing topic", "test_user")
    assert response.text == NO_MATCH_RESPONSE


@pytest.mark.asyncio
async def test_general_chat_does_not_save(test_settings, memory_repo) -> None:
    """General chat should respond without creating a memory."""
    llm = MockLLMClient(test_settings, text_response="Hey! What can I remember for you?")
    service = HippoService(llm, test_settings)
    response = await service.handle_general_chat("hey hippo")
    assert "remember" in response.lower()
    assert await memory_repo.list_active() == []
