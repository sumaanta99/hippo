"""Tests for memory retrieval and re-ranking."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from config import MemoryType
from retriever import MemoryRetriever
from tests.conftest import make_memory


@pytest.mark.asyncio
async def test_keyword_search_finds_passport(memory_repo, test_settings) -> None:
    """Keyword search should find a stored passport location."""
    await memory_repo.create(
        make_memory("Passport location", "Passport is in the locker.", MemoryType.OBJECT_LOCATION)
    )
    retriever = MemoryRetriever(memory_repo.store, test_settings)
    results = await memory_repo.search("where's my passport", limit=5)
    assert results
    assert "locker" in results[0].content.lower()


@pytest.mark.asyncio
async def test_entity_search_finds_chirag_memories(memory_repo, test_settings) -> None:
    """Entity search should return all chirag-related memories."""
    await memory_repo.create(make_memory("Chirag birthday", "Chirag's birthday is May 20."))
    await memory_repo.create(make_memory("Chirag gift", "Chirag's gift is Lattafa perfume."))
    retriever = MemoryRetriever(memory_repo.store, test_settings)
    results = await memory_repo.search_by_entity("chirag")
    assert len(results) >= 2


@pytest.mark.asyncio
async def test_plural_query_finds_singular_memory(memory_repo) -> None:
    """Searching for a shared topic token should find singular titles."""
    await memory_repo.create(
        make_memory(
            "PM resource",
            "https://example.com/pm-interview",
            MemoryType.LIST,
        )
    )
    results = await memory_repo.search_by_entity("pm resources")
    assert results
    assert "pm" in results[0].title.lower()


@pytest.mark.asyncio
async def test_empty_query_returns_no_results(memory_repo, test_settings) -> None:
    """Empty queries should return an empty result set."""
    retriever = MemoryRetriever(memory_repo.store, test_settings)
    results = await retriever.retrieve_and_rerank("", user_id=test_settings.user_id)
    assert results == []


@pytest.mark.asyncio
async def test_rerank_filters_irrelevant_candidates(memory_repo, test_settings) -> None:
    """Re-ranking should drop candidates the LLM marks as irrelevant."""
    passport = await memory_repo.create(
        make_memory("Passport location", "Passport is in the locker.", MemoryType.OBJECT_LOCATION)
    )
    await memory_repo.create(make_memory("Milk reminder", "Remember to buy milk.", MemoryType.MISC))

    retriever = MemoryRetriever(memory_repo.store, test_settings)
    retriever._rerank_with_llm = AsyncMock(
        return_value=[passport],
    )
    results = await retriever.retrieve_and_rerank(
        "where's my passport",
        user_id=test_settings.user_id,
        top_k=5,
    )
    assert len(results) == 1
    assert results[0].id == passport.id


@pytest.mark.asyncio
async def test_rerank_failure_falls_back_to_candidates(memory_repo, test_settings) -> None:
    """When re-ranking fails, semantic/keyword candidates should still be returned."""
    created = await memory_repo.create(
        make_memory("Passport location", "Passport is in the locker.", MemoryType.OBJECT_LOCATION)
    )
    retriever = MemoryRetriever(memory_repo.store, test_settings)

    async def failing_rerank(query: str, candidates):
        from retriever import RetrievalError

        raise RetrievalError("Re-ranking timed out.")

    retriever._rerank_with_llm = failing_rerank
    results = await retriever.retrieve_and_rerank(
        "passport",
        user_id=test_settings.user_id,
        top_k=5,
    )
    assert any(memory.id == created.id for memory in results)


@pytest.mark.asyncio
async def test_rerank_confidence_threshold(memory_repo, test_settings) -> None:
    """Confident keyword matches should skip reranking and return results directly."""
    created = await memory_repo.create(
        make_memory("Passport location", "Passport is in the locker.", MemoryType.OBJECT_LOCATION)
    )
    retriever = MemoryRetriever(memory_repo.store, test_settings)
    retriever._rerank_with_llm = AsyncMock(return_value=[])
    results = await retriever.retrieve_and_rerank(
        "passport",
        user_id=test_settings.user_id,
        top_k=5,
    )
    assert len(results) == 1
    assert results[0].id == created.id
