"""Tests for collection-style recall (listing all resources/links)."""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from config import MemoryType
from memory import format_collection_answer, is_collection_query, merge_memories
from retriever import MemoryRetriever
from services.memory_service import MemoryService
from tests.conftest import MockLLMClient, make_memory


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("pm resources", True),
        ("list my pm resources", True),
        ("show all pm links", True),
        ("where is my passport", False),
        ("pm resource", False),
    ],
)
def test_is_collection_query(query: str, expected: bool) -> None:
    """Plural and list-style queries should request full collections."""
    assert is_collection_query(query) is expected


def test_format_collection_answer_lists_all_urls() -> None:
    """Collection answers should include every stored URL."""
    memories = [
        make_memory(
            "PM resources",
            "https://example.com/a\n- https://example.com/b",
            MemoryType.LIST,
        )
    ]
    answer = format_collection_answer(memories)
    assert "https://example.com/a" in answer
    assert "https://example.com/b" in answer
    assert answer.startswith("Found it.")


def test_merge_memories_preserves_unique_order() -> None:
    """Merged recall results should stay unique and ordered."""
    first = SimpleNamespace(id="mem-1")
    second = SimpleNamespace(id="mem-2")
    merged = merge_memories([first], [second], [first])  # type: ignore[list-item]
    assert [memory.id for memory in merged] == ["mem-1", "mem-2"]


@pytest.mark.asyncio
async def test_query_collection_returns_all_urls(
    test_settings,
    memory_repo,
) -> None:
    """Querying pm resources should list every saved link."""
    await memory_repo.create(
        make_memory(
            "PM Resource Link",
            "https://pm-hiring-zamp.zampapps.com/index.html#manifesto",
            MemoryType.LIST,
        )
    )
    await memory_repo.create(
        make_memory(
            "PM Resource Link",
            "https://github.com/sumaanta99/hippo-studio",
            MemoryType.LIST,
        )
    )

    llm = MockLLMClient(test_settings, text_response="should not be used")
    service = MemoryService(
        memory_repo,
        MemoryRetriever(memory_repo.store, test_settings),
        llm,
        test_settings,
    )

    response = await service.query_memory("pm resources", "test_user")

    assert "https://pm-hiring-zamp.zampapps.com/index.html#manifesto" in response.text
    assert "https://github.com/sumaanta99/hippo-studio" in response.text
    assert len(response.search_results) >= 2
