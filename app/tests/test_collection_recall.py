"""Tests for collection-style recall (listing all resources/links)."""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from config import MemoryType
from memory import format_collection_answer, format_recall_answer, format_schedule_answer, is_collection_query, is_schedule_query, merge_memories, to_second_person
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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("do i have any upcoming meets", True),
        ("any meetings tomorrow", True),
        ("when is my meeting", True),
        ("where is my passport", False),
        ("pm resources", False),
    ],
)
def test_is_schedule_query(query: str, expected: bool) -> None:
    """Schedule-style queries should use meeting phrasing."""
    assert is_schedule_query(query) is expected


def test_format_schedule_answer_single_meeting() -> None:
    """Schedule answers should say 'You have a meeting...' not 'Found it.'"""
    memories = [
        make_memory(
            "Meeting",
            "Meeting tomorrow at 12 pm",
            MemoryType.FACT,
        )
    ]
    answer = format_schedule_answer(memories)
    assert answer == "You have a meeting tomorrow at 12 pm."
    assert not answer.startswith("Found it")


def test_format_schedule_answer_strips_first_person_phrasing() -> None:
    """Saved first-person phrasing should not duplicate in the answer."""
    memories = [
        make_memory(
            "Meeting",
            "I have a meeting at 12pm tomorrow",
            MemoryType.FACT,
        )
    ]
    answer = format_schedule_answer(memories)
    assert answer == "You have a meeting at 12pm tomorrow."
    assert "I have" not in answer
    assert "—" not in answer


def test_to_second_person_rewrites_location_memory() -> None:
    """First-person location saves should recall in second person."""
    assert (
        to_second_person("I put my passport in the locker")
        == "You put your passport in the locker"
    )


def test_format_recall_answer_location_query() -> None:
    """Location queries should answer in second person without echoing 'I'."""
    memories = [
        make_memory(
            "Passport location",
            "I put my passport in the locker",
            MemoryType.OBJECT_LOCATION,
        )
    ]
    answer = format_recall_answer(memories, query="where is my passport")
    assert answer == "You put your passport in the locker."
    assert "Found it" not in answer
    assert "I put" not in answer


@pytest.mark.asyncio
async def test_query_location_returns_second_person(
    test_settings,
    memory_repo,
) -> None:
    """Querying a first-person location save should use second person."""
    await memory_repo.create(
        make_memory(
            "Passport location",
            "I put my passport in the locker",
            MemoryType.OBJECT_LOCATION,
        )
    )

    llm = MockLLMClient(test_settings, text_response="should not be used")
    service = MemoryService(
        memory_repo,
        MemoryRetriever(memory_repo.store, test_settings),
        llm,
        test_settings,
    )

    response = await service.query_memory("where is my passport", "test_user")

    assert response.text == "You put your passport in the locker."


@pytest.mark.asyncio
async def test_query_schedule_returns_meeting_phrasing(
    test_settings,
    memory_repo,
) -> None:
    """Querying upcoming meetings should answer with 'You have a meeting'."""
    await memory_repo.create(
        make_memory(
            "Meeting",
            "Meeting tomorrow at 12 pm",
            MemoryType.FACT,
        )
    )

    llm = MockLLMClient(test_settings, text_response="should not be used")
    service = MemoryService(
        memory_repo,
        MemoryRetriever(memory_repo.store, test_settings),
        llm,
        test_settings,
    )

    response = await service.query_memory(
        "do i have any upcoming meets",
        "test_user",
    )

    assert response.text == "You have a meeting tomorrow at 12 pm."
    assert "Found it" not in response.text


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
