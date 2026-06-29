"""Tests for the memory repository CRUD layer."""

from __future__ import annotations

import pytest

from config import MemoryType
from memory import MemoryUpdate
from tests.conftest import make_memory


@pytest.mark.asyncio
async def test_create_and_get_by_id(memory_repo) -> None:
    """Creating a memory should persist it and allow retrieval by id."""
    created = await memory_repo.create(
        make_memory("Passport", "Passport is in the locker.", MemoryType.OBJECT_LOCATION)
    )
    fetched = await memory_repo.get_by_id(created.id)
    assert fetched is not None
    assert fetched.title == "Passport"
    assert fetched.content == "Passport is in the locker."


@pytest.mark.asyncio
async def test_list_active_excludes_archived(memory_repo) -> None:
    """Active listing should exclude archived memories."""
    created = await memory_repo.create(make_memory("Keys", "Keys are on the table."))
    active_before = await memory_repo.list_active()
    assert any(memory.id == created.id for memory in active_before)

    deleted = await memory_repo.delete(created.id)
    assert deleted is True

    active_after = await memory_repo.list_active()
    assert all(memory.id != created.id for memory in active_after)


@pytest.mark.asyncio
async def test_update_increments_version(memory_repo) -> None:
    """Updating a memory should increment its version number."""
    created = await memory_repo.create(
        make_memory("Hair clip", "Hair clip is on the bookshelf.", MemoryType.OBJECT_LOCATION)
    )
    assert created.version_number == 1

    updated = await memory_repo.update(
        created.id,
        MemoryUpdate(
            title="Hair clip",
            content="Hair clip is in the drawer.",
            memory_type=MemoryType.OBJECT_LOCATION,
            category="household",
        ),
    )
    assert updated is not None
    assert updated.version_number == 2
    assert "drawer" in updated.content


@pytest.mark.asyncio
async def test_delete_archives_instead_of_purging(memory_repo) -> None:
    """Deleting a memory should archive it rather than hard-delete the row."""
    created = await memory_repo.create(make_memory("Temp", "Temporary memory."))
    await memory_repo.delete(created.id)

    assert await memory_repo.get_by_id(created.id) is None
    assert await memory_repo.list_active() == []


@pytest.mark.asyncio
async def test_search_finds_keyword_match(memory_repo) -> None:
    """Keyword search should return memories matching query tokens."""
    await memory_repo.create(
        make_memory("Passport location", "Passport is in the locker.", MemoryType.OBJECT_LOCATION)
    )
    results = await memory_repo.search("where's my passport", limit=5)
    assert results
    assert "passport" in results[0].content.lower()


@pytest.mark.asyncio
async def test_find_similar_detects_same_subject(memory_repo) -> None:
    """Similar-memory detection should match overlapping subjects."""
    await memory_repo.create(
        make_memory("Passport location", "Passport is in the locker.", MemoryType.OBJECT_LOCATION)
    )
    similar = await memory_repo.find_similar(
        "Passport location",
        "Passport is now in the drawer.",
        "passport is now in the drawer",
        memory_type=MemoryType.OBJECT_LOCATION,
    )
    assert similar is not None
    assert "passport" in similar.title.lower()


@pytest.mark.asyncio
async def test_entity_search_finds_related_memories(memory_repo) -> None:
    """Entity search should return all memories mentioning a subject."""
    await memory_repo.create(make_memory("Chirag birthday", "Chirag's birthday is May 20."))
    await memory_repo.create(make_memory("Chirag gift", "Chirag's gift is Lattafa perfume."))
    results = await memory_repo.search_by_entity("chirag")
    assert len(results) >= 2
    assert all("chirag" in f"{memory.title} {memory.content}".lower() for memory in results)
