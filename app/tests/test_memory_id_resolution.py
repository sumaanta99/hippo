"""Tests for LLM memory id resolution."""

from __future__ import annotations

from datetime import datetime, timezone

from config import MemoryType
from memory import MemoryRecord
from services.memory_service import _resolve_memory_id


def _record(title: str, content: str, memory_id: str = "mem-test-1") -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        user_id="test_user",
        title=title,
        content=content,
        memory_type=MemoryType.OBJECT_LOCATION,
        category="personal",
        timestamp=datetime.now(timezone.utc),
        is_archived=False,
        version_number=1,
    )


def test_resolve_memory_id_accepts_candidate() -> None:
    """A candidate id returned by the model should be accepted."""
    memory = _record("Passport", "In the locker.")
    assert _resolve_memory_id(memory.id, [memory]) == memory.id


def test_resolve_memory_id_rejects_unknown() -> None:
    """Unknown ids should not be accepted."""
    memory = _record("Passport", "In the locker.")
    assert _resolve_memory_id("not-a-real-id", [memory]) is None


def test_resolve_memory_id_allows_single_candidate_fallback() -> None:
    """An empty id may fall back only when there is one candidate."""
    memory = _record("Passport", "In the locker.")
    assert _resolve_memory_id("", [memory]) == memory.id


def test_resolve_memory_id_rejects_empty_with_multiple_candidates() -> None:
    """An empty id must not guess when multiple candidates exist."""
    first = _record("Passport", "In the locker.", memory_id="mem-1")
    second = _record("Hair clip", "On the shelf.", memory_id="mem-2")
    assert _resolve_memory_id("", [first, second]) is None
