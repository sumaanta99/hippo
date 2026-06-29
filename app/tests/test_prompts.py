"""Tests for rerank prompt helpers."""

from __future__ import annotations

from config import MemoryType
from memory import MemoryRecord
from datetime import datetime, timezone

from prompts.retrieval import format_candidates_for_rerank, rerank_prompt


def _record(title: str, content: str) -> MemoryRecord:
    """Build a memory record for prompt tests."""
    return MemoryRecord(
        id="test-id",
        user_id="user",
        title=title,
        content=content,
        memory_type=MemoryType.FACT,
        category="personal",
        timestamp=datetime.now(timezone.utc),
    )


def test_format_candidates_empty() -> None:
    """Empty candidate lists should render as (none)."""
    assert format_candidates_for_rerank([]) == "(none)"


def test_rerank_prompt_includes_plural_guidance() -> None:
    """Rerank prompts should mention plural and singular matching."""
    prompt = rerank_prompt("pm resources", [_record("PM resource", "https://example.com")])
    assert "pm resources" in prompt
    assert "Singulars" in prompt or "singular" in prompt.lower()
    assert "PM resource" in prompt
