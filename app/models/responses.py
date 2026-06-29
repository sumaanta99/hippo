"""Structured response models returned by HippoEngine."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from config import Intent, MemoryType
from memory import MemoryRecord


INTENT_LABELS: dict[Intent, str] = {
    Intent.SAVE_MEMORY: "save",
    Intent.QUERY_MEMORY: "recall",
    Intent.UPDATE_MEMORY: "update",
    Intent.DELETE_MEMORY: "delete",
    Intent.SHOPPING_ADD: "shopping_add",
    Intent.SHOPPING_REMOVE: "shopping_remove",
    Intent.SHOPPING_SHOW: "shopping_show",
    Intent.GENERAL_CHAT: "chat",
    Intent.UNKNOWN: "unknown",
}


def intent_label(intent: Intent) -> str:
    """Map an internal intent enum to a public-facing label."""
    return INTENT_LABELS.get(intent, intent.value.lower())


class MemorySnapshot(BaseModel):
    """A serializable view of a stored memory."""

    id: str
    session_id: str
    title: str
    content: str
    memory_type: MemoryType
    category: str
    timestamp: datetime
    version_number: int = 1

    @classmethod
    def from_record(cls, record: MemoryRecord) -> MemorySnapshot:
        """Build a snapshot from a database memory record."""
        return cls(
            id=record.id,
            session_id=record.user_id,
            title=record.title,
            content=record.content,
            memory_type=record.memory_type,
            category=record.category,
            timestamp=record.timestamp,
            version_number=record.version_number,
        )


class ChatResponse(BaseModel):
    """Structured result from a single engine.chat() call."""

    response: str
    intent: str
    session_id: str
    confidence: float = 0.0
    memories_created: list[MemorySnapshot] = Field(default_factory=list)
    memories_updated: list[MemorySnapshot] = Field(default_factory=list)
    memories_deleted: list[MemorySnapshot] = Field(default_factory=list)
    search_results: list[MemorySnapshot] = Field(default_factory=list)
    latency_ms: float = 0.0


class SessionStats(BaseModel):
    """Summary statistics for a session."""

    session_id: str
    memory_count: int
    shopping_item_count: int
