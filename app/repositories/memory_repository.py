"""CRUD access for stored memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory import MemoryCreate, MemoryRecord, MemoryStore, MemoryUpdate
from config import Settings, get_settings

if TYPE_CHECKING:
    from embeddings import EmbeddingClient


class MemoryRepository:
    """Thin repository delegating memory persistence to SQLite."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        """Initialize the memory repository."""
        self._store = MemoryStore(settings, embedding_client=embedding_client)

    async def initialize(self) -> None:
        """Create tables and backfill embeddings."""
        await self._store.initialize()

    async def create(self, data: MemoryCreate) -> MemoryRecord:
        """Persist a new memory."""
        return await self._store.create(data)

    async def get_by_id(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a single active memory by id."""
        return await self._store.get_by_id(memory_id)

    async def list_active(self) -> list[MemoryRecord]:
        """Return all active memories for the current user."""
        return await self._store.list_active()

    async def update(self, memory_id: str, data: MemoryUpdate) -> MemoryRecord | None:
        """Update an existing memory."""
        return await self._store.update(memory_id, data)

    async def delete(self, memory_id: str) -> bool:
        """Archive a memory (soft delete)."""
        return await self._store.delete(memory_id)

    async def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        """Keyword search over active memories."""
        return await self._store.search(query, limit=limit)

    async def search_by_entity(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        """Entity-token search over active memories."""
        return await self._store.search_by_entity(query, limit=limit)

    async def semantic_search(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        """Embedding similarity search over active memories."""
        return await self._store.semantic_search(query, user_id=user_id, top_k=top_k)

    async def find_similar(
        self,
        title: str,
        content: str,
        message: str = "",
        memory_type=None,
        limit: int = 5,
    ) -> MemoryRecord | None:
        """Find an existing memory about the same subject."""
        return await self._store.find_similar(
            title, content, message, memory_type=memory_type, limit=limit
        )

    @property
    def store(self) -> MemoryStore:
        """Expose the underlying store for components that need direct access."""
        return self._store
