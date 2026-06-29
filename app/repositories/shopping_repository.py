"""CRUD access for shopping list items."""

from __future__ import annotations

from config import Settings, get_settings
from shopping import ShoppingItem, ShoppingItemCreate, ShoppingStore


class ShoppingRepository:
    """Thin repository delegating shopping list persistence to SQLite."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the shopping repository."""
        self._store = ShoppingStore(settings)

    async def initialize(self) -> None:
        """Create shopping list tables if needed."""
        await self._store.initialize()

    async def add(self, data: ShoppingItemCreate) -> ShoppingItem:
        """Add or update a shopping list item."""
        return await self._store.add(data)

    async def remove(self, item_name: str) -> bool:
        """Remove an item from the shopping list."""
        return await self._store.remove(item_name)

    async def list_active(self) -> list[ShoppingItem]:
        """Return incomplete shopping list items."""
        return await self._store.list_active()
