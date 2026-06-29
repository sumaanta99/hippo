"""Shopping list storage and management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import aiosqlite
from pydantic import BaseModel

from config import Settings, get_database_path, get_settings


class ShoppingItem(BaseModel):
    """A single shopping list entry."""

    id: str
    item: str
    quantity: str = ""
    added_date: datetime
    completed: bool = False


class ShoppingItemCreate(BaseModel):
    """Payload for adding a shopping item."""

    item: str
    quantity: str = ""


class ShoppingStore:
    """Async SQLite store for the shopping list."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the shopping store."""
        self._settings = settings or get_settings()
        self._db_path = get_database_path(self._settings)

    async def initialize(self) -> None:
        """Create shopping list tables if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS shopping_items (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    item TEXT NOT NULL,
                    quantity TEXT NOT NULL DEFAULT '',
                    added_date TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_shopping_user ON shopping_items(user_id)"
            )
            await db.commit()

    async def add(self, data: ShoppingItemCreate) -> ShoppingItem:
        """Add an item to the shopping list."""
        normalized_item = data.item.strip().lower()
        existing = await self._find_by_name(normalized_item)
        if existing is not None:
            if data.quantity.strip():
                updated_quantity = data.quantity.strip()
                async with aiosqlite.connect(self._db_path) as db:
                    await db.execute(
                        """
                        UPDATE shopping_items
                        SET quantity = ?, added_date = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (
                            updated_quantity,
                            datetime.now(timezone.utc).isoformat(),
                            existing.id,
                            self._settings.user_id,
                        ),
                    )
                    await db.commit()
                return existing.model_copy(
                    update={
                        "quantity": updated_quantity,
                        "added_date": datetime.now(timezone.utc),
                    }
                )
            return existing

        record = ShoppingItem(
            id=str(uuid.uuid4()),
            item=normalized_item,
            quantity=data.quantity.strip(),
            added_date=datetime.now(timezone.utc),
            completed=False,
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO shopping_items (
                    id, user_id, item, quantity, added_date, completed
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    self._settings.user_id,
                    record.item,
                    record.quantity,
                    record.added_date.isoformat(),
                    int(record.completed),
                ),
            )
            await db.commit()
        return record

    async def remove(self, item_name: str) -> bool:
        """Remove an item from the shopping list."""
        existing = await self._find_by_name(item_name.strip().lower())
        if existing is None:
            return False

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM shopping_items WHERE id = ? AND user_id = ?",
                (existing.id, self._settings.user_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def list_active(self) -> list[ShoppingItem]:
        """Return incomplete shopping list items."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM shopping_items
                WHERE user_id = ? AND completed = 0
                ORDER BY added_date ASC
                """,
                (self._settings.user_id,),
            )
            rows = await cursor.fetchall()
        return [_row_to_item(row) for row in rows]

    async def _find_by_name(self, item_name: str) -> ShoppingItem | None:
        """Find a shopping item by normalized name."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM shopping_items
                WHERE user_id = ? AND completed = 0 AND item = ?
                LIMIT 1
                """,
                (self._settings.user_id, item_name),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_item(row)


def _row_to_item(row: aiosqlite.Row) -> ShoppingItem:
    """Convert a database row into a shopping item."""
    return ShoppingItem(
        id=row["id"],
        item=row["item"],
        quantity=row["quantity"] or "",
        added_date=datetime.fromisoformat(row["added_date"]),
        completed=bool(row["completed"]),
    )


def format_shopping_list(items: list[ShoppingItem]) -> str:
    """Format the shopping list for terminal output."""
    lines: list[str] = []
    for entry in items:
        if entry.quantity:
            lines.append(f"- {entry.item} ({entry.quantity})")
        else:
            lines.append(f"- {entry.item}")
    return "\n".join(lines)


def format_shopping_for_prompt(items: list[ShoppingItem]) -> str:
    """Format shopping items for LLM prompts."""
    if not items:
        return "(empty)"
    return ", ".join(item.item for item in items)
