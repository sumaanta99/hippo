"""Execute Hippo agent tools against session services."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import MemoryType
from engine.session import SessionServices
from memory import (
    MemoryCreate,
    MemoryRecord,
    MemoryUpdate,
    format_recall_answer,
    format_schedule_answer,
    is_collection_query,
    is_schedule_query,
    merge_memories,
)
from prompts import NO_MATCH_RESPONSE, SAVE_CONFIRM_RESPONSE, UPDATE_CONFIRM_RESPONSE
from shopping import ShoppingItemCreate


logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when a tool call cannot be executed."""


_CATEGORY_TO_TYPE: dict[str, MemoryType] = {
    "object_location": MemoryType.OBJECT_LOCATION,
    "follow_up": MemoryType.FACT,
    "deadline": MemoryType.FACT,
    "contact": MemoryType.CONTACT,
    "fact": MemoryType.FACT,
    "list_item": MemoryType.LIST,
}


@dataclass
class ToolEffects:
    """Side effects accumulated across tool calls in one agent turn."""

    created: list[MemoryRecord] = field(default_factory=list)
    updated: list[MemoryRecord] = field(default_factory=list)
    deleted: list[MemoryRecord] = field(default_factory=list)
    search_results: list[MemoryRecord] = field(default_factory=list)


@dataclass
class ToolExecutionResult:
    """Structured payload returned to the model as tool_result content."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ToolExecutor:
    """Run agent tools by delegating to existing repositories and services."""

    def __init__(self, services: SessionServices) -> None:
        self._services = services
        self.effects = ToolEffects()

    async def execute(self, name: str, tool_input: dict[str, Any]) -> ToolExecutionResult:
        """Dispatch a tool call by name."""
        handlers = {
            "save_memory": self._save_memory,
            "search_memory": self._search_memory,
            "update_memory": self._update_memory,
            "delete_memory": self._delete_memory,
            "add_shopping_item": self._add_shopping_item,
            "remove_shopping_item": self._remove_shopping_item,
            "list_shopping": self._list_shopping,
        }
        handler = handlers.get(name)
        if handler is None:
            raise ToolExecutionError(f"Unknown tool: {name}")
        return await handler(tool_input)

    async def _save_memory(self, tool_input: dict[str, Any]) -> ToolExecutionResult:
        content = str(tool_input.get("content", "")).strip()
        if not content:
            return ToolExecutionResult(success=False, error="content is required")

        category = str(tool_input.get("category", "fact")).strip().lower() or "fact"
        memory_type = _CATEGORY_TO_TYPE.get(category, MemoryType.FACT)
        title = content[:60]

        existing = await self._services.memory_repository.find_similar(
            title,
            content,
            content,
            memory_type=memory_type,
        )
        if existing is not None:
            updated = await self._services.memory_repository.update(
                existing.id,
                MemoryUpdate(
                    title=existing.title,
                    content=content,
                    memory_type=memory_type,
                    category=existing.category,
                ),
            )
            if updated is None:
                return ToolExecutionResult(success=False, error="Update failed")
            self.effects.updated.append(updated)
            return ToolExecutionResult(
                success=True,
                data={
                    "memory_id": updated.id,
                    "content": updated.content,
                    "message": UPDATE_CONFIRM_RESPONSE,
                },
            )

        created = await self._services.memory_repository.create(
            MemoryCreate(
                title=title,
                content=content,
                memory_type=memory_type,
                category=category if category in _CATEGORY_TO_TYPE else "personal",
            )
        )
        self.effects.created.append(created)
        return ToolExecutionResult(
            success=True,
            data={
                "memory_id": created.id,
                "content": created.content,
                "message": SAVE_CONFIRM_RESPONSE,
            },
        )

    async def _search_memory(self, tool_input: dict[str, Any]) -> ToolExecutionResult:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(success=False, error="query is required")

        top_k = int(tool_input.get("top_k") or 5)
        matches = await self._services.retriever.retrieve_and_rerank(
            query,
            user_id=self._services.session_id,
            top_k=top_k,
        )
        if is_collection_query(query):
            entity_matches = await self._services.memory_repository.search_by_entity(query)
            keyword_matches = await self._services.memory_repository.search(query, limit=10)
            matches = merge_memories(matches, entity_matches, keyword_matches)

        self.effects.search_results = matches
        payload = [
            {
                "memory_id": memory.id,
                "content": memory.content,
                "created_at": memory.timestamp.isoformat(),
                "category": memory.category,
                "memory_type": memory.memory_type.value,
            }
            for memory in matches
        ]
        return ToolExecutionResult(success=True, data={"matches": payload, "count": len(payload)})

    async def _update_memory(self, tool_input: dict[str, Any]) -> ToolExecutionResult:
        memory_id = str(tool_input.get("memory_id", "")).strip()
        new_content = str(tool_input.get("new_content", "")).strip()
        if not memory_id or not new_content:
            return ToolExecutionResult(
                success=False,
                error="memory_id and new_content are required",
            )

        target = await self._services.memory_repository.get_by_id(memory_id)
        if target is None:
            return ToolExecutionResult(success=False, error="Memory not found")

        updated = await self._services.memory_repository.update(
            memory_id,
            MemoryUpdate(
                title=target.title,
                content=new_content,
                memory_type=target.memory_type,
                category=target.category,
            ),
        )
        if updated is None:
            return ToolExecutionResult(success=False, error="Update failed")

        self.effects.updated.append(updated)
        return ToolExecutionResult(
            success=True,
            data={"memory_id": updated.id, "content": updated.content},
        )

    async def _delete_memory(self, tool_input: dict[str, Any]) -> ToolExecutionResult:
        memory_id = str(tool_input.get("memory_id", "")).strip()
        if not memory_id:
            return ToolExecutionResult(success=False, error="memory_id is required")

        target = await self._services.memory_repository.get_by_id(memory_id)
        if target is None:
            return ToolExecutionResult(success=False, error="Memory not found")

        deleted = await self._services.memory_repository.delete(memory_id)
        if not deleted:
            return ToolExecutionResult(success=False, error="Delete failed")

        self.effects.deleted.append(target)
        return ToolExecutionResult(
            success=True,
            data={"memory_id": memory_id, "content": target.content},
        )

    async def _add_shopping_item(self, tool_input: dict[str, Any]) -> ToolExecutionResult:
        item = str(tool_input.get("item", "")).strip()
        if not item:
            return ToolExecutionResult(success=False, error="item is required")

        await self._services.shopping_repository.add(ShoppingItemCreate(item=item))
        return ToolExecutionResult(success=True, data={"item": item, "added": True})

    async def _remove_shopping_item(self, tool_input: dict[str, Any]) -> ToolExecutionResult:
        item = str(tool_input.get("item", "")).strip().lower()
        if not item:
            return ToolExecutionResult(success=False, error="item is required")

        removed = await self._services.shopping_repository.remove(item)
        return ToolExecutionResult(success=True, data={"item": item, "removed": removed})

    async def _list_shopping(self, _tool_input: dict[str, Any]) -> ToolExecutionResult:
        items = await self._services.shopping_repository.list_active()
        names = [entry.item for entry in items]
        return ToolExecutionResult(
            success=True,
            data={"items": names, "count": len(names)},
        )


def format_search_answer(query: str, matches: list[MemoryRecord]) -> str:
    """Format a recall answer using existing phrasing rules."""
    if not matches:
        if is_schedule_query(query):
            return "You don't have any meetings coming up."
        return NO_MATCH_RESPONSE
    if is_schedule_query(query):
        return format_schedule_answer(matches)
    return format_recall_answer(matches, query=query)
