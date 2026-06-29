"""Internal service-layer result objects."""

from __future__ import annotations

from pydantic import BaseModel, Field

from memory import MemoryRecord


class MemoryServiceResult(BaseModel):
    """Result of a memory service operation."""

    text: str
    created: list[MemoryRecord] = Field(default_factory=list)
    updated: list[MemoryRecord] = Field(default_factory=list)
    deleted: list[MemoryRecord] = Field(default_factory=list)
    search_results: list[MemoryRecord] = Field(default_factory=list)


class ShoppingServiceResult(BaseModel):
    """Result of a shopping list service operation."""

    text: str
