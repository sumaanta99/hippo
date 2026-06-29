"""Data access layer for Hippo Terminal."""

from repositories.memory_repository import MemoryRepository
from repositories.shopping_repository import ShoppingRepository

__all__ = ["MemoryRepository", "ShoppingRepository"]
