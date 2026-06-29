"""Shopping list add, remove, and show business logic."""

from __future__ import annotations

import logging

from config import Settings, get_settings
from llm_client import LLMClient, LLMError
from models.operations import ShoppingServiceResult
from prompts import (
    SHOPPING_ADD_PROMPT,
    SHOPPING_EMPTY_RESPONSE,
    SHOPPING_NOT_FOUND_RESPONSE,
    SHOPPING_REMOVE_PROMPT,
)
from prompts.safety import wrap_user_content
from repositories.shopping_repository import ShoppingRepository
from shopping import ShoppingItemCreate, format_shopping_for_prompt


logger = logging.getLogger(__name__)


class ShoppingServiceError(Exception):
    """Raised when shopping list operations fail."""


class ShoppingService:
    """Orchestrates shopping list workflows."""

    def __init__(
        self,
        repository: ShoppingRepository,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the shopping service with its dependencies."""
        self._repository = repository
        self._llm = llm
        self._settings = settings or get_settings()

    async def add_item(self, user_input: str, session_id: str) -> ShoppingServiceResult:
        """Extract item(s) from natural language and add them to the shopping list."""
        _ = session_id
        message = user_input.strip()
        payload = await self._complete_json(
            SHOPPING_ADD_PROMPT.format(message=wrap_user_content(message))
        )
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list) or not raw_items:
            raise ShoppingServiceError("No shopping items extracted.")

        added_names: list[str] = []
        for entry in raw_items:
            if not isinstance(entry, dict):
                continue
            item_name = str(entry.get("item", "")).strip()
            if not item_name:
                continue
            quantity = str(entry.get("quantity", "")).strip()
            await self._repository.add(
                ShoppingItemCreate(item=item_name, quantity=quantity)
            )
            added_names.append(item_name.lower())

        if not added_names:
            raise ShoppingServiceError("No valid shopping items extracted.")

        logger.info("Added shopping items: %s", added_names)
        return ShoppingServiceResult(text=_format_added_response(added_names))

    async def remove_item(self, user_input: str, session_id: str) -> ShoppingServiceResult:
        """Extract item(s) from natural language and remove them from the shopping list."""
        _ = session_id
        message = user_input.strip()
        current_items = await self._repository.list_active()
        payload = await self._complete_json(
            SHOPPING_REMOVE_PROMPT.format(
                message=wrap_user_content(message),
                items=format_shopping_for_prompt(current_items),
            )
        )
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list) or not raw_items:
            return ShoppingServiceResult(text=SHOPPING_NOT_FOUND_RESPONSE)

        removed_names: list[str] = []
        for item_name in raw_items:
            if not isinstance(item_name, str):
                continue
            normalized = item_name.strip().lower()
            if not normalized:
                continue
            if await self._repository.remove(normalized):
                removed_names.append(normalized)

        if not removed_names:
            return ShoppingServiceResult(text=SHOPPING_NOT_FOUND_RESPONSE)

        remaining = await self._repository.list_active()
        logger.info("Removed shopping items: %s", removed_names)
        return ShoppingServiceResult(text=_format_removed_response(removed_names, remaining))

    async def show_list(self, session_id: str) -> ShoppingServiceResult:
        """Return a formatted view of the current shopping list."""
        _ = session_id
        items = await self._repository.list_active()
        if not items:
            return ShoppingServiceResult(text=SHOPPING_EMPTY_RESPONSE)
        names = [item.item for item in items]
        return ShoppingServiceResult(text=f"You have: {', '.join(names)}.")

    async def _complete_json(self, prompt: str) -> dict:
        """Call the LLM and return parsed JSON, wrapping errors."""
        try:
            return await self._llm.complete_json(prompt)
        except LLMError as exc:
            raise ShoppingServiceError(str(exc)) from exc


def _format_added_response(item_names: list[str]) -> str:
    """Build a confirmation message for added shopping items."""
    if len(item_names) == 1:
        return f"Added {item_names[0]} to your shopping list."
    joined = " and ".join(item_names)
    return f"Added {joined} to your shopping list."


def _format_removed_response(removed_names: list[str], remaining_items) -> str:
    """Build a confirmation message for removed shopping items."""
    if len(removed_names) == 1:
        prefix = f"Removed {removed_names[0]}."
    else:
        prefix = f"Removed {', '.join(removed_names)}."

    if not remaining_items:
        return f"{prefix} Your shopping list is empty."

    remaining_names = ", ".join(item.item for item in remaining_items)
    return f"{prefix} Your shopping list now has: {remaining_names}."
