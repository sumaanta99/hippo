"""Tests for shopping list service behavior."""

from __future__ import annotations

import pytest

from services.shopping_service import ShoppingService
from tests.conftest import MockLLMClient


def _shopping_json_handler(prompt: str, *, system: str | None = None) -> dict:
    """Return canned shopping extraction payloads."""
    lowered = prompt.lower()
    if "remove" in lowered or "forget" in lowered or "no more" in lowered:
        if "eggs" in lowered:
            return {"items": ["eggs"]}
        if "milk" in lowered:
            return {"items": ["milk"]}
        return {"items": []}
    items: list[dict[str, str]] = []
    for name in ("eggs", "milk", "detergent", "bread"):
        if name in lowered:
            items.append({"item": name, "quantity": ""})
    return {"items": items}


@pytest.fixture
def shopping_service(test_settings, shopping_repo) -> ShoppingService:
    """Provide a shopping service with a mock LLM extractor."""
    llm = MockLLMClient(test_settings, json_handler=_shopping_json_handler)
    return ShoppingService(shopping_repo, llm, test_settings)


@pytest.mark.asyncio
async def test_add_single_item(shopping_service: ShoppingService) -> None:
    """Adding one item should confirm with a friendly message."""
    response = await shopping_service.add_item("buy eggs", "test_user")
    assert response.text == "Added eggs to your shopping list."


@pytest.mark.asyncio
async def test_add_multiple_items(shopping_service: ShoppingService) -> None:
    """Multiple items in one message should all be added."""
    response = await shopping_service.add_item("need milk and detergent", "test_user")
    assert "milk" in response.text
    assert "detergent" in response.text
    show = await shopping_service.show_list("test_user")
    assert "milk" in show.text
    assert "detergent" in show.text


@pytest.mark.asyncio
async def test_remove_item_updates_remaining_list(shopping_service: ShoppingService) -> None:
    """Removing an item should report the remaining shopping list."""
    await shopping_service.add_item("buy eggs", "test_user")
    await shopping_service.add_item("need milk", "test_user")
    response = await shopping_service.remove_item("remove eggs", "test_user")
    assert "Removed eggs." in response.text
    assert "milk" in response.text


@pytest.mark.asyncio
async def test_show_list_displays_items_in_order(shopping_service: ShoppingService) -> None:
    """Showing the list should include all active items."""
    await shopping_service.add_item("buy eggs", "test_user")
    await shopping_service.add_item("need bread", "test_user")
    response = await shopping_service.show_list("test_user")
    assert response.text.startswith("You have:")
    assert "eggs" in response.text
    assert "bread" in response.text


@pytest.mark.asyncio
async def test_duplicate_add_does_not_duplicate(shopping_repo, test_settings) -> None:
    """Adding the same item twice should not create duplicates."""
    llm = MockLLMClient(test_settings, json_handler=_shopping_json_handler)
    service = ShoppingService(shopping_repo, llm, test_settings)
    await service.add_item("buy eggs", "test_user")
    await service.add_item("buy eggs", "test_user")
    items = await shopping_repo.list_active()
    assert len(items) == 1
    assert items[0].item == "eggs"
