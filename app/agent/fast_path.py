"""Fast-path routing that executes a single tool without the full agent loop."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import Intent
from engine.session import SessionServices
from fast_router import try_fast_classify
from memory import MemoryRecord


_COMPOUND_SEP = re.compile(r"\band\b", re.IGNORECASE)
_OUT_OF = re.compile(r"\bout of\b", re.IGNORECASE)
_REMIND = re.compile(r"\bremind\s+me\b", re.IGNORECASE)


@dataclass
class FastPathResult:
    """Result of a fast-path tool execution."""

    response_text: str
    intent: Intent
    tool_calls: list[dict[str, Any]]
    confidence: float
    reasoning: str
    created: list[MemoryRecord] = field(default_factory=list)
    updated: list[MemoryRecord] = field(default_factory=list)
    deleted: list[MemoryRecord] = field(default_factory=list)
    search_results: list[MemoryRecord] = field(default_factory=list)


def is_compound_message(message: str) -> bool:
    """Return True when a message likely needs multi-tool reasoning."""
    if not _COMPOUND_SEP.search(message):
        return False

    from fast_router import (
        _SAVE,
        _SHOPPING_ADD,
        _SHOPPING_REMOVE,
        _UPDATE,
        _DELETE,
    )

    signals = 0
    if _SHOPPING_ADD.search(message) or _OUT_OF.search(message):
        signals += 1
    if _SHOPPING_REMOVE.search(message):
        signals += 1
    if _SAVE.search(message) or _REMIND.search(message):
        signals += 1
    if _UPDATE.search(message):
        signals += 1
    if _DELETE.search(message):
        signals += 1
    return signals >= 2


def is_fast_path_eligible(message: str, intent: Intent) -> bool:
    """Return True when the fast router can skip the agent loop."""
    if intent in {Intent.UNKNOWN, Intent.GENERAL_CHAT}:
        return intent == Intent.GENERAL_CHAT
    if is_compound_message(message):
        return False
    return True


async def try_fast_path(
    message: str,
    services: SessionServices,
) -> FastPathResult | None:
    """Execute a high-confidence single-intent message without the agent loop."""
    classified = try_fast_classify(message)
    if classified is None:
        return None

    intent, confidence, reasoning = classified
    if not is_fast_path_eligible(message, intent):
        return None

    from agent.executor import ToolExecutor

    executor = ToolExecutor(services)
    tool_calls: list[dict[str, Any]] = []

    if intent == Intent.GENERAL_CHAT:
        from services.hippo_service import HippoService
        from llm_client import LLMClient
        from config import get_settings

        chat = HippoService(LLMClient(get_settings()), get_settings())
        text = await chat.handle_general_chat(message)
        return FastPathResult(
            response_text=text,
            intent=intent,
            tool_calls=[],
            confidence=confidence,
            reasoning=reasoning,
        )

    if intent == Intent.SHOPPING_SHOW:
        result = await executor.execute("list_shopping", {})
        tool_calls.append(
            {
                "name": "list_shopping",
                "input": {},
                "result": result.data,
                "fast_path": True,
            }
        )
        items = result.data.get("items") or []
        text = (
            "Your shopping list is empty."
            if not items
            else f"You have: {', '.join(items)}."
        )
        return FastPathResult(
            response_text=text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=confidence,
            reasoning=reasoning,
        )

    if intent == Intent.SHOPPING_ADD:
        service_result = await services.shopping_service.add_item(
            message,
            services.session_id,
        )
        tool_calls.append(
            {
                "name": "add_shopping_item",
                "input": {"message": message},
                "result": {"text": service_result.text},
                "fast_path": True,
            }
        )
        return FastPathResult(
            response_text=service_result.text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=confidence,
            reasoning=reasoning,
        )

    if intent == Intent.SHOPPING_REMOVE:
        service_result = await services.shopping_service.remove_item(
            message,
            services.session_id,
        )
        tool_calls.append(
            {
                "name": "remove_shopping_item",
                "input": {"message": message},
                "result": {"text": service_result.text},
                "fast_path": True,
            }
        )
        return FastPathResult(
            response_text=service_result.text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=confidence,
            reasoning=reasoning,
        )

    if intent == Intent.SAVE_MEMORY:
        service_result = await services.memory_service.save_memory(
            message,
            services.session_id,
        )
        tool_calls.append(
            {
                "name": "save_memory",
                "input": {"content": message},
                "result": {"text": service_result.text},
                "fast_path": True,
            }
        )
        return FastPathResult(
            response_text=service_result.text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=confidence,
            reasoning=reasoning,
            created=service_result.created,
            updated=service_result.updated,
            deleted=service_result.deleted,
            search_results=service_result.search_results,
        )

    if intent == Intent.QUERY_MEMORY:
        service_result = await services.memory_service.query_memory(
            message,
            services.session_id,
        )
        tool_calls.append(
            {
                "name": "search_memory",
                "input": {"query": message},
                "result": {"text": service_result.text},
                "fast_path": True,
            }
        )
        return FastPathResult(
            response_text=service_result.text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=confidence,
            reasoning=reasoning,
            created=service_result.created,
            updated=service_result.updated,
            deleted=service_result.deleted,
            search_results=service_result.search_results,
        )

    if intent == Intent.UPDATE_MEMORY:
        service_result = await services.memory_service.update_memory(
            message,
            services.session_id,
        )
        tool_calls.append(
            {
                "name": "update_memory",
                "input": {"message": message},
                "result": {"text": service_result.text},
                "fast_path": True,
            }
        )
        return FastPathResult(
            response_text=service_result.text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=confidence,
            reasoning=reasoning,
            created=service_result.created,
            updated=service_result.updated,
            deleted=service_result.deleted,
            search_results=service_result.search_results,
        )

    if intent == Intent.DELETE_MEMORY:
        service_result = await services.memory_service.delete_memory(
            message,
            services.session_id,
        )
        tool_calls.append(
            {
                "name": "delete_memory",
                "input": {"message": message},
                "result": {"text": service_result.text},
                "fast_path": True,
            }
        )
        return FastPathResult(
            response_text=service_result.text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=confidence,
            reasoning=reasoning,
            created=service_result.created,
            updated=service_result.updated,
            deleted=service_result.deleted,
            search_results=service_result.search_results,
        )

    return None
