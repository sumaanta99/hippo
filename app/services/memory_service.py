"""Memory save, query, update, and delete business logic."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from config import MemoryType, Settings, get_settings
from llm_client import LLMClient, LLMError
from memory import (
    MemoryCreate,
    MemoryRecord,
    MemoryUpdate,
    append_list_items,
    append_topic,
    extract_urls,
    format_collection_answer,
    format_entity_answer,
    format_memories_for_prompt,
    has_new_list_items,
    is_append_request,
    is_collection_query,
    merge_memories,
    should_append_to_existing,
    _subject_tokens,
)
from constants import RETRIEVING_STATUS_MESSAGE
from models.operations import MemoryServiceResult
from prompts import (
    DELETE_EXTRACTION_PROMPT,
    DELETE_NO_MATCH_RESPONSE,
    LIST_ADDED_RESPONSE,
    NO_MATCH_RESPONSE,
    SAVE_CONFIRM_RESPONSE,
    SAVE_EXTRACTION_PROMPT,
    UPDATE_CONFIRM_RESPONSE,
    UPDATE_EXTRACTION_PROMPT,
    UPDATE_NO_MATCH_RESPONSE,
)
from repositories.memory_repository import MemoryRepository
from retriever import MemoryRetriever


logger = logging.getLogger(__name__)


class MemoryServiceError(Exception):
    """Raised when memory operations fail."""


class MemoryService:
    """Orchestrates memory persistence and retrieval workflows."""

    def __init__(
        self,
        repository: MemoryRepository,
        retriever: MemoryRetriever,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the memory service with its dependencies."""
        self._repository = repository
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or get_settings()

    async def save_memory(self, user_input: str, session_id: str) -> MemoryServiceResult:
        """Extract fields from natural language and persist a new or updated memory."""
        _ = session_id
        message = user_input.strip()
        payload = await self._complete_json(
            SAVE_EXTRACTION_PROMPT.format(message=message)
        )
        memory_type = _parse_memory_type(payload.get("memory_type"))
        title = str(payload.get("title", "")).strip()
        content = str(payload.get("content", message)).strip()
        category = str(payload.get("category", "personal")).strip() or "personal"

        if not title:
            title = content[:60]

        existing = await self._repository.find_similar(
            title, content, message, memory_type=memory_type
        )
        if existing is None and is_append_request(message):
            topic = append_topic(message)
            candidates = await self._repository.search_by_entity(topic)
            if not candidates:
                candidates = await self._repository.search(topic, limit=1)
            if candidates:
                existing = candidates[0]

        if existing is not None:
            if should_append_to_existing(existing, message, content, memory_type):
                merged_content = append_list_items(
                    existing.content, message, content
                )
                if merged_content != existing.content:
                    updated = await self._repository.update(
                        existing.id,
                        MemoryUpdate(
                            title=existing.title,
                            content=merged_content,
                            memory_type=MemoryType.LIST,
                            category=existing.category,
                        ),
                    )
                    if updated is not None:
                        return MemoryServiceResult(
                            text=LIST_ADDED_RESPONSE,
                            updated=[updated],
                        )
                return MemoryServiceResult(text=UPDATE_CONFIRM_RESPONSE, updated=[existing])

            message_tokens = _subject_tokens(message, message)
            existing_tokens = _subject_tokens(existing.title, existing.content)
            is_label_resave = bool(message_tokens and message_tokens <= existing_tokens)

            update_title = existing.title if is_label_resave else title
            update_content = existing.content if is_label_resave else content
            update_type = existing.memory_type if is_label_resave else memory_type
            update_category = existing.category if is_label_resave else category

            updated = await self._repository.update(
                existing.id,
                MemoryUpdate(
                    title=update_title,
                    content=update_content,
                    memory_type=update_type,
                    category=update_category,
                ),
            )
            if updated is not None:
                return MemoryServiceResult(text=UPDATE_CONFIRM_RESPONSE, updated=[updated])

        append_target = await self._find_append_target(message, content, memory_type)
        if append_target is not None:
            merged_content = append_list_items(
                append_target.content, message, content
            )
            if merged_content != append_target.content:
                updated = await self._repository.update(
                    append_target.id,
                    MemoryUpdate(
                        title=append_target.title,
                        content=merged_content,
                        memory_type=MemoryType.LIST,
                        category=append_target.category,
                    ),
                )
                if updated is not None:
                    return MemoryServiceResult(text=LIST_ADDED_RESPONSE, updated=[updated])

        created = await self._repository.create(
            MemoryCreate(
                title=title,
                content=content,
                memory_type=memory_type,
                category=category,
            )
        )
        logger.info("Saved memory: title=%r", title)
        return MemoryServiceResult(text=SAVE_CONFIRM_RESPONSE, created=[created])

    async def query_memory(
        self,
        user_input: str,
        session_id: str,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> MemoryServiceResult:
        """Search memories and return a concise natural-language answer."""
        if on_status is not None:
            on_status(RETRIEVING_STATUS_MESSAGE)

        message = user_input.strip().rstrip("?").strip()
        matches = await self._retriever.retrieve_and_rerank(
            message,
            user_id=session_id,
        )
        if is_collection_query(message):
            entity_matches = await self._repository.search_by_entity(message)
            keyword_matches = await self._repository.search(message, limit=10)
            matches = merge_memories(matches, entity_matches, keyword_matches)
            if matches:
                return MemoryServiceResult(
                    text=format_collection_answer(matches),
                    search_results=matches,
                )
            return MemoryServiceResult(text=NO_MATCH_RESPONSE)

        if not matches:
            return MemoryServiceResult(text=NO_MATCH_RESPONSE)

        if len(matches) == 1:
            answer = f"Found it. {matches[0].content}"
        else:
            answer = f"Found it. {format_entity_answer(matches)}"

        if _looks_like_no_match(answer):
            return MemoryServiceResult(
                text=format_entity_answer(matches),
                search_results=matches,
            )
        return MemoryServiceResult(text=answer, search_results=matches)

    async def update_memory(self, user_input: str, session_id: str) -> MemoryServiceResult:
        """Update an existing memory identified from natural language."""
        _ = session_id
        message = user_input.strip()
        candidates = await self._repository.search(message, limit=5)
        if not candidates:
            entity_candidates = await self._repository.search_by_entity(message)
            candidates = entity_candidates[:5]

        if not candidates:
            logger.info(
                "No memory matched for update %r; falling back to save.",
                message,
            )
            return await self.save_memory(user_input, session_id)

        payload = await self._complete_json(
            UPDATE_EXTRACTION_PROMPT.format(
                message=message,
                memories=format_memories_for_prompt(candidates),
            )
        )
        memory_id = str(payload.get("memory_id", "")).strip()
        if not memory_id:
            memory_id = candidates[0].id

        updated = await self._repository.update(
            memory_id,
            MemoryUpdate(
                title=str(payload.get("title", candidates[0].title)).strip(),
                content=str(payload.get("content", message)).strip(),
                memory_type=_parse_memory_type(
                    payload.get("memory_type", candidates[0].memory_type.value)
                ),
                category=str(payload.get("category", candidates[0].category)).strip(),
            ),
        )
        if updated is None:
            return MemoryServiceResult(text=UPDATE_NO_MATCH_RESPONSE)

        logger.info("Updated memory: id=%s title=%r", memory_id, updated.title)
        return MemoryServiceResult(text=UPDATE_CONFIRM_RESPONSE, updated=[updated])

    async def delete_memory(self, user_input: str, session_id: str) -> MemoryServiceResult:
        """Archive a memory identified from natural language."""
        _ = session_id
        message = user_input.strip()
        candidates = await self._repository.search(message, limit=5)
        if not candidates:
            entity_candidates = await self._repository.search_by_entity(message)
            candidates = entity_candidates[:5]

        if not candidates:
            return MemoryServiceResult(text=DELETE_NO_MATCH_RESPONSE)

        payload = await self._complete_json(
            DELETE_EXTRACTION_PROMPT.format(
                message=message,
                memories=format_memories_for_prompt(candidates),
            )
        )
        memory_id = str(payload.get("memory_id", "")).strip()
        if not memory_id:
            memory_id = candidates[0].id

        target = next(
            (memory for memory in candidates if memory.id == memory_id),
            candidates[0],
        )
        deleted = await self._repository.delete(memory_id)
        if not deleted:
            return MemoryServiceResult(text=DELETE_NO_MATCH_RESPONSE)

        label = target.title.strip().lower()
        logger.info("Deleted memory: id=%s title=%r", memory_id, target.title)
        return MemoryServiceResult(
            text=f"Removed {label} from memory.",
            deleted=[target],
        )

    async def try_implicit_query(
        self,
        user_input: str,
        session_id: str,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> MemoryServiceResult | None:
        """Attempt a memory lookup when intent classification is uncertain."""
        message = user_input.strip()
        matches = await self._repository.search(message, limit=1)
        if not matches:
            entity_matches = await self._repository.search_by_entity(message, limit=1)
            if entity_matches:
                matches = entity_matches[:1]

        if not matches:
            return None

        return await self.query_memory(message, session_id, on_status=on_status)

    async def _find_append_target(
        self,
        message: str,
        content: str,
        memory_type: MemoryType,
    ) -> MemoryRecord | None:
        """Find an existing list memory to append a new item to."""
        if not extract_urls(message) and not is_append_request(message):
            return None

        query = append_topic(message) if is_append_request(message) else message
        candidates = await self._repository.search_by_entity(query)
        if not candidates:
            candidates = await self._repository.search(query, limit=5)

        for candidate in candidates:
            if should_append_to_existing(candidate, message, content, memory_type):
                return candidate
            if has_new_list_items(candidate.content, message, content):
                return candidate
        return None

    async def _complete_json(self, prompt: str) -> dict[str, Any]:
        """Call the LLM and return parsed JSON, wrapping errors."""
        try:
            return await self._llm.complete_json(prompt)
        except LLMError as exc:
            raise MemoryServiceError(str(exc)) from exc


def _looks_like_no_match(answer: str) -> bool:
    """Detect when the model ignored stored memories."""
    normalized = answer.strip().lower()
    return (
        "don't have that stored" in normalized
        or "do not have that stored" in normalized
        or normalized == NO_MATCH_RESPONSE.lower()
    )


def _parse_memory_type(value: Any) -> MemoryType:
    """Parse a memory type string into an enum value."""
    if isinstance(value, MemoryType):
        return value

    normalized = str(value or MemoryType.MISC.value).strip().lower()
    try:
        return MemoryType(normalized)
    except ValueError:
        return MemoryType.MISC
