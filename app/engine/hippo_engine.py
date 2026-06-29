"""Central orchestrator for the Hippo memory engine."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from classifier import ClassificationError, IntentClassifier
from config import Intent, Settings, get_settings
from constants import LOW_CONFIDENCE_THRESHOLD
from embeddings import EmbeddingClient
from engine.session import SessionServices, build_session_services
from llm_client import LLMClient
from logger import configure_logging, get_logger
from models.operations import MemoryServiceResult, ShoppingServiceResult
from models.responses import ChatResponse, MemorySnapshot, SessionStats, intent_label
from prompts import API_FAILURE_RESPONSE, INPUT_TOO_LONG_RESPONSE, UNKNOWN_RESPONSE
from services.hippo_service import HippoService
from services.memory_service import MemoryServiceError
from services.shopping_service import ShoppingServiceError


logger = get_logger(__name__)


class HippoEngineError(Exception):
    """Raised when the engine cannot complete a request."""


class HippoEngine:
    """Interface-agnostic entry point for all Hippo memory operations."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        embedding_client: EmbeddingClient | None = None,
        classifier: IntentClassifier | None = None,
        chat_service: HippoService | None = None,
    ) -> None:
        """Wire engine dependencies (all injectable for testing).

        Args:
            settings: Application configuration.
            llm: OpenAI chat client.
            embedding_client: Embedding generation client.
            classifier: Intent classification service.
            chat_service: General conversation handler.
        """
        self._settings = settings or get_settings()
        configure_logging(
            self._settings.log_level,
            structured=self._settings.structured_logging,
        )
        self._llm = llm or LLMClient(self._settings)
        self._embedding_client = embedding_client or EmbeddingClient(self._settings)
        self._classifier = classifier or IntentClassifier(self._llm, self._settings)
        self._chat_service = chat_service or HippoService(self._llm, self._settings)
        self._sessions: dict[str, SessionServices] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Prepare shared database schema (idempotent)."""
        if self._initialized:
            return
        bootstrap = build_session_services(
            self._settings.user_id,
            self._settings,
            self._llm,
            self._embedding_client,
        )
        await bootstrap.memory_repository.initialize()
        await bootstrap.shopping_repository.initialize()
        self._initialized = True

    def _session(self, session_id: str) -> SessionServices:
        """Return cached session services, creating them if needed."""
        if session_id not in self._sessions:
            self._sessions[session_id] = build_session_services(
                session_id,
                self._settings,
                self._llm,
                self._embedding_client,
            )
        return self._sessions[session_id]

    async def chat(
        self,
        message: str,
        session_id: str,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        """Process a natural-language message and return a structured response.

        Args:
            message: User input text.
            session_id: Session or user identifier scoping all memory operations.
            on_status: Optional callback for in-progress status text (e.g. CLI).

        Returns:
            ChatResponse with reply text, intent, and affected memories.
        """
        await self.initialize()
        cleaned = message.strip()
        if not cleaned:
            return ChatResponse(
                response="",
                intent=intent_label(Intent.UNKNOWN),
                session_id=session_id,
            )

        if len(cleaned) > self._settings.max_input_length:
            return ChatResponse(
                response=INPUT_TOO_LONG_RESPONSE,
                intent=intent_label(Intent.UNKNOWN),
                session_id=session_id,
            )

        started = time.perf_counter()
        logger.log_event(
            "message_received",
            session_id=session_id,
            message_length=len(cleaned),
        )

        try:
            classification = await self._classifier.classify_intent(cleaned, session_id)
        except ClassificationError as exc:
            logger.error(
                "Intent classification failed.",
                error_type="ClassificationError",
                recovery_action="return API failure response",
                exc=exc,
            )
            return ChatResponse(
                response=API_FAILURE_RESPONSE,
                intent=intent_label(Intent.UNKNOWN),
                session_id=session_id,
            )

        intent = classification.intent
        if classification.confidence < LOW_CONFIDENCE_THRESHOLD and intent != Intent.UNKNOWN:
            intent = Intent.UNKNOWN

        services = self._session(session_id)
        try:
            result = await self._dispatch(
                intent, cleaned, services, on_status=on_status
            )
        except (MemoryServiceError, ShoppingServiceError) as exc:
            logger.error(
                "Handler failed.",
                error_type=type(exc).__name__,
                intent=intent.value,
                recovery_action="return API failure response",
                exc=exc,
            )
            return ChatResponse(
                response=API_FAILURE_RESPONSE,
                intent=intent_label(intent),
                session_id=session_id,
                confidence=classification.confidence,
            )

        resolved_intent = intent
        if (
            intent == Intent.UNKNOWN
            and isinstance(result, MemoryServiceResult)
            and result.search_results
        ):
            resolved_intent = Intent.QUERY_MEMORY

        elapsed_ms = (time.perf_counter() - started) * 1000
        response = self._to_chat_response(
            result,
            intent=resolved_intent,
            session_id=session_id,
            confidence=classification.confidence,
            latency_ms=elapsed_ms,
        )
        logger.log_event(
            "response_completed",
            session_id=session_id,
            intent=resolved_intent.value,
            confidence=classification.confidence,
            latency_ms=round(elapsed_ms, 2),
            response_length=len(response.response),
        )
        return response

    def chat_sync(self, message: str, session_id: str) -> ChatResponse:
        """Synchronous wrapper around :meth:`chat` for simple scripts."""
        return asyncio.run(self.chat(message, session_id))

    async def get_memories(self, session_id: str) -> list[MemorySnapshot]:
        """List all active memories for a session."""
        await self.initialize()
        records = await self._session(session_id).memory_repository.list_active()
        return [MemorySnapshot.from_record(record) for record in records]

    async def delete_memory(self, memory_id: str, session_id: str) -> bool:
        """Archive a memory by id within a session."""
        await self.initialize()
        return await self._session(session_id).memory_repository.delete(memory_id)

    async def get_stats(self, session_id: str) -> SessionStats:
        """Return summary counts for a session."""
        await self.initialize()
        services = self._session(session_id)
        memories = await services.memory_repository.list_active()
        shopping = await services.shopping_repository.list_active()
        return SessionStats(
            session_id=session_id,
            memory_count=len(memories),
            shopping_item_count=len(shopping),
        )

    async def clear_session(self, session_id: str) -> None:
        """Archive all memories and clear the shopping list for a session."""
        await self.initialize()
        services = self._session(session_id)
        for memory in await services.memory_repository.list_active():
            await services.memory_repository.delete(memory.id)
        for item in await services.shopping_repository.list_active():
            await services.shopping_repository.remove(item.item)

    async def _dispatch(
        self,
        intent: Intent,
        user_input: str,
        services: SessionServices,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> MemoryServiceResult | ShoppingServiceResult | str:
        """Route a classified intent to the appropriate session service."""
        session_id = services.session_id

        if intent == Intent.SAVE_MEMORY:
            return await services.memory_service.save_memory(user_input, session_id)

        if intent == Intent.QUERY_MEMORY:
            return await services.memory_service.query_memory(
                user_input, session_id, on_status=on_status
            )

        if intent == Intent.UPDATE_MEMORY:
            return await services.memory_service.update_memory(user_input, session_id)

        if intent == Intent.DELETE_MEMORY:
            return await services.memory_service.delete_memory(user_input, session_id)

        if intent == Intent.SHOPPING_ADD:
            return await services.shopping_service.add_item(user_input, session_id)

        if intent == Intent.SHOPPING_REMOVE:
            return await services.shopping_service.remove_item(user_input, session_id)

        if intent == Intent.SHOPPING_SHOW:
            return await services.shopping_service.show_list(session_id)

        if intent == Intent.GENERAL_CHAT:
            text = await self._chat_service.handle_general_chat(user_input)
            return MemoryServiceResult(text=text)

        implicit = await services.memory_service.try_implicit_query(
            user_input, session_id, on_status=on_status
        )
        if implicit is not None:
            return implicit

        return MemoryServiceResult(text=UNKNOWN_RESPONSE)

    def _to_chat_response(
        self,
        result: MemoryServiceResult | ShoppingServiceResult | str,
        *,
        intent: Intent,
        session_id: str,
        confidence: float,
        latency_ms: float,
    ) -> ChatResponse:
        """Convert an internal service result into a public ChatResponse."""
        if isinstance(result, str):
            return ChatResponse(
                response=result,
                intent=intent_label(intent),
                session_id=session_id,
                confidence=confidence,
                latency_ms=latency_ms,
            )

        if isinstance(result, ShoppingServiceResult):
            return ChatResponse(
                response=result.text,
                intent=intent_label(intent),
                session_id=session_id,
                confidence=confidence,
                latency_ms=latency_ms,
            )

        return ChatResponse(
            response=result.text,
            intent=intent_label(intent),
            session_id=session_id,
            confidence=confidence,
            memories_created=[MemorySnapshot.from_record(r) for r in result.created],
            memories_updated=[MemorySnapshot.from_record(r) for r in result.updated],
            memories_deleted=[MemorySnapshot.from_record(r) for r in result.deleted],
            search_results=[MemorySnapshot.from_record(r) for r in result.search_results],
            latency_ms=latency_ms,
        )
