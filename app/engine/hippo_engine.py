"""Central orchestrator for the Hippo memory engine."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.loop import AgentLoop, AgentRunResult

from config import Intent, Settings, get_settings
from embeddings import EmbeddingClient
from engine.session import SessionServices, build_session_services
from llm_client import LLMClient
from logger import configure_logging, get_logger
from memory import MemoryRecord
from models.operations import MemoryServiceResult, ShoppingServiceResult
from models.responses import ChatResponse, MemorySnapshot, SessionStats, intent_label
from prompts import API_FAILURE_RESPONSE, INPUT_TOO_LONG_RESPONSE, UNKNOWN_RESPONSE
from stores.conversation_store import ConversationStore
from stores.feedback_store import FeedbackRating, FeedbackStore


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
        agent_loop: AgentLoop | None = None,
        conversation_store: ConversationStore | None = None,
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        """Wire engine dependencies (all injectable for testing)."""
        self._settings = settings or get_settings()
        configure_logging(
            self._settings.log_level,
            structured=self._settings.structured_logging,
        )
        self._llm = llm or LLMClient(self._settings)
        self._embedding_client = embedding_client or EmbeddingClient(self._settings)
        if agent_loop is None:
            from agent.loop import AgentLoop

            self._agent_loop = AgentLoop(self._settings)
        else:
            self._agent_loop = agent_loop
        self._conversation_store = conversation_store or ConversationStore(self._settings)
        self._feedback_store = feedback_store or FeedbackStore(
            self._settings,
            embedding_client=self._embedding_client,
        )
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
        await self._conversation_store.initialize()
        await self._feedback_store.initialize()
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
        """Process a natural-language message and return a structured response."""
        _ = on_status
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

        services = self._session(session_id)
        history = await self._conversation_store.get_recent_history(
            session_id,
            limit=self._settings.agent_history_turns,
        )

        try:
            from agent.fast_path import try_fast_path
            from agent.loop import AgentLoopError
            from services.memory_service import MemoryServiceError
            from services.shopping_service import ShoppingServiceError

            fast_result = await try_fast_path(cleaned, services)
            if fast_result is not None:
                response = await self._finalize_turn(
                    session_id=session_id,
                    user_message=cleaned,
                    response_text=fast_result.response_text,
                    intent=fast_result.intent,
                    confidence=fast_result.confidence,
                    tool_calls=fast_result.tool_calls,
                    started=started,
                    created=fast_result.created,
                    updated=fast_result.updated,
                    deleted=fast_result.deleted,
                    search_results=fast_result.search_results,
                )
                return response

            corrections = await self._feedback_store.get_similar_corrections(
                cleaned,
                limit=3,
            )
            agent_result = await self._agent_loop.run(
                user_message=cleaned,
                services=services,
                history=history,
                corrections=corrections,
            )
            response = await self._finalize_agent_turn(
                session_id=session_id,
                user_message=cleaned,
                agent_result=agent_result,
                started=started,
            )
            return response
        except (MemoryServiceError, ShoppingServiceError) as exc:
            logger.error(
                "Handler failed.",
                error_type=type(exc).__name__,
                recovery_action="return API failure response",
                exc=exc,
            )
            return ChatResponse(
                response=API_FAILURE_RESPONSE,
                intent=intent_label(Intent.UNKNOWN),
                session_id=session_id,
            )
        except AgentLoopError as exc:
            logger.error(
                "Agent loop failed.",
                error_type="AgentLoopError",
                recovery_action="return API failure response",
                exc=exc,
            )
            return ChatResponse(
                response=API_FAILURE_RESPONSE,
                intent=intent_label(Intent.UNKNOWN),
                session_id=session_id,
            )

    async def submit_feedback(
        self,
        *,
        session_id: str,
        message_id: str,
        rating: FeedbackRating,
        note: str | None = None,
    ) -> str:
        """Record helpful/not_helpful feedback for a prior chat turn."""
        await self.initialize()
        turn = await self._conversation_store.get_turn(message_id)
        if turn is None or turn.session_id != session_id:
            raise HippoEngineError("Chat turn not found for this session.")

        return await self._feedback_store.record_feedback(
            session_id=session_id,
            message_id=message_id,
            rating=rating,
            note=note,
            user_message=turn.user_message,
            assistant_response=turn.assistant_response,
            tool_calls=turn.tool_calls,
        )

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

    async def get_admin_insights(self) -> dict[str, Any]:
        """Return agent tool-call and feedback data for admin analytics."""
        await self.initialize()
        return {
            "feedback": await self._feedback_store.get_summary(),
            "recentToolCalls": await self._conversation_store.list_recent_tool_calls(
                limit=50
            ),
        }

    async def _finalize_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        response_text: str,
        intent: Intent,
        confidence: float,
        tool_calls: list[dict[str, Any]],
        started: float,
        agent_trace: list[dict[str, Any]] | None = None,
        created: list[MemoryRecord] | None = None,
        updated: list[MemoryRecord] | None = None,
        deleted: list[MemoryRecord] | None = None,
        search_results: list[MemoryRecord] | None = None,
    ) -> ChatResponse:
        message_id = await self._conversation_store.append_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_response=response_text,
            tool_calls=tool_calls,
            agent_trace=agent_trace,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        response = ChatResponse(
            response=response_text,
            intent=intent_label(intent),
            session_id=session_id,
            confidence=confidence,
            memories_created=[MemorySnapshot.from_record(r) for r in (created or [])],
            memories_updated=[MemorySnapshot.from_record(r) for r in (updated or [])],
            memories_deleted=[MemorySnapshot.from_record(r) for r in (deleted or [])],
            search_results=[
                MemorySnapshot.from_record(r) for r in (search_results or [])
            ],
            latency_ms=elapsed_ms,
            message_id=message_id,
        )
        logger.log_event(
            "response_completed",
            session_id=session_id,
            intent=intent.value,
            confidence=confidence,
            latency_ms=round(elapsed_ms, 2),
            response_length=len(response.response),
            message_id=message_id,
        )
        return response

    async def _finalize_agent_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        agent_result: AgentRunResult,
        started: float,
    ) -> ChatResponse:
        return await self._finalize_turn(
            session_id=session_id,
            user_message=user_message,
            response_text=agent_result.final_text or UNKNOWN_RESPONSE,
            intent=Intent.AGENT,
            confidence=1.0,
            tool_calls=agent_result.tool_calls,
            started=started,
            agent_trace=agent_result.agent_trace,
            created=agent_result.effects.created,
            updated=agent_result.effects.updated,
            deleted=agent_result.effects.deleted,
            search_results=agent_result.effects.search_results,
        )
