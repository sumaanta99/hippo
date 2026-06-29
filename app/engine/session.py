"""Per-session dependency wiring for the Hippo engine."""

from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from embeddings import EmbeddingClient
from llm_client import LLMClient
from repositories.memory_repository import MemoryRepository
from repositories.shopping_repository import ShoppingRepository
from retriever import MemoryRetriever
from services.memory_service import MemoryService
from services.shopping_service import ShoppingService


@dataclass
class SessionServices:
    """All services scoped to a single session (user)."""

    session_id: str
    memory_repository: MemoryRepository
    shopping_repository: ShoppingRepository
    memory_service: MemoryService
    shopping_service: ShoppingService
    retriever: MemoryRetriever


def build_session_services(
    session_id: str,
    settings: Settings,
    llm: LLMClient,
    embedding_client: EmbeddingClient | None = None,
) -> SessionServices:
    """Create a fully wired service bundle for one session.

    Args:
        session_id: Unique session or user identifier.
        settings: Base application settings.
        llm: Shared LLM client instance.
        embedding_client: Optional shared embedding client.

    Returns:
        SessionServices with repositories filtered to the session_id.
    """
    scoped_settings = settings.model_copy(update={"user_id": session_id})
    memory_repository = MemoryRepository(scoped_settings, embedding_client=embedding_client)
    shopping_repository = ShoppingRepository(scoped_settings)
    retriever = MemoryRetriever(memory_repository.store, scoped_settings)
    memory_service = MemoryService(memory_repository, retriever, llm, scoped_settings)
    shopping_service = ShoppingService(shopping_repository, llm, scoped_settings)
    return SessionServices(
        session_id=session_id,
        memory_repository=memory_repository,
        shopping_repository=shopping_repository,
        memory_service=memory_service,
        shopping_service=shopping_service,
        retriever=retriever,
    )
