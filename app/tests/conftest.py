"""Shared pytest fixtures for Hippo Terminal tests."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from config import MemoryType, Settings
from embeddings import EmbeddingClient
from llm_client import LLMClient


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("HIPPO_ENV", "development")
os.environ.setdefault("HIPPO_SESSION_SECRET", "test-session-secret")
os.environ.setdefault("WHATSAPP_WEBHOOK_SECRET", "test-webhook-secret")


class FakeEmbeddingClient(EmbeddingClient):
    """Deterministic embedding client for tests without network access."""

    async def embed_one(self, text: str) -> list[float]:
        """Return a stable vector derived from the input text."""
        seed = sum(ord(char) for char in text.strip().lower()) % 997
        return [1.0, seed / 997.0, 0.25]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return stable vectors for each input text."""
        return [await self.embed_one(text) for text in texts]


class MockLLMClient(LLMClient):
    """Configurable LLM client that returns canned responses in tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        json_handler: Callable[..., dict[str, Any]] | None = None,
        text_handler: Callable[..., str] | None = None,
        json_responses: dict[str, dict[str, Any]] | None = None,
        text_response: str = "Found it. Test answer.",
    ) -> None:
        """Initialize the mock with optional handlers or keyed JSON responses."""
        super().__init__(settings)
        self.json_handler = json_handler
        self.text_handler = text_handler
        self.json_responses = json_responses or {}
        self.text_response = text_response
        self.json_calls: list[str] = []
        self.text_calls: list[str] = []

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Return a configured JSON payload for the given prompt."""
        _ = max_tokens, model
        self.json_calls.append(prompt)
        if self.json_handler is not None:
            result = self.json_handler(prompt, system=system)
            if inspect.isawaitable(result):
                result = await result
            return result
        for key, payload in self.json_responses.items():
            if key in prompt:
                return payload
        return self.json_responses.get("__default__", {})

    async def complete_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 120,
        model: str | None = None,
    ) -> str:
        """Return a configured text payload for the given prompt."""
        _ = max_tokens, model
        self.text_calls.append(prompt)
        if self.text_handler is not None:
            result = self.text_handler(prompt, temperature=temperature)
            if inspect.isawaitable(result):
                result = await result
            return result
        return self.text_response


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide a unique SQLite database path for each test."""
    return tmp_path / "test_hippo.db"


@pytest.fixture
def test_settings(temp_db_path: Path) -> Settings:
    """Provide isolated settings for tests."""
    return Settings(
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        database_path=str(temp_db_path),
        user_id="test_user",
        log_level="WARNING",
        llm_timeout_seconds=5.0,
        session_secret="test-session-secret",
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingClient:
    """Provide a deterministic embedding client."""
    return FakeEmbeddingClient()


@pytest.fixture
async def memory_repo(test_settings: Settings, fake_embeddings: FakeEmbeddingClient):
    """Initialize a memory repository backed by a temporary database."""
    from repositories.memory_repository import MemoryRepository

    repository = MemoryRepository(test_settings, embedding_client=fake_embeddings)
    await repository.initialize()
    return repository


@pytest.fixture
async def shopping_repo(test_settings: Settings):
    """Initialize a shopping repository backed by a temporary database."""
    from repositories.shopping_repository import ShoppingRepository

    repository = ShoppingRepository(test_settings)
    await repository.initialize()
    return repository


def make_memory(title: str, content: str, memory_type: MemoryType = MemoryType.FACT):
    """Build a MemoryCreate payload for tests."""
    from memory import MemoryCreate

    return MemoryCreate(
        title=title,
        content=content,
        memory_type=memory_type,
        category="personal",
    )
