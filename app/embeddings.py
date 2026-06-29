"""OpenAI embedding generation for semantic memory search."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, APITimeoutError, OpenAIError

from config import Settings, get_settings


logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingClient:
    """Generate text embeddings via the OpenAI API."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the embedding client."""
        self._settings = settings or get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    async def embed_one(self, text: str) -> list[float]:
        """Return the embedding vector for a single text."""
        vectors = await self.embed([text])
        return vectors[0]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for one or more texts."""
        if not texts:
            return []

        cleaned = [text.strip() or " " for text in texts]
        try:
            response = await self._client.embeddings.create(
                model=self._settings.embedding_model,
                input=cleaned,
                timeout=self._settings.llm_timeout_seconds,
            )
        except APITimeoutError as exc:
            raise EmbeddingError("Embedding request timed out.") from exc
        except OpenAIError as exc:
            raise EmbeddingError("Embedding request failed.") from exc

        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]
        logger.debug("Generated %d embedding(s).", len(vectors))
        return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def memory_embedding_text(title: str, content: str) -> str:
    """Build the text used when embedding a stored memory."""
    return f"{title.strip()}\n{content.strip()}"
