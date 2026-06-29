"""Tests for embedding utilities."""

from __future__ import annotations

import pytest

from embeddings import cosine_similarity, memory_embedding_text


def test_cosine_similarity_identical_vectors() -> None:
    """Identical vectors should have similarity 1.0."""
    vector = [1.0, 0.0, 0.0]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Orthogonal vectors should have similarity 0.0."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_invalid_vectors() -> None:
    """Mismatched or empty vectors should return 0.0."""
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0


def test_memory_embedding_text() -> None:
    """Embedding text should combine title and content."""
    text = memory_embedding_text("Passport", "Passport is in the locker.")
    assert "Passport" in text
    assert "locker" in text
