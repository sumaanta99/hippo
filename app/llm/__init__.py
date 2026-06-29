"""LLM layer — classification, completions, and embeddings."""

from classifier import ClassificationError, ClassificationResult, IntentClassifier
from embeddings import EmbeddingClient, EmbeddingError, cosine_similarity
from llm.client import LLMClient, LLMError

__all__ = [
    "ClassificationError",
    "ClassificationResult",
    "EmbeddingClient",
    "EmbeddingError",
    "IntentClassifier",
    "LLMClient",
    "LLMError",
    "cosine_similarity",
]
