"""Shim — canonical implementation lives in retriever.py."""

from retriever import MemoryRetriever, RetrievalError, RerankResult

__all__ = ["MemoryRetriever", "RetrievalError", "RerankResult"]
