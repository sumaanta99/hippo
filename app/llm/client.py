"""Shim — canonical implementation lives in llm_client.py."""

from llm_client import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError"]
