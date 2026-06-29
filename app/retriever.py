"""Two-stage memory retrieval: semantic search + LLM re-ranking."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Sequence

from openai import AsyncOpenAI, APITimeoutError, OpenAIError
from pydantic import BaseModel, Field

from config import Settings, get_settings
from json_utils import parse_json_object
from logger import get_logger
from memory import MemoryRecord, MemoryStore, format_memories_for_prompt
from prompts import GENERAL_CHAT_PROMPT, HIPPO_SYSTEM, QUERY_RESPONSE_PROMPT, rerank_prompt


logger = get_logger(__name__)


class RetrievalError(Exception):
    """Raised when memory retrieval or answer generation fails."""


class RerankResult(BaseModel):
    """Structured LLM re-ranking response."""

    relevant_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class MemoryRetriever:
    """Search stored memories and generate concise responses."""

    def __init__(
        self,
        memory_store: MemoryStore,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the retriever."""
        self._memory_store = memory_store
        self._settings = settings or get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    async def retrieve_and_rerank(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int | None = None,
    ) -> list[MemoryRecord]:
        """Retrieve candidate memories and re-rank them with an LLM."""
        resolved_user_id = user_id or self._settings.user_id
        candidate_limit = top_k or self._settings.semantic_search_top_k
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        logger.info(
            "Stage 1: retrieving up to %d candidates for query=%r user=%s",
            candidate_limit,
            cleaned_query,
            resolved_user_id,
        )
        candidates = await self._retrieve_candidates(cleaned_query, candidate_limit)
        if not candidates:
            logger.info("No retrieval candidates found for query=%r", cleaned_query)
            return []

        if _is_confident_match(cleaned_query, candidates):
            logger.info(
                "Skipping rerank for confident keyword match on query=%r",
                cleaned_query,
            )
            return candidates[:candidate_limit]

        logger.info(
            "Stage 2: re-ranking %d candidates for query=%r",
            len(candidates),
            cleaned_query,
        )
        try:
            reranked = await self._rerank_with_llm(cleaned_query, candidates)
        except RetrievalError as exc:
            logger.warning("LLM rerank failed, using semantic candidates: %s", exc)
            return candidates

        if reranked:
            logger.info("Rerank kept %d relevant memories.", len(reranked))
        return reranked

    async def find_matches(self, query: str) -> list[MemoryRecord]:
        """Return memories relevant to a query using two-stage retrieval."""
        return await self.retrieve_and_rerank(
            query,
            user_id=self._settings.user_id,
            top_k=self._settings.semantic_search_top_k,
        )

    async def answer_query(
        self,
        query: str,
        matches: list[MemoryRecord] | None = None,
    ) -> str:
        """Find relevant memories and answer the user's question."""
        resolved_matches = (
            matches
            if matches is not None
            else await self.find_matches(query)
        )
        if not resolved_matches:
            return await self._generate_text(
                QUERY_RESPONSE_PROMPT.format(
                    query=query.strip(),
                    memories="(none)",
                )
            )

        return await self._generate_text(
            QUERY_RESPONSE_PROMPT.format(
                query=query.strip(),
                memories=format_memories_for_prompt(resolved_matches),
            )
        )

    async def general_chat(self, message: str) -> str:
        """Respond to casual conversation without touching memory."""
        prompt = GENERAL_CHAT_PROMPT.format(message=message.strip())
        return await self._generate_text(prompt)

    async def _retrieve_candidates(
        self,
        query: str,
        top_k: int,
    ) -> list[MemoryRecord]:
        """Stage 1: gather semantic and keyword candidates in parallel."""
        semantic, keyword, entity = await asyncio.gather(
            self._memory_store.semantic_search(
                query,
                user_id=self._settings.user_id,
                top_k=top_k,
            ),
            self._memory_store.search(query, limit=top_k),
            self._memory_store.search_by_entity(query),
        )
        return _merge_candidates(semantic, keyword, entity, top_k)

    async def _rerank_with_llm(
        self,
        query: str,
        candidates: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Stage 2: ask the LLM which candidates truly answer the query."""
        prompt = rerank_prompt(query, candidates)
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": HIPPO_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=self._settings.llm_timeout_seconds,
            )
        except APITimeoutError as exc:
            raise RetrievalError("Re-ranking timed out.") from exc
        except OpenAIError as exc:
            raise RetrievalError("Re-ranking request failed.") from exc

        content = response.choices[0].message.content
        if not content:
            raise RetrievalError("Re-ranking returned empty content.")

        try:
            parsed = parse_json_object(content)
            result = RerankResult.model_validate(parsed)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RetrievalError("Re-ranking returned invalid JSON.") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        logger.log_event(
            "retriever_rerank_completed",
            latency_ms=round(elapsed_ms, 2),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            candidate_count=len(candidates),
            relevant_count=len(result.relevant_ids),
        )

        logger.info(
            "Rerank result: ids=%s confidence=%.2f threshold=%.2f",
            result.relevant_ids,
            result.confidence,
            self._settings.rerank_confidence_threshold,
        )

        if (
            not result.relevant_ids
            or result.confidence < self._settings.rerank_confidence_threshold
        ):
            return []

        by_id = {memory.id: memory for memory in candidates}
        ranked = [by_id[memory_id] for memory_id in result.relevant_ids if memory_id in by_id]
        return ranked

    async def _generate_text(self, prompt: str) -> str:
        """Generate a short natural-language response."""
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": HIPPO_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                timeout=self._settings.llm_timeout_seconds,
            )
        except APITimeoutError as exc:
            raise RetrievalError("Response generation timed out.") from exc
        except OpenAIError as exc:
            raise RetrievalError("Response generation failed.") from exc

        content = response.choices[0].message.content
        if not content:
            raise RetrievalError("Response generation returned empty content.")

        return content.strip()


def _merge_candidates(
    semantic: Sequence[MemoryRecord],
    keyword: Sequence[MemoryRecord],
    entity: Sequence[MemoryRecord],
    top_k: int,
) -> list[MemoryRecord]:
    """Merge candidate lists while preserving priority and uniqueness."""
    merged: list[MemoryRecord] = []
    seen: set[str] = set()

    for memory in (*semantic, *keyword, *entity):
        if memory.id in seen:
            continue
        seen.add(memory.id)
        merged.append(memory)
        if len(merged) >= top_k:
            break

    return merged


_RERANK_SKIP_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "in",
        "on",
        "at",
        "to",
        "of",
        "my",
        "where",
        "what",
        "how",
        "are",
        "was",
        "do",
        "did",
    }
)


def _query_tokens(query: str) -> list[str]:
    """Extract meaningful tokens from a retrieval query."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 1 and token not in _RERANK_SKIP_STOPWORDS
    ]


def _is_confident_match(query: str, candidates: Sequence[MemoryRecord]) -> bool:
    """Return True when keyword-style matches are strong enough to skip reranking."""
    if not candidates or len(candidates) > 3:
        return False

    tokens = _query_tokens(query)
    if not tokens:
        return False

    haystacks = [
        f"{memory.title} {memory.content}".lower() for memory in candidates
    ]
    if len(candidates) == 1:
        matches = sum(1 for token in tokens if token in haystacks[0])
        return matches >= max(1, len(tokens) // 2)

    if len(candidates) <= 3:
        shared_subject = tokens[0]
        return all(shared_subject in haystack for haystack in haystacks)

    return False
