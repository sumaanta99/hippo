"""Tests for rate limiting helpers."""

from __future__ import annotations

import pytest

from rate_limit import MemoryRateLimiter


@pytest.mark.asyncio
async def test_memory_rate_limiter_blocks_after_limit() -> None:
    """The in-memory limiter should enforce fixed-window limits."""
    limiter = MemoryRateLimiter()
    for _ in range(3):
        assert await limiter.allow("user-a", limit=3, window_seconds=60)

    assert not await limiter.allow("user-a", limit=3, window_seconds=60)


@pytest.mark.asyncio
async def test_memory_rate_limiter_is_isolated_by_key() -> None:
    """Different keys should not share counters."""
    limiter = MemoryRateLimiter()
    assert await limiter.allow("user-a", limit=1, window_seconds=60)
    assert await limiter.allow("user-b", limit=1, window_seconds=60)
