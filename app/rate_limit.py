"""Shared rate limiting with optional Redis backing."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Protocol

logger = logging.getLogger(__name__)


class RateLimiter(Protocol):
    """Rate limiter contract used by middleware and route handlers."""

    async def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        """Return True when the request should be allowed."""


class MemoryRateLimiter:
    """Process-local fixed-window rate limiter."""

    def __init__(self) -> None:
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        now = time.time()
        bucket = self._buckets[key]
        bucket[:] = [
            timestamp for timestamp in bucket if now - timestamp < window_seconds
        ]
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


class RedisRateLimiter:
    """Fixed-window rate limiter backed by Redis."""

    def __init__(self, redis_url: str) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "redis package is required when REDIS_URL is configured."
            ) from exc

        self._redis = Redis.from_url(redis_url, decode_responses=True)

    async def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        window_id = int(time.time()) // window_seconds
        redis_key = f"hippo:rl:{key}:{window_id}"
        count = await self._redis.incr(redis_key)
        if count == 1:
            await self._redis.expire(redis_key, window_seconds)
        return int(count) <= limit


def create_rate_limiter(redis_url: str | None) -> RateLimiter:
    """Create a distributed limiter when Redis is configured."""
    if not redis_url:
        return MemoryRateLimiter()

    try:
        return RedisRateLimiter(redis_url)
    except Exception:
        logger.exception("Failed to initialize Redis rate limiter; using in-memory fallback.")
        return MemoryRateLimiter()
