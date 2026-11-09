from __future__ import annotations

import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    def __init__(self, limit: int = 50, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self.events[key]
        while bucket and bucket[0] <= now - self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


class RedisRateLimiter:
    def __init__(self, url: str, limit: int = 50, window_seconds: int = 60) -> None:
        if __import__("os").getenv("APP_ENV", "development") == "production" and not url.startswith("rediss://"):
            raise RuntimeError("Production REDIS_URL must use TLS (rediss://)")
        try:
            import redis.asyncio as redis
        except ImportError as exc:
            raise RuntimeError("Install the 'redis' extra to use Redis rate limiting") from exc
        self.client = redis.from_url(url, decode_responses=True)
        self.limit = limit
        self.window_seconds = window_seconds

    async def allow(self, key: str) -> bool:
        now_ms = int(time.time() * 1000)
        redis_key = f"rl:{key}"
        member = f"{now_ms}:{time.time_ns()}"
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(redis_key, 0, now_ms - self.window_seconds * 1000)
            pipe.zadd(redis_key, {member: now_ms})
            pipe.zcard(redis_key)
            pipe.pexpire(redis_key, self.window_seconds * 1000)
            _, _, count, _ = await pipe.execute()
        if count <= self.limit:
            return True
        # Remove only the request that exceeded the allowance.
        await self.client.zrem(redis_key, member)
        return False
