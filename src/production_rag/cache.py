from __future__ import annotations

import json
import time
from typing import Any

from .embeddings import Embedder, HashEmbedder, build_embedder, cosine_similarity
from .telemetry import CACHE_HITS


class SemanticCache:
    def __init__(self, embedder: Embedder | None = None, threshold: float = 0.94,
                 ttl_seconds: int = 300, max_entries: int = 500) -> None:
        self.embedder = embedder or build_embedder()
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.items: list[dict[str, Any]] = []

    async def get(self, query: str, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        vector = self.embedder.embed(query)
        now = time.time()
        best: tuple[float, dict[str, Any]] | None = None
        self.items = [item for item in self.items if item["expires"] > now]
        for item in self.items:
            if item["tenant_id"] != tenant_id or item["user_id"] != user_id:
                continue
            score = cosine_similarity(vector, item["embedding"])
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, item)
        if best:
            CACHE_HITS.inc()
            return dict(best[1]["response"])
        return None

    async def set(self, query: str, tenant_id: str, user_id: str, response: dict[str, Any]) -> None:
        now = time.time()
        self.items = [item for item in self.items if item["expires"] > now]
        self.items.append({"tenant_id": tenant_id, "user_id": user_id,
                           "embedding": self.embedder.embed(query), "response": response,
                           "expires": now + self.ttl_seconds})
        if len(self.items) > self.max_entries:
            self.items = self.items[-self.max_entries:]


class RedisSemanticCache:
    """Tenant/user-scoped semantic cache; requires Redis and an embedding backend."""
    def __init__(self, url: str, embedder: Embedder | None = None,
                 threshold: float = 0.94, ttl_seconds: int = 300, max_entries: int = 200) -> None:
        if __import__("os").getenv("APP_ENV", "development") == "production" and not url.startswith("rediss://"):
            raise RuntimeError("Production REDIS_URL must use TLS (rediss://)")
        try:
            import redis.asyncio as redis
        except ImportError as exc:
            raise RuntimeError("Install the 'redis' extra to use Redis cache") from exc
        self.client = redis.from_url(url, decode_responses=True)
        self.embedder = embedder or build_embedder()
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

    async def get(self, query: str, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        key = f"semantic:{tenant_id}:{user_id}"
        members = await self.client.hgetall(key)
        vector = self.embedder.embed(query)
        best_score, best = 0.0, None
        for raw in members.values():
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if item.get("expires", 0) < time.time():
                continue
            from .embeddings import cosine_similarity
            score = cosine_similarity(vector, item["embedding"])
            if score > best_score:
                best_score, best = score, item
        if best and best_score >= self.threshold:
            CACHE_HITS.inc()
            return best["response"]
        return None

    async def set(self, query: str, tenant_id: str, user_id: str, response: dict[str, Any]) -> None:
        key = f"semantic:{tenant_id}:{user_id}"
        item_id = str(time.time_ns())
        value = json.dumps({"embedding": self.embedder.embed(query), "response": response,
                            "expires": time.time() + self.ttl_seconds})
        members = await self.client.hgetall(key)
        stale: list[str] = []
        for field, raw in members.items():
            try:
                if float(json.loads(raw).get("expires", 0)) < time.time():
                    stale.append(field)
            except (json.JSONDecodeError, TypeError, ValueError):
                stale.append(field)
        if stale:
            await self.client.hdel(key, *stale)
        members = {field: raw for field, raw in members.items() if field not in set(stale)}
        if len(members) >= self.max_entries:
            oldest = sorted(members)[: len(members) - self.max_entries + 1]
            if oldest:
                await self.client.hdel(key, *oldest)
        await self.client.hset(key, item_id, value)
        await self.client.expire(key, self.ttl_seconds)
