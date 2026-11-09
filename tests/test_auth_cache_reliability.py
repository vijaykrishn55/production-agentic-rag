import asyncio
import time
import pytest
from fastapi import HTTPException
from production_rag.auth import create_token, verify_token
from production_rag.cache import SemanticCache
from production_rag.rate_limit import InMemoryRateLimiter
from production_rag.reliability import CircuitBreaker, CircuitState, retry_async


def test_jwt_round_trip():
    token = create_token("alice", "tenant-a", secret="test-secret")
    principal = verify_token(token, secret="test-secret")
    assert principal.user_id == "alice"
    assert principal.tenant_id == "tenant-a"


def test_jwt_rejects_wrong_secret():
    token = create_token("alice", "tenant-a", secret="one")
    with pytest.raises(HTTPException):
        verify_token(token, secret="two")


def test_jwt_rejects_expired_token():
    token = create_token("alice", "tenant-a", secret="test-secret", expires_seconds=-1)
    with pytest.raises(HTTPException):
        verify_token(token, secret="test-secret")


def test_rate_limiter_enforces_window():
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)
    async def run():
        assert await limiter.allow("tenant:user")
        assert await limiter.allow("tenant:user")
        assert not await limiter.allow("tenant:user")
    asyncio.run(run())


def test_semantic_cache_is_scoped_by_tenant_and_user():
    cache = SemanticCache(threshold=0.9)
    async def run():
        await cache.set("budget for cloud generative AI", "tenant-a", "alice", {"answer": "45 percent"})
        assert await cache.get("budget for cloud generative AI", "tenant-a", "alice") == {"answer": "45 percent"}
        assert await cache.get("budget for cloud generative AI", "tenant-b", "alice") is None
        assert await cache.get("budget for cloud generative AI", "tenant-a", "bob") is None
    asyncio.run(run())


def test_circuit_breaker_opens_and_recovers():
    async def run():
        breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=0.01)
        await breaker.failure(); await breaker.failure()
        assert breaker.state == CircuitState.OPEN
        assert not await breaker.allow()
        time.sleep(0.02)
        assert await breaker.allow()
        await breaker.success()
        assert breaker.state == CircuitState.CLOSED
    asyncio.run(run())


def test_retry_async_retries_transient_error():
    async def run():
        calls = {"count": 0}
        async def operation():
            calls["count"] += 1
            if calls["count"] < 2:
                raise ValueError("transient")
            return "ok"
        result = await retry_async(operation, attempts=2, base_delay=0.001)
        assert result == "ok"
        assert calls["count"] == 2
    asyncio.run(run())


def test_limiter_handles_one_hundred_concurrent_requests():
    limiter = InMemoryRateLimiter(limit=50, window_seconds=60)
    async def run():
        results = await asyncio.gather(*(limiter.allow("tenant:load-user") for _ in range(100)))
        assert sum(results) == 50
    asyncio.run(run())


def test_semantic_cache_prunes_oldest_entries_when_bounded():
    cache = SemanticCache(max_entries=2)
    async def run():
        await cache.set("first unique query", "tenant-a", "alice", {"answer": "one"})
        await cache.set("second unique query", "tenant-a", "alice", {"answer": "two"})
        await cache.set("third unique query", "tenant-a", "alice", {"answer": "three"})
        assert len(cache.items) == 2
        assert [item["response"]["answer"] for item in cache.items] == ["two", "three"]
    asyncio.run(run())
