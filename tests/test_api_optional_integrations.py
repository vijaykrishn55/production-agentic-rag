import os
import pytest
from fastapi.testclient import TestClient


def test_gateway_rejects_missing_bearer_token():
    from production_rag.services.gateway import app
    client = TestClient(app)
    response = client.post("/v1/query", json={"query": "What is API key rotation?"})
    assert response.status_code == 401


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="PostgreSQL integration requires TEST_DATABASE_URL")
def test_postgres_integration_smoke():
    from production_rag.repository import PostgresRepository
    repo = PostgresRepository(os.environ["TEST_DATABASE_URL"])
    assert repo is not None


@pytest.mark.skipif(not os.getenv("TEST_REDIS_URL"), reason="Redis integration requires TEST_REDIS_URL")
def test_redis_integration_smoke():
    import asyncio
    import redis.asyncio as redis
    async def run():
        client = redis.from_url(os.environ["TEST_REDIS_URL"])
        assert await client.ping()
        await client.aclose()
    asyncio.run(run())
