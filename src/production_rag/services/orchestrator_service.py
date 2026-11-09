from __future__ import annotations

import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import settings
from ..telemetry import configure_tracing
from ..models import InternalQuery
from ..internal_auth import require_internal_token
from ..orchestrator import AgentOrchestrator
from ..repository import InMemoryRepository, PostgresRepository
from .repository_client import RemoteRepository

retrieval_url = os.getenv("RETRIEVAL_URL", "")
if retrieval_url:
    repo = RemoteRepository(retrieval_url)
else:
    use_pg = os.getenv("USE_POSTGRES", "false").lower() == "true"
    repo = PostgresRepository(settings.postgres_dsn) if use_pg else InMemoryRepository()
agent = AgentOrchestrator(repository=repo)
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_tracing(app)
    yield


app = FastAPI(title="RAG Agent Orchestrator", version="1.0.0", lifespan=lifespan)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "orchestrator"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    try:
        if hasattr(agent.repository, "ready") and not agent.repository.ready():
            raise RuntimeError("Repository is not ready")
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Orchestrator dependencies are not ready") from exc


@app.post("/internal/query")
async def query(body: InternalQuery, _: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        return await agent.run(body.query, body.tenant_id, body.user_id,
                               body.top_k, body.metadata_filter, body.use_external_search,
                               body.include_contexts)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent execution failed") from exc

