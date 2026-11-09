from __future__ import annotations

import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ..config import settings
from ..telemetry import configure_tracing
from ..ingestion import DocumentIngestor
from ..internal_auth import require_internal_token
from ..repository import InMemoryRepository, PostgresRepository


class SearchBody(BaseModel):
    query: str
    tenant_id: str
    top_k: int = Field(default=20, ge=1, le=100)
    metadata_filter: dict[str, str | int | bool] = Field(default_factory=dict)


class IngestBody(BaseModel):
    document_id: str
    tenant_id: str
    source: str
    text: str
    metadata: dict[str, str | int | bool] = Field(default_factory=dict)


use_pg = os.getenv("USE_POSTGRES", "false").lower() == "true"
try:
    repository = PostgresRepository(settings.postgres_dsn) if use_pg else InMemoryRepository()
except Exception:
    if os.getenv("APP_ENV", "development") == "production":
        raise
    repository = InMemoryRepository()
ingestor = DocumentIngestor()
@asynccontextmanager
async def lifespan(app: FastAPI):
    await seed_demo_data()
    configure_tracing(app)
    yield


app = FastAPI(title="RAG Retrieval Service", version="1.0.0", lifespan=lifespan)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "retrieval"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    try:
        if not repository.ready():
            raise RuntimeError("Repository is not ready")
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Retrieval backend is not ready") from exc


@app.post("/internal/search")
async def search(body: SearchBody, _: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        results = repository.search_hybrid(body.query, body.tenant_id, body.top_k, body.metadata_filter)
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Retrieval backend unavailable") from exc


@app.post("/internal/ingest")
async def ingest(body: IngestBody, _: None = Depends(require_internal_token)) -> dict[str, int]:
    chunks = ingestor.ingest(body.document_id, body.text, body.source, body.tenant_id, body.metadata)
    return {"chunks_added": repository.replace_document(body.document_id, body.tenant_id, chunks)}


async def seed_demo_data() -> None:
    if os.getenv("SEED_DEMO_DATA", "true").lower() != "true":
        return
    if getattr(repository, "chunks", None) is not None and repository.chunks:
        return
    base = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "knowledge")
    folder = os.path.abspath(base)
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if name.endswith(".md") or name.endswith(".txt"):
            text = open(os.path.join(folder, name), encoding="utf-8").read()
            document_id = name.rsplit(".", 1)[0]
            chunks = ingestor.ingest(document_id, text, name, "demo-tenant", {"domain": "enterprise"})
            repository.replace_document(document_id, "demo-tenant", chunks)

