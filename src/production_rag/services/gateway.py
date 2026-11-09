from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..auth import Principal, require_principal
from ..config import settings
from ..telemetry import configure_tracing
from ..models import DocumentInput, EvaluationRequest, QueryRequest
from ..internal_auth import require_internal_token
from ..rate_limit import InMemoryRateLimiter, RedisRateLimiter
from ..telemetry import REQUESTS, log_event

_production = os.getenv("APP_ENV", "development") == "production"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_tracing(app)
    yield


app = FastAPI(title="Production Agentic RAG API", version="1.0.0",
              docs_url=None if _production else "/docs", redoc_url=None if _production else "/redoc",
              lifespan=lifespan)
INTERNAL_HEADERS = {"X-Internal-Token": os.getenv("INTERNAL_SERVICE_TOKEN", "")}
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8001").rstrip("/")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8002").rstrip("/")
try:
    limiter: Any = RedisRateLimiter(settings.redis_url, settings.rate_limit_per_minute)
except Exception:
    if os.getenv("APP_ENV", "development") == "production":
        raise
    limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)


async def enforce_user_rate_limit(principal: Principal) -> None:
    key = f"{principal.tenant_id}:{principal.user_id}"
    if not await limiter.allow(key):
        REQUESTS.labels(status="rate_limited").inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 50 requests per minute per user")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            a = await client.get(f"{ORCHESTRATOR_URL}/ready")
            b = await client.get(f"{RETRIEVAL_URL}/ready")
            a.raise_for_status(); b.raise_for_status()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="A downstream service is not ready") from exc


@app.post("/v1/query")
async def query(body: QueryRequest, principal: Principal = Depends(require_principal)) -> dict[str, Any]:
    await enforce_user_rate_limit(principal)
    payload = {"query": body.query, "tenant_id": principal.tenant_id, "user_id": principal.user_id,
               "top_k": body.top_k, "metadata_filter": body.metadata_filter,
               "use_external_search": body.use_external_search}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{ORCHESTRATOR_URL}/internal/query", headers=INTERNAL_HEADERS, json=payload)
            response.raise_for_status()
        REQUESTS.labels(status="success").inc()
        return response.json()
    except httpx.HTTPStatusError as exc:
        REQUESTS.labels(status="downstream_error").inc()
        raise HTTPException(status=502, detail="Orchestrator rejected the request") from exc
    except Exception as exc:
        REQUESTS.labels(status="unavailable").inc()
        raise HTTPException(status=503, detail="Orchestrator unavailable") from exc


@app.post("/v1/documents")
async def ingest(body: DocumentInput, principal: Principal = Depends(require_principal)) -> dict[str, int]:
    await enforce_user_rate_limit(principal)
    payload = {"document_id": body.document_id, "tenant_id": principal.tenant_id,
               "source": body.source, "text": body.text, "metadata": body.metadata}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{RETRIEVAL_URL}/internal/ingest", headers=INTERNAL_HEADERS, json=payload)
            response.raise_for_status()
        return response.json()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document ingestion service unavailable") from exc


@app.post("/v1/documents/upload")
async def upload(file: UploadFile = File(...), principal: Principal = Depends(require_principal)) -> dict[str, int]:
    await enforce_user_rate_limit(principal)
    from ..ingestion import DocumentIngestor
    from ..security import SecurityInterceptor
    import tempfile
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    content = await file.read(20_000_001)
    if len(content) > 20_000_000:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit")
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(content); temp.flush()
        try:
            text = DocumentIngestor().parse_file(temp.name)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
    payload = {"document_id": Path(file.filename or "upload").stem, "tenant_id": principal.tenant_id,
               "source": file.filename or "uploaded_file", "text": text, "metadata": {"media_type": file.content_type or "unknown"}}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{RETRIEVAL_URL}/internal/ingest", headers=INTERNAL_HEADERS, json=payload)
        response.raise_for_status()
    return response.json()


@app.post("/v1/evaluations")
async def submit_evaluation(body: EvaluationRequest, principal: Principal = Depends(require_principal)) -> dict[str, str]:
    await enforce_user_rate_limit(principal)
    job_id = uuid.uuid4().hex
    try:
        import redis.asyncio as redis
        queue = redis.from_url(settings.redis_url, decode_responses=True)
        await queue.lpush("rag:evaluation:queue", __import__("json").dumps({
            "job_id": job_id, "tenant_id": principal.tenant_id, "dataset": body.dataset,
            "threshold": body.threshold,
        }))
        await queue.close()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Redis queue client is not installed") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Evaluation queue unavailable") from exc
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/evaluations/{job_id}")
async def evaluation_status(job_id: str, principal: Principal = Depends(require_principal)) -> dict[str, Any]:
    await enforce_user_rate_limit(principal)
    try:
        import redis.asyncio as redis
        client = redis.from_url(settings.redis_url, decode_responses=True)
        raw = await client.get(f"rag:evaluation:result:{principal.tenant_id}:{job_id}")
        await client.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Evaluation store unavailable") from exc
    if not raw:
        return {"job_id": job_id, "status": "pending_or_not_found"}
    return __import__("json").loads(raw)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ops/dashboard")
async def dashboard() -> dict[str, str]:
    return {"grafana": os.getenv("GRAFANA_URL", "http://localhost:3000"),
            "prometheus": os.getenv("PROMETHEUS_URL", "http://localhost:9090")}

