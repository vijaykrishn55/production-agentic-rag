from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from ..config import settings
from ..evaluation import load_jsonl, run_proxy_evaluation, run_ragas_evaluation
from ..telemetry import log_event


async def process_job(redis_client: Any, job: dict[str, Any]) -> dict[str, Any]:
    if not settings.ragas_enabled:
        raise RuntimeError("Set RAGAS_ENABLED=true before starting the evaluation worker")
    rows = load_jsonl(job.get("dataset", "data/evaluation/golden.jsonl"))
    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8001")
    internal_token = os.getenv("INTERNAL_SERVICE_TOKEN", "")
    if not internal_token:
        raise RuntimeError("INTERNAL_SERVICE_TOKEN is required by the evaluation worker")
    samples = []
    async with httpx.AsyncClient(timeout=90) as client:
        for row in rows:
            response = await client.post(f"{orchestrator_url}/internal/query",
                headers={"X-Internal-Token": internal_token},
                json={"query": row["question"], "tenant_id": job["tenant_id"],
                      "user_id": "evaluation-worker", "include_contexts": True})
            response.raise_for_status()
            result = response.json()
            samples.append({"question": row["question"], "answer": result["answer"],
                            "contexts": result.get("retrieved_contexts", []),
                            "reference": row.get("reference", "")})
    result = run_ragas_evaluation(samples, float(job.get("threshold", 0.85)))
    result["job_id"] = job["job_id"]
    result["tenant_id"] = job["tenant_id"]
    await redis_client.set(f"rag:evaluation:result:{job['tenant_id']}:{job['job_id']}",
                           json.dumps(result), ex=86400)
    return result


async def main() -> None:
    try:
        import redis.asyncio as redis
    except ImportError as exc:
        raise RuntimeError("Install the redis extra in worker image") from exc
    client = redis.from_url(settings.redis_url, decode_responses=True)
    log_event("evaluation_worker_started")
    while True:
        item = await client.brpop("rag:evaluation:queue", timeout=5)
        if not item:
            continue
        _, raw = item
        job = json.loads(raw)
        try:
            result = await process_job(client, job)
            log_event("evaluation_job_complete", job_id=job["job_id"], passed=result["passed"])
        except Exception as exc:
            await client.set(f"rag:evaluation:result:{job.get('tenant_id')}:{job.get('job_id')}",
                             json.dumps({"job_id": job.get("job_id"), "status": "failed", "error": type(exc).__name__}), ex=86400)
            log_event("evaluation_job_failed", job_id=job.get("job_id"), error=type(exc).__name__)


if __name__ == "__main__":
    asyncio.run(main())
