from __future__ import annotations
import argparse
import asyncio
import json
import os
import statistics
import time
import httpx

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("GATEWAY_URL", "http://localhost:8000"))
    parser.add_argument("--token", default=os.getenv("RAG_TOKEN", ""), required=False)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--query", default="What does the enterprise API key policy specify?")
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("Set --token or RAG_TOKEN")
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    statuses: list[int] = []
    async with httpx.AsyncClient(timeout=70) as client:
        async def send(index: int) -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(f"{args.url}/v1/query", headers={"Authorization": f"Bearer {args.token}"},
                                                 json={"query": f"{args.query} request {index}"})
                    statuses.append(response.status_code)
                except Exception:
                    statuses.append(0)
                latencies.append((time.perf_counter() - started) * 1000)
        await asyncio.gather(*(send(i) for i in range(args.requests)))
    ordered = sorted(latencies)
    percentile = lambda p: ordered[min(len(ordered)-1, int((len(ordered)-1)*p))] if ordered else 0
    report = {"requests": args.requests, "concurrency": args.concurrency,
              "success": statuses.count(200), "rate_limited": statuses.count(429),
              "errors": len(statuses) - statuses.count(200) - statuses.count(429),
              "latency_ms": {"mean": statistics.mean(latencies) if latencies else 0,
                             "p50": percentile(.50), "p95": percentile(.95), "p99": percentile(.99)}}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
