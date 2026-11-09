from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import httpx
from dotenv import load_dotenv
load_dotenv()
from production_rag.evaluation import load_jsonl, run_ragas_evaluation

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ragas",), default="ragas")
    parser.add_argument("--dataset", default="data/evaluation/golden.jsonl")
    parser.add_argument("--tenant", default="demo-tenant")
    parser.add_argument("--orchestrator", default=os.getenv("ORCHESTRATOR_URL", "http://localhost:8001"))
    parser.add_argument("--internal-token", default=os.getenv("INTERNAL_SERVICE_TOKEN", ""))
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", default="reports/live-ragas-evaluation.json")
    args = parser.parse_args()
    if not args.internal_token:
        raise SystemExit("Set INTERNAL_SERVICE_TOKEN or pass --internal-token")
    rows = load_jsonl(args.dataset)
    generated = []
    async with httpx.AsyncClient(timeout=90) as client:
        for row in rows:
            response = await client.post(f"{args.orchestrator}/internal/query",
                headers={"X-Internal-Token": args.internal_token},
                json={"query": row["question"], "tenant_id": args.tenant,
                      "user_id": "evaluation-runner", "include_contexts": True})
            response.raise_for_status()
            result = response.json()
            generated.append({"question": row["question"], "answer": result["answer"],
                "contexts": result.get("retrieved_contexts", []),
                "reference": row.get("reference", "")})
    evaluation = run_ragas_evaluation(generated, args.threshold)
    evaluation["generated_samples"] = generated
    evaluation["tenant_id"] = args.tenant
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    raise SystemExit(0 if evaluation["passed"] else 1)

if __name__ == "__main__":
    asyncio.run(main())
