from __future__ import annotations
import argparse
import asyncio
import os
from pathlib import Path
import httpx
from production_rag.auth import create_token

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="demo-tenant")
    parser.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://localhost:8000"))
    args = parser.parse_args()
    token = os.getenv("RAG_TOKEN") or create_token("seed-operator", args.tenant)
    root = Path(__file__).resolve().parents[1]
    async with httpx.AsyncClient(timeout=30) as client:
        for path in (root / "data/knowledge").glob("*"):
            if path.is_file():
                payload = {"document_id": path.stem, "source": path.name,
                           "text": path.read_text(encoding="utf-8"), "metadata": {"domain": "enterprise"}}
                response = await client.post(f"{args.gateway}/v1/documents", headers={"Authorization": f"Bearer {token}"}, json=payload)
                response.raise_for_status()
                print(path.name, response.json())

if __name__ == "__main__":
    asyncio.run(main())
