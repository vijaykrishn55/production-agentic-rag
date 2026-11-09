from __future__ import annotations

import httpx
import os
from typing import Any


class RemoteRepository:
    """Synchronous repository adapter used by the async orchestrator's worker threads."""
    def __init__(self, base_url: str, timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"X-Internal-Token": os.getenv("INTERNAL_SERVICE_TOKEN", "")}

    def ready(self) -> bool:
        response = httpx.get(f"{self.base_url}/ready", headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("status") == "ready"

    def search_hybrid(self, query: str, tenant_id: str, top_k: int = 20,
                      metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        response = httpx.post(f"{self.base_url}/internal/search", headers=self.headers, json={
            "query": query, "tenant_id": tenant_id, "top_k": top_k,
            "metadata_filter": metadata_filter or {},
        }, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["results"]
