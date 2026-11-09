from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .embeddings import Embedder, HashEmbedder, build_embedder, cosine_similarity, lexical_score


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    tenant_id: str
    source: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


class InMemoryRepository:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or build_embedder()
        self.chunks: dict[str, Chunk] = {}

    def ready(self) -> bool:
        return True

    def replace_document(self, document_id: str, tenant_id: str, items: list[dict[str, Any]]) -> int:
        self.chunks = {key: chunk for key, chunk in self.chunks.items()
                       if not (chunk.document_id == document_id and chunk.tenant_id == tenant_id)}
        return self.upsert_chunks(items)

    def upsert_chunks(self, items: list[dict[str, Any]]) -> int:
        for item in items:
            chunk_id = str(item["chunk_id"])
            self.chunks[chunk_id] = Chunk(
                chunk_id=chunk_id,
                document_id=str(item["document_id"]),
                tenant_id=str(item.get("tenant_id", "demo-tenant")),
                source=str(item.get("source", item["document_id"])),
                text=str(item["text"]),
                metadata=dict(item.get("metadata", {})),
                embedding=item.get("embedding") or self.embedder.embed(str(item["text"])),
            )
        return len(items)

    def search_hybrid(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed(query)
        dense: list[tuple[float, Chunk]] = []
        sparse: list[tuple[float, Chunk]] = []
        for chunk in self.chunks.values():
            if chunk.tenant_id != tenant_id:
                continue
            if metadata_filter and any(chunk.metadata.get(k) != v for k, v in metadata_filter.items()):
                continue
            dense_score = cosine_similarity(query_vector, chunk.embedding)
            sparse_score = lexical_score(query, chunk.text)
            # Reject zero-evidence matches rather than returning a ranked but irrelevant chunk.
            if sparse_score < 0.12 and dense_score < 0.30:
                continue
            dense.append((dense_score, chunk))
            sparse.append((sparse_score, chunk))
        dense.sort(key=lambda pair: pair[0], reverse=True)
        sparse.sort(key=lambda pair: pair[0], reverse=True)
        rrf: dict[str, float] = {}
        records: dict[str, Chunk] = {}
        for ranking in (dense, sparse):
            for rank, (_, chunk) in enumerate(ranking, start=1):
                rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (60 + rank)
                records[chunk.chunk_id] = chunk
        ordered = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        results = []
        for chunk_id in ordered:
            c = records[chunk_id]
            results.append({
                "chunk_id": c.chunk_id, "document_id": c.document_id,
                "tenant_id": c.tenant_id, "source": c.source, "text": c.text,
                "metadata": c.metadata, "score": rrf[chunk_id],
            })
        return results


class PostgresRepository:
    """PostgreSQL 15 + pgvector repository. Requires the optional postgres extra."""

    def __init__(self, dsn: str, embedder: Embedder | None = None) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
            try:
                from psycopg_pool import ConnectionPool
            except ImportError:
                ConnectionPool = None
        except ImportError as exc:
            raise RuntimeError("Install the 'postgres' extra to use PostgreSQL") from exc
        self.psycopg = psycopg
        self.dict_row = dict_row
        self.dsn = dsn
        self.pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=int(__import__("os").getenv("POSTGRES_POOL_MAX", "10")), open=True) if ConnectionPool else None
        if __import__("os").getenv("APP_ENV", "development") == "production":
            lowered = dsn.lower()
            if "sslmode=require" not in lowered and "sslmode=verify-ca" not in lowered and "sslmode=verify-full" not in lowered:
                raise RuntimeError("Production DATABASE_URL must enforce PostgreSQL TLS with sslmode=require or stronger")
        self.embedder = embedder or build_embedder()

    def ready(self) -> bool:
        connection_context = self.pool.connection() if self.pool else self.psycopg.connect(self.dsn)
        with connection_context as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1

    def _write_chunks(self, items: list[dict[str, Any]], replace_key: tuple[str, str] | None = None) -> int:
        rows = []
        for item in items:
            embedding = item.get("embedding") or self.embedder.embed(item["text"])
            vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
            rows.append((item["chunk_id"], item["document_id"], item.get("tenant_id", "demo-tenant"),
                         item.get("source", item["document_id"]), item["text"],
                         json.dumps(item.get("metadata", {})), vector_literal))
        sql = """
        INSERT INTO rag_chunks (chunk_id, document_id, tenant_id, source, content, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET document_id=EXCLUDED.document_id,
          source=EXCLUDED.source, content=EXCLUDED.content,
          metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding
        """
        connection_context = self.pool.connection() if self.pool else self.psycopg.connect(self.dsn)
        with connection_context as conn:
            with conn.cursor() as cur:
                if replace_key:
                    cur.execute("DELETE FROM rag_chunks WHERE document_id=%s AND tenant_id=%s", replace_key)
                if rows:
                    cur.executemany(sql, rows)
        return len(rows)

    def upsert_chunks(self, items: list[dict[str, Any]]) -> int:
        return self._write_chunks(items)

    def replace_document(self, document_id: str, tenant_id: str,
                         items: list[dict[str, Any]]) -> int:
        return self._write_chunks(items, (document_id, tenant_id))

    def search_hybrid(self, query: str, tenant_id: str, top_k: int = 20,
                      metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        vector = self.embedder.embed(query)
        vector_literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
        sql = """
        WITH dense AS (
          SELECT chunk_id, row_number() OVER (ORDER BY distance) AS rank
          FROM (
            SELECT chunk_id, embedding <=> %s::vector AS distance
            FROM rag_chunks WHERE tenant_id = %s
              AND (%s::jsonb = '{}'::jsonb OR metadata @> %s::jsonb)
          ) candidates
          WHERE distance <= 0.70
          ORDER BY distance LIMIT %s
        ), sparse AS (
          SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC) AS rank
          FROM rag_chunks WHERE tenant_id = %s
            AND (%s::jsonb = '{}'::jsonb OR metadata @> %s::jsonb)
            AND search_vector @@ plainto_tsquery('english', %s)
          ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC LIMIT %s
        ), fused AS (
          SELECT chunk_id, sum(score) AS score FROM (
            SELECT chunk_id, 1.0/(60+rank) AS score FROM dense
            UNION ALL
            SELECT chunk_id, 1.0/(60+rank) AS score FROM sparse
          ) s GROUP BY chunk_id ORDER BY score DESC LIMIT %s
        )
        SELECT c.chunk_id, c.document_id, c.tenant_id, c.source, c.content AS text,
               c.metadata, f.score
        FROM fused f JOIN rag_chunks c USING(chunk_id) ORDER BY f.score DESC
        """
        meta = json.dumps(metadata_filter or {})
        params = (vector_literal, tenant_id, meta, meta, top_k,
                  query, tenant_id, meta, meta, query, query, top_k, top_k)
        connection_context = self.pool.connection() if self.pool else self.psycopg.connect(self.dsn, row_factory=self.dict_row)
        with connection_context as conn:
            with conn.cursor(row_factory=self.dict_row) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    def close(self) -> None:
        if self.pool:
            self.pool.close()


def build_repository(dsn: str, use_postgres: bool) -> InMemoryRepository | PostgresRepository:
    if use_postgres:
        return PostgresRepository(dsn)
    return InMemoryRepository()
