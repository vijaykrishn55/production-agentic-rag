CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(384) NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_chunks_tenant_idx ON rag_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS rag_chunks_search_idx ON rag_chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS rag_chunks_metadata_idx ON rag_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops);
