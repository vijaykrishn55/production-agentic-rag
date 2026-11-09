from production_rag.embeddings import HashEmbedder, cosine_similarity
from production_rag.ingestion import DocumentIngestor
from production_rag.repository import InMemoryRepository


def test_ingestion_chunks_document_and_masks_pii():
    ingestor = DocumentIngestor(chunk_chars=80, overlap_chars=10)
    chunks = ingestor.ingest("d1", "Contact user@example.com. " + ("Enterprise policy text. " * 15), "policy", "t1")
    assert len(chunks) > 1
    assert all("user@example.com" not in item["text"] for item in chunks)
    assert any("email" in item["metadata"]["pii_redactions"] for item in chunks)


def test_chunking_empty_text_returns_empty_list():
    assert DocumentIngestor().chunk_text(" \n ") == []


def test_hash_embedding_is_deterministic_and_normalized():
    embedder = HashEmbedder()
    first = embedder.embed("enterprise retrieval policy")
    second = embedder.embed("enterprise retrieval policy")
    assert first == second
    assert abs(sum(x * x for x in first) ** 0.5 - 1.0) < 1e-6


def test_cosine_similarity_bounds():
    embedder = HashEmbedder()
    vector = embedder.embed("same text")
    assert cosine_similarity(vector, vector) > 0.999
    assert -1 <= cosine_similarity(vector, embedder.embed("different topic")) <= 1


def test_hybrid_search_returns_relevant_document():
    repo = InMemoryRepository()
    repo.upsert_chunks([
        {"chunk_id": "a", "document_id": "budget", "tenant_id": "t1", "source": "budget.md",
         "text": "45 percent of cloud infrastructure spending goes to generative AI."},
        {"chunk_id": "b", "document_id": "security", "tenant_id": "t1", "source": "security.md",
         "text": "API keys must rotate every 90 days."},
    ])
    results = repo.search_hybrid("cloud infrastructure generative AI percentage", "t1")
    assert results[0]["chunk_id"] == "a"


def test_retrieval_is_tenant_isolated():
    repo = InMemoryRepository()
    repo.upsert_chunks([
        {"chunk_id": "a", "document_id": "a", "tenant_id": "tenant-a", "text": "secret budget value"},
        {"chunk_id": "b", "document_id": "b", "tenant_id": "tenant-b", "text": "secret budget value"},
    ])
    results = repo.search_hybrid("secret budget", "tenant-a")
    assert {row["tenant_id"] for row in results} == {"tenant-a"}


def test_metadata_filter_applied():
    repo = InMemoryRepository()
    repo.upsert_chunks([
        {"chunk_id": "a", "document_id": "a", "tenant_id": "t1", "text": "cloud budget", "metadata": {"domain": "finance"}},
        {"chunk_id": "b", "document_id": "b", "tenant_id": "t1", "text": "cloud budget", "metadata": {"domain": "security"}},
    ])
    results = repo.search_hybrid("cloud budget", "t1", metadata_filter={"domain": "finance"})
    assert [row["chunk_id"] for row in results] == ["a"]


def test_ingestion_chunk_ids_are_stable_and_tenant_scoped():
    ingestor = DocumentIngestor()
    first = ingestor.ingest("policy", "API keys rotate every 90 days.", "policy.md", "tenant-a")
    repeated = ingestor.ingest("policy", "API keys rotate every 90 days.", "policy.md", "tenant-a")
    other_tenant = ingestor.ingest("policy", "API keys rotate every 90 days.", "policy.md", "tenant-b")
    assert first[0]["chunk_id"] == repeated[0]["chunk_id"]
    assert first[0]["chunk_id"] != other_tenant[0]["chunk_id"]


def test_replace_document_removes_old_chunks_without_cross_tenant_deletion():
    repo = InMemoryRepository()
    ingestor = DocumentIngestor()
    old = ingestor.ingest("policy", "Old API key rotation policy.", "policy.md", "tenant-a")
    other = ingestor.ingest("policy", "Tenant B private policy.", "policy.md", "tenant-b")
    repo.upsert_chunks(old + other)
    new = ingestor.ingest("policy", "New API key rotation policy.", "policy.md", "tenant-a")
    repo.replace_document("policy", "tenant-a", new)
    tenant_a = repo.search_hybrid("New API key rotation", "tenant-a")
    tenant_b = repo.search_hybrid("Tenant B private policy", "tenant-b")
    assert tenant_a and all("Old API key" not in row["text"] for row in tenant_a)
    assert tenant_b and any("Tenant B" in row["text"] for row in tenant_b)
