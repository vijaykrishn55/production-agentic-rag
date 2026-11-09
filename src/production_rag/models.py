from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=12000)
    top_k: int = Field(default=20, ge=1, le=100)
    metadata_filter: dict[str, str | int | bool] = Field(default_factory=dict)
    use_external_search: bool = False
    include_contexts: bool = False


class Citation(BaseModel):
    source_id: str
    document_id: str
    chunk_id: str
    quote: str = ""


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    evaluation_passed: bool
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    answer_relevance_score: float = Field(ge=0.0, le=1.0)
    iterations: int = 1
    retrieved_count: int = 0
    context_tokens: int = 0
    rerank_latency_ms: float = 0.0
    reranker_backend: str = "lexical"
    tools_used: list[str] = Field(default_factory=list)
    security_events: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    trace_id: str = ""
    cached: bool = False
    refusal: bool = False


class DocumentInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=180)
    text: str = Field(min_length=1, max_length=1_000_000)
    source: str = Field(default="user_upload", max_length=256)
    metadata: dict[str, str | int | bool] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    dataset: str = "data/evaluation/golden.jsonl"
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("dataset")
    @classmethod
    def validate_dataset_path(cls, value: str) -> str:
        from pathlib import PurePosixPath, PureWindowsPath
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
            raise ValueError("Dataset must be a relative path under data/evaluation")
        if not value.replace("\\", "/").startswith("data/evaluation/") or not value.lower().endswith(".jsonl"):
            raise ValueError("Dataset must be a JSONL file under data/evaluation")
        return value


class InternalQuery(BaseModel):
    query: str
    tenant_id: str
    user_id: str
    top_k: int = 20
    metadata_filter: dict[str, str | int | bool] = Field(default_factory=dict)
    use_external_search: bool = False
    include_contexts: bool = False
