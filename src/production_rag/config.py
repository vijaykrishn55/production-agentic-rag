from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    service_role: str = os.getenv("SERVICE_ROLE", "gateway")
    environment: str = os.getenv("APP_ENV", "development")
    jwt_secret: str = os.getenv("JWT_SECRET", "local-development-secret-change-me")
    jwt_issuer: str = os.getenv("JWT_ISSUER", "production-agentic-rag")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "50"))
    request_token_budget: int = int(os.getenv("REQUEST_TOKEN_BUDGET", "8192"))
    generation_reserve: int = int(os.getenv("GENERATION_TOKEN_RESERVE", "900"))
    evaluation_reserve: int = int(os.getenv("EVALUATION_TOKEN_RESERVE", "500"))
    max_agent_iterations: int = min(3, max(1, int(os.getenv("MAX_AGENT_ITERATIONS", "3"))))
    max_retrieval_candidates: int = int(os.getenv("MAX_RETRIEVAL_CANDIDATES", "20"))
    max_context_passages: int = int(os.getenv("MAX_CONTEXT_PASSAGES", "5"))
    rerank_timeout_seconds: float = float(os.getenv("RERANK_TIMEOUT_SECONDS", "0.22"))
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "hash")
    reranker_backend: str = os.getenv("RERANKER_BACKEND", "lexical")
    postgres_dsn: str = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY", ""))
    llm_model: str = os.getenv("LLM_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
    llm_enabled: bool = _bool("LLM_ENABLED", False)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    cache_enabled: bool = _bool("CACHE_ENABLED", True)
    ragas_enabled: bool = _bool("RAGAS_ENABLED", False)
    otel_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    service_name: str = os.getenv("OTEL_SERVICE_NAME", "production-agentic-rag")


settings = Settings()
