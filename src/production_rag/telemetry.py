from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("rag_requests_total", "RAG requests", ["status"])
REQUEST_LATENCY = Histogram("rag_request_latency_seconds", "End-to-end request latency")
RERANK_LATENCY = Histogram("rag_rerank_latency_seconds", "Reranking latency")
TOKEN_USAGE = Counter("rag_prompt_tokens_total", "Estimated prompt token usage")
SECURITY_EVENTS = Counter("rag_security_events_total", "Security events", ["kind"])
EVALUATION_SCORE = Gauge("rag_latest_faithfulness_score", "Latest faithfulness score")
CACHE_HITS = Counter("rag_semantic_cache_hits_total", "Semantic cache hits")
CONTEXT_COMPRESSION_RATIO = Histogram("rag_context_compression_ratio", "Selected/original evidence character ratio")

try:
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _log = structlog.get_logger("production_rag")
    _structured = True
except ImportError:
    _fallback = logging.getLogger("production_rag")
    if not _fallback.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        _fallback.addHandler(handler)
        _fallback.setLevel(logging.INFO)
    _log = _fallback
    _structured = False


def trace_id() -> str:
    try:
        from opentelemetry import trace
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return format(context.trace_id, "032x")
    except ImportError:
        pass
    return uuid.uuid4().hex


def log_event(event: str, **fields: object) -> None:
    if _structured:
        _log.info(event, **fields)
    else:
        _log.info(json.dumps({"timestamp": time.time(), "event": event, **fields}, default=str))


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    result = {"milliseconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["milliseconds"] = (time.perf_counter() - start) * 1000


@contextmanager
def span(name: str):
    try:
        from opentelemetry import trace
    except ImportError:
        with nullcontext() as current:
            yield current
        return
    tracer = trace.get_tracer("production-agentic-rag")
    with tracer.start_as_current_span(name) as current:
        yield current


_httpx_instrumented = False


def configure_tracing(app=None) -> None:
    """Configure OTLP tracing and instrument FastAPI/HTTPX when extras are installed."""
    global _httpx_instrumented
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        import os

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            return
        provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "production-agentic-rag")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        log_event("otel_tracing_configured", endpoint=endpoint)
    except ImportError:
        log_event("otel_unavailable", message="Install the observability extra to export traces")
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        if app is not None:
            FastAPIInstrumentor.instrument_app(app)
        if not _httpx_instrumented:
            HTTPXClientInstrumentor().instrument()
            _httpx_instrumented = True
    except ImportError:
        pass
