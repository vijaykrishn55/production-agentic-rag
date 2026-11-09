from __future__ import annotations

import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from .config import settings
from .embeddings import lexical_score
from .telemetry import RERANK_LATENCY


class TokenCounter:
    def __init__(self) -> None:
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self.encoder = None

    def count(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return max(1, (len(text) + 3) // 4) if text else 0


class ContextBudgetExceeded(RuntimeError):
    pass


@dataclass
class ContextPackage:
    text: str
    chunks: list[dict[str, Any]]
    token_count: int
    budget: int
    rerank_ms: float
    compression_ratio: float
    reranker_backend: str = "lexical"


class ContextOptimizer:
    def __init__(self, token_budget: int | None = None, max_candidates: int = 20,
                 max_passages: int = 5, generation_reserve: int | None = None,
                 evaluation_reserve: int | None = None, reranker_backend: str | None = None,
                 rerank_timeout_seconds: float | None = None, cross_encoder: Any | None = None) -> None:
        self.counter = TokenCounter()
        self.token_budget = token_budget or settings.request_token_budget
        self.max_candidates = max_candidates
        self.max_passages = max_passages
        self.generation_reserve = settings.generation_reserve if generation_reserve is None else generation_reserve
        self.evaluation_reserve = settings.evaluation_reserve if evaluation_reserve is None else evaluation_reserve
        self.reranker_backend = reranker_backend or settings.reranker_backend
        self.rerank_timeout_seconds = rerank_timeout_seconds or settings.rerank_timeout_seconds
        self.cross_encoder = cross_encoder
        self._rerank_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-reranker")
        self._rerank_lock = threading.Lock()
        if self.reranker_backend == "cross-encoder" and self.cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except ImportError:
                self.reranker_backend = "lexical"

    @staticmethod
    def _sentence_compress(query: str, text: str, max_chars: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        ranked = sorted(enumerate(sentences), key=lambda item: lexical_score(query, item[1]), reverse=True)
        selected: list[tuple[int, str]] = []
        used = 0
        for index, sentence in ranked:
            candidate = sentence.strip()
            if not candidate:
                continue
            if used + len(candidate) > max_chars:
                continue
            selected.append((index, candidate))
            used += len(candidate)
        selected.sort(key=lambda item: item[0])
        return " ".join(sentence for _, sentence in selected)

    def _rerank(self, query: str, chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, str]:
        start = time.perf_counter()
        candidates = chunks[: self.max_candidates]
        backend = "lexical"
        scores = None
        if self.cross_encoder and candidates and self._rerank_lock.acquire(blocking=False):
            pairs = [[query, c["text"]] for c in candidates]
            future = self._rerank_executor.submit(self.cross_encoder.predict, pairs)
            future.add_done_callback(lambda _: self._rerank_lock.release())
            try:
                scores = future.result(timeout=self.rerank_timeout_seconds)
                backend = "cross-encoder"
            except (FutureTimeoutError, Exception):
                backend = "lexical_timeout_fallback"
                future.cancel()
        if scores is not None:
            candidates = [{**c, "rerank_score": float(score)} for c, score in zip(candidates, scores)]
        else:
            candidates = [{**c, "rerank_score": lexical_score(query, c["text"])} for c in candidates]
        candidates.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
        elapsed = (time.perf_counter() - start) * 1000
        RERANK_LATENCY.observe(elapsed / 1000)
        return candidates, elapsed, backend

    def optimize(self, query: str, chunks: list[dict[str, Any]], system_prompt: str,
                 user_envelope: str = "", tool_results: str = "") -> ContextPackage:
        ranked, rerank_ms, reranker_backend = self._rerank(query, chunks)
        # Include prompt and message overhead, tool outputs and generation/evaluation reserves.
        fixed_text = f"{system_prompt}\n{user_envelope}\n{tool_results}"
        fixed_tokens = self.counter.count(fixed_text) + self.generation_reserve + self.evaluation_reserve + 64
        available = max(0, self.token_budget - fixed_tokens)
        selected: list[dict[str, Any]] = []
        blocks: list[str] = []
        used = 0
        original_chars = sum(len(c.get("text", "")) for c in ranked)
        included_chars = 0
        seen_text: set[str] = set()
        for chunk in ranked:
            if len(selected) >= self.max_passages or available <= used:
                break
            normalized_text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip().lower()
            if not normalized_text or normalized_text in seen_text:
                continue
            seen_text.add(normalized_text)
            source = str(chunk.get("source", chunk.get("document_id", "unknown")))
            prefix = f"[source_id={chunk['chunk_id']} | source={source}] "
            body = str(chunk["text"])
            full = prefix + body
            needed = self.counter.count(full)
            if used + needed <= available:
                selected.append(chunk)
                blocks.append(f"{prefix}{body}")
                used += needed
                included_chars += len(body)
                continue
            remaining = max(0, available - used)
            # Roughly 4 chars/token, reserving room for source metadata.
            compressed = self._sentence_compress(query, body, max(0, (remaining - self.counter.count(prefix)) * 4))
            if compressed:
                compact = prefix + compressed
                compact_tokens = self.counter.count(compact)
                if compact_tokens <= remaining:
                    selected.append({**chunk, "text": compressed, "compressed": True})
                    blocks.append(compact)
                    used += compact_tokens
                    included_chars += len(compressed)
        final_text = "\n\n".join(blocks)
        total = fixed_tokens + used
        if total > self.token_budget:
            raise ContextBudgetExceeded(f"Fixed prompt envelope requires {total} tokens; budget is {self.token_budget}")
        ratio = (included_chars / original_chars) if original_chars else 1.0
        return ContextPackage(final_text, selected, total, self.token_budget, rerank_ms, ratio, reranker_backend)
