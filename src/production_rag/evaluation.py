from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .embeddings import lexical_score


def proxy_metrics(query: str, answer: str, contexts: list[str], reference: str | None = None) -> dict[str, float]:
    """Deterministic lexical regression metrics; not a substitute for RAGAS."""
    context = " ".join(contexts)
    faithfulness = lexical_score(answer, context) if context else 0.0
    relevance = lexical_score(query, answer)
    precision = lexical_score(answer, context) if answer and context else 0.0
    recall = lexical_score(reference, context) if reference and context else 0.0
    return {"faithfulness": faithfulness, "answer_relevance": relevance,
            "context_precision": precision, "context_recall": recall}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_proxy_evaluation(rows: list[dict[str, Any]], threshold: float = 0.85) -> dict[str, Any]:
    results = []
    for row in rows:
        metrics = proxy_metrics(row["question"], row.get("answer", ""), row.get("contexts", []), row.get("reference"))
        results.append({"id": row.get("id"), **metrics})
    aggregate = {key: sum(item[key] for item in results) / max(1, len(results))
                 for key in ("faithfulness", "answer_relevance", "context_precision", "context_recall")}
    return {"backend": "proxy_lexical_not_ragas", "threshold": threshold,
            "passed": aggregate["faithfulness"] >= threshold, "aggregate": aggregate,
            "samples": results, "sample_count": len(results)}


def run_ragas_evaluation(rows: list[dict[str, Any]], threshold: float = 0.85,
                         llm: Any = None, embeddings: Any = None) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError("Install the evaluation extra (RAGAS 0.1.7) to run RAGAS") from exc
    if llm is None or embeddings is None:
        try:
            import os
            from langchain_openai import ChatOpenAI
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from .config import settings
            llm = llm or ChatOpenAI(
                model=settings.llm_model,
                openai_api_key=settings.llm_api_key,
                openai_api_base=settings.llm_base_url,
                temperature=0,
            )
            embeddings = embeddings or HuggingFaceEmbeddings(
                model_name=os.getenv("RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                model_kwargs={"device": os.getenv("RAGAS_EMBEDDING_DEVICE", "cpu")},
            )
        except ImportError as exc:
            raise RuntimeError("Install evaluation, llm and ml extras for configured live RAGAS models") from exc
    dataset = Dataset.from_list([{
        "question": row["question"],
        "answer": row.get("answer", ""),
        "contexts": row.get("contexts", []),
        "ground_truth": row.get("reference", ""),
    } for row in rows])
    kwargs: dict[str, Any] = {}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], **kwargs)
    raw_scores = result.to_pandas().to_dict(orient="records")
    scores = []
    for row in raw_scores:
        cleaned = {}
        for key, value in row.items():
            try:
                numeric = float(value)
                cleaned[key] = numeric if numeric == numeric else 0.0
            except (TypeError, ValueError):
                cleaned[key] = str(value)
        scores.append(cleaned)
    keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    aggregate = {key: sum(float(row.get(key, 0) or 0) for row in scores) / max(1, len(scores)) for key in keys}
    return {"backend": "ragas", "ragas_version": "0.1.7-compatible-api", "threshold": threshold,
            "passed": aggregate["faithfulness"] >= threshold, "aggregate": aggregate, "samples": scores,
            "sample_count": len(scores)}


def validate_citations(citations: list[dict[str, Any] | str], chunks: list[dict[str, Any]],
                       tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    allowed = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    for tool in tools or []:
        source_id = str(tool.get("source_id", f"tool:{tool.get('tool', 'unknown')}"))
        allowed[source_id] = {"chunk_id": source_id, "document_id": tool.get("document_id", tool.get("tool", "tool")),
                              "text": tool.get("content", tool.get("title", "Tool result"))}
    valid = []
    for citation in citations:
        chunk_id = str(citation if isinstance(citation, str) else citation.get("chunk_id", ""))
        if chunk_id in allowed:
            chunk = allowed[chunk_id]
            valid.append({"source_id": chunk_id, "document_id": str(chunk["document_id"]),
                          "chunk_id": chunk_id, "quote": str(chunk["text"])[:240]})
    return valid


def local_faithfulness(answer: str, contexts: list[str]) -> float:
    """Conservative lexical support proxy used only in offline/no-judge mode."""
    if not answer or not contexts:
        return 0.0
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))
    context_tokens = set(re.findall(r"[a-z0-9]+", " ".join(contexts).lower()))
    meaningful = {token for token in answer_tokens if len(token) > 3}
    return len(meaningful & context_tokens) / max(1, len(meaningful))
