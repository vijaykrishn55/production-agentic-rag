from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic feature-hash embeddings for tests and lightweight development."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the 'ml' extra to use SentenceTransformers") from exc
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when", "which", "who", "why", "with", "you", "your"}


def lexical_score(query: str, text: str) -> float:
    q = Counter(token for token in re.findall(r"[a-z0-9_]+", query.lower()) if token not in _STOPWORDS)
    d = Counter(token for token in re.findall(r"[a-z0-9_]+", text.lower()) if token not in _STOPWORDS)
    if not q or not d:
        return 0.0
    overlap = sum(min(count, d[token]) for token, count in q.items())
    return overlap / max(1, sum(q.values()))


def build_embedder(backend: str | None = None) -> Embedder:
    import os
    selected = (backend or os.getenv("EMBEDDING_BACKEND", "hash")).lower()
    if selected in {"sentence-transformer", "sentence_transformer", "minilm"}:
        return SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return HashEmbedder()
