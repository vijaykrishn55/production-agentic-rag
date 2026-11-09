from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any

from .security import SecurityInterceptor


class DocumentIngestor:
    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 160) -> None:
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.security = SecurityInterceptor()

    def parse_file(self, path: str | Path) -> str:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md", ".csv", ".log"}:
            return file_path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".json":
            return json.dumps(json.loads(file_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Install the 'documents' extra for PDF ingestion") from exc
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(file_path)).pages)
        if suffix == ".docx":
            try:
                from docx import Document
            except ImportError as exc:
                raise RuntimeError("Install the 'documents' extra for DOCX ingestion") from exc
            return "\n".join(p.text for p in Document(str(file_path)).paragraphs)
        if suffix == ".pptx":
            try:
                from pptx import Presentation
            except ImportError as exc:
                raise RuntimeError("Install the 'documents' extra for PPTX ingestion") from exc
            return "\n".join(shape.text for slide in Presentation(str(file_path)).slides for shape in slide.shapes if hasattr(shape, "text"))
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            try:
                import pytesseract
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Install the 'ocr' extra and Tesseract for image OCR") from exc
            return pytesseract.image_to_string(Image.open(file_path))
        raise ValueError(f"Unsupported document format: {suffix}")

    def chunk_text(self, text: str) -> list[str]:
        clean = re.sub(r"\r\n?", "\n", text).strip()
        if not clean:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + self.chunk_chars)
            if end < len(clean):
                boundary = max(clean.rfind("\n", start, end), clean.rfind(". ", start, end))
                if boundary > start + self.chunk_chars // 2:
                    end = boundary + 1
            chunks.append(clean[start:end].strip())
            if end >= len(clean):
                break
            start = max(start + 1, end - self.overlap_chars)
        return [chunk for chunk in chunks if chunk]

    def ingest(self, document_id: str, text: str, source: str, tenant_id: str,
               metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        sanitized, redactions = self.security.sanitize(text)
        chunks = self.chunk_text(sanitized)
        return [{
            "chunk_id": f"{document_id}:{i}:{hashlib.sha256(f'{tenant_id}|{document_id}|{i}|{chunk}'.encode()).hexdigest()[:12]}",
            "document_id": document_id,
            "tenant_id": tenant_id,
            "source": source,
            "text": chunk,
            "metadata": {**(metadata or {}), "pii_redactions": sorted(set(redactions)), "chunk_index": i},
        } for i, chunk in enumerate(chunks)]
