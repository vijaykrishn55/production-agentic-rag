FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app

# Lightweight service image used by local Docker Compose.
FROM runtime AS app
ARG INSTALL_FEATURES=lite
RUN python -m pip install --upgrade pip && \
    if [ "$INSTALL_FEATURES" = "full" ]; then \
      pip install ".[postgres,redis,llm,ml,evaluation,documents,ocr,tokenizer,observability]"; \
    else \
      pip install ".[postgres,redis,llm,documents,tokenizer,observability]"; \
    fi
USER appuser
EXPOSE 8000 8001 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1
CMD ["uvicorn", "production_rag.services.gateway:app", "--host", "0.0.0.0", "--port", "8000"]

# Production API image includes neural embedding and cross-encoder packages.
FROM runtime AS production
RUN python -m pip install --upgrade pip && \
    pip install ".[postgres,redis,llm,ml,documents,ocr,tokenizer,observability]"
USER appuser
EXPOSE 8000 8001 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1
CMD ["uvicorn", "production_rag.services.gateway:app", "--host", "0.0.0.0", "--port", "8000"]

# Evaluation worker has the additional RAGAS and dataset dependencies.
FROM runtime AS evaluation
RUN python -m pip install --upgrade pip && \
    pip install ".[postgres,redis,llm,ml,evaluation,documents,ocr,tokenizer,observability]"
USER appuser
CMD ["python", "-m", "production_rag.services.evaluation_worker"]
