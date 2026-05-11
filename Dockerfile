# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache \
    PORT=8080

# ffmpeg = audio/video ingestion. libgl/glib = opencv-headless transitive deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e .

# Pre-download the CLIP model so cold starts don't pay for it.
# (Cloud Run cold start is dominated by image pull + first model load.)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/healthz" || exit 1

# Cloud Run injects $PORT (defaults to 8080).  uvicorn must bind to 0.0.0.0:$PORT.
CMD ["sh", "-c", "uvicorn mosaicmind.api.main:app --host 0.0.0.0 --port ${PORT}"]
