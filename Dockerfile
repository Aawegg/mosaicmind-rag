# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache \
    PORT=7860

# ffmpeg = audio/video ingestion. libgl/glib = opencv-headless transitive deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user (UID 1000).
# This UID/GID is also fine for Cloud Run / generic hosts.
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN chown -R appuser:appuser /app
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src ./src

RUN pip install --user --upgrade pip && pip install --user -e .

# Pre-download CLIP weights so the first request doesn't pay the download cost.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"

# Make sure the app's data dirs exist & are writable by appuser.
RUN mkdir -p /app/data/uploads /app/data/index /app/data/chroma /app/data/mlruns

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/healthz" || exit 1

# $PORT is honored by both Cloud Run (8080) and HF Spaces (7860).
CMD ["sh", "-c", "uvicorn mosaicmind.api.main:app --host 0.0.0.0 --port ${PORT}"]
