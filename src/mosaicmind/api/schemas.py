"""Request/response schemas for the FastAPI surface."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    source_name: str
    kind: str
    text_nodes: int
    image_nodes: int
    summary: str = ""


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 6
    use_images: bool | None = None


class CitationOut(BaseModel):
    n: int
    source_name: str
    page: int | None = None
    timestamp_s: int | None = None
    score: float = 0.0
    modality: str = "text"


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut] = []
    plan: dict = {}
    trace: list[dict] = []
    latency_ms: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    heavy_model: str
    fast_model: str
    asr_provider: str
    text_embed_model: str
