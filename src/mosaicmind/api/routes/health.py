"""Health and readiness."""
from __future__ import annotations

from fastapi import APIRouter

from mosaicmind import __version__
from mosaicmind.api.schemas import HealthResponse
from mosaicmind.config import get_settings

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        version=__version__,
        heavy_model=s.heavy_model,
        fast_model=s.fast_model,
        asr_provider=s.asr_provider,
        text_embed_model=s.text_embed_model,
    )
