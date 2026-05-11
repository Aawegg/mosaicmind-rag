"""Route an arbitrary uploaded file to the right ingestion pipeline."""
from __future__ import annotations

from pathlib import Path

from mosaicmind.ingestion.audio import ingest_audio
from mosaicmind.ingestion.base import IngestResult
from mosaicmind.ingestion.image import ingest_image
from mosaicmind.ingestion.pdf import ingest_pdf
from mosaicmind.ingestion.video import ingest_video

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in PDF_EXTS:
        return "pdf"
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in AUDIO_EXTS:
        return "audio"
    if suffix in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Unsupported file type: {suffix} ({path.name})")


def ingest_path(path: Path) -> IngestResult:
    path = Path(path)
    kind = detect_kind(path)
    return {
        "pdf": ingest_pdf,
        "image": ingest_image,
        "audio": ingest_audio,
        "video": ingest_video,
    }[kind](path)
