"""Multimodal ingestion: PDFs, images, audio, video -> normalized Documents."""

from mosaicmind.ingestion.audio import ingest_audio
from mosaicmind.ingestion.image import ingest_image
from mosaicmind.ingestion.pdf import ingest_pdf
from mosaicmind.ingestion.router import ingest_path
from mosaicmind.ingestion.video import ingest_video

__all__ = [
    "ingest_audio",
    "ingest_image",
    "ingest_path",
    "ingest_pdf",
    "ingest_video",
]
