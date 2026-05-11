"""Image ingestion: stores the image and a Gemini-generated caption.

We index two things per image:
  - the image path (for the CLIP-based image vector store)
  - a textual caption (for the text vector store, enabling cross-modal search)
"""
from __future__ import annotations

from pathlib import Path

from mosaicmind.config import get_settings
from mosaicmind.ingestion.base import IngestedChunk, IngestResult
from mosaicmind.llm.gemini import gemini_native_client
from mosaicmind.utils.ids import file_sha1, new_doc_id
from mosaicmind.utils.logging import logger

CAPTION_PROMPT = (
    "Describe this image in 3-6 sentences for retrieval. "
    "Mention any visible text verbatim, key objects, scene, layout, "
    "and any chart/table contents. Be concrete; no preamble."
)


def _caption_with_gemini(image_path: Path) -> str:
    settings = get_settings()
    client = gemini_native_client()
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    try:
        from google.genai import types

        resp = client.models.generate_content(
            model=settings.mm_model,
            contents=[
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime),
                CAPTION_PROMPT,
            ],
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[image] caption failed for {image_path.name}: {e}")
        return f"[uncaptioned image: {image_path.name}]"


def ingest_image(path: Path, caption: str | None = None) -> IngestResult:
    path = Path(path)
    doc_id = new_doc_id()
    sha = file_sha1(path)
    logger.info(f"[image] ingesting {path.name} doc_id={doc_id}")

    cap = caption or _caption_with_gemini(path)

    chunks = [
        IngestedChunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-img",
            modality="image",
            content=str(path),
            metadata={
                "source": str(path),
                "source_name": path.name,
                "modality": "image",
                "caption": cap,
                "sha1": sha,
            },
        ),
        IngestedChunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-cap",
            modality="text",
            content=f"[Image: {path.name}]\n{cap}",
            metadata={
                "source": str(path),
                "source_name": path.name,
                "modality": "image-caption",
                "sha1": sha,
            },
        ),
    ]
    return IngestResult(doc_id=doc_id, source_path=path, chunks=chunks, summary=cap)
