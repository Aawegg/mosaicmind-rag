"""PDF ingestion: text via PyMuPDF + embedded images."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from mosaicmind.config import get_settings
from mosaicmind.ingestion.base import IngestedChunk, IngestResult
from mosaicmind.utils.ids import file_sha1, new_doc_id
from mosaicmind.utils.logging import logger


def _split(text: str, target: int = 1200, overlap: int = 150) -> list[str]:
    """Simple character-window splitter; LlamaIndex's parser handles further splitting."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + target])
        i += target - overlap
    return out


def ingest_pdf(path: Path) -> IngestResult:
    settings = get_settings()
    path = Path(path)
    doc_id = new_doc_id()
    sha = file_sha1(path)
    logger.info(f"[pdf] ingesting {path.name} doc_id={doc_id} sha1={sha[:8]}")

    chunks: list[IngestedChunk] = []
    images_dir = settings.upload_dir / "images" / doc_id
    images_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(path) as pdf:
        for page_idx, page in enumerate(pdf):
            page_text = page.get_text("text") or ""
            for j, piece in enumerate(_split(page_text)):
                chunks.append(
                    IngestedChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}-p{page_idx}-t{j}",
                        modality="text",
                        content=piece,
                        metadata={
                            "source": str(path),
                            "source_name": path.name,
                            "page": page_idx + 1,
                            "modality": "text",
                            "sha1": sha,
                        },
                    )
                )

            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(pdf, xref)
                if pix.n > 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_path = images_dir / f"p{page_idx + 1}-i{img_idx}.png"
                pix.save(img_path)
                chunks.append(
                    IngestedChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}-p{page_idx}-img{img_idx}",
                        modality="image",
                        content=str(img_path),
                        metadata={
                            "source": str(path),
                            "source_name": path.name,
                            "page": page_idx + 1,
                            "modality": "image",
                            "sha1": sha,
                        },
                    )
                )

    logger.info(f"[pdf] {path.name}: {len(chunks)} chunks")
    return IngestResult(doc_id=doc_id, source_path=path, chunks=chunks)
