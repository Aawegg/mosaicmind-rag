"""Common types for the ingestion pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Modality = Literal["text", "image", "audio", "video"]


@dataclass
class IngestedChunk:
    """A unit of content to send to the index.

    `content` is text for text/audio/video chunks, or a path to an image file
    for image-modality chunks.  `metadata` holds provenance.
    """

    doc_id: str
    chunk_id: str
    modality: Modality
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    doc_id: str
    source_path: Path
    chunks: list[IngestedChunk]
    summary: str = ""

    def by_modality(self) -> dict[Modality, list[IngestedChunk]]:
        out: dict[Modality, list[IngestedChunk]] = {"text": [], "image": [], "audio": [], "video": []}
        for c in self.chunks:
            out[c.modality].append(c)
        return out
