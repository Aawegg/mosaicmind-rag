"""Audio ingestion: transcribe via Groq Whisper, then chunk."""
from __future__ import annotations

from pathlib import Path

from mosaicmind.config import get_settings
from mosaicmind.ingestion.base import IngestedChunk, IngestResult
from mosaicmind.llm.groq import groq_transcribe
from mosaicmind.utils.ids import file_sha1, new_doc_id
from mosaicmind.utils.logging import logger


def _split(text: str, target: int = 1500, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + target])
        i += target - overlap
    return out


def _transcribe(audio_path: Path) -> str:
    settings = get_settings()
    if settings.asr_provider.lower() == "gemini":
        return _transcribe_gemini(audio_path)
    return groq_transcribe(audio_path, model=settings.asr_model)


def _transcribe_gemini(audio_path: Path) -> str:
    """Fallback path: native Gemini audio understanding via File API."""
    from mosaicmind.llm.gemini import gemini_native_client

    client = gemini_native_client()
    uploaded = client.files.upload(file=str(audio_path))
    resp = client.models.generate_content(
        model="gemini-2.5-flash-native-audio-latest",
        contents=[uploaded, "Transcribe this audio verbatim. No commentary."],
    )
    return getattr(resp, "text", "")


def ingest_audio(path: Path) -> IngestResult:
    path = Path(path)
    doc_id = new_doc_id()
    sha = file_sha1(path)
    logger.info(f"[audio] transcribing {path.name} doc_id={doc_id}")

    transcript = _transcribe(path).strip()
    chunks: list[IngestedChunk] = []
    for j, piece in enumerate(_split(transcript)):
        chunks.append(
            IngestedChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-a{j}",
                modality="audio",
                content=piece,
                metadata={
                    "source": str(path),
                    "source_name": path.name,
                    "modality": "audio",
                    "chunk_index": j,
                    "sha1": sha,
                },
            )
        )
    logger.info(f"[audio] {path.name}: {len(chunks)} transcript chunks")
    return IngestResult(doc_id=doc_id, source_path=path, chunks=chunks, summary=transcript[:500])
