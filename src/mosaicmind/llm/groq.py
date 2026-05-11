"""Groq clients: chat (Llama) + ASR (Whisper)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from groq import Groq
from langchain_groq import ChatGroq

from mosaicmind.config import get_settings


@lru_cache(maxsize=1)
def _groq_raw() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)


@lru_cache(maxsize=4)
def groq_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatGroq:
    s = get_settings()
    return ChatGroq(
        model=model or "llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=s.groq_api_key,
        timeout=60,
        max_retries=2,
    )


def groq_transcribe(audio_path: Path, model: str | None = None, language: str | None = None) -> str:
    """Transcribe an audio file with Groq's hosted Whisper."""
    s = get_settings()
    client = _groq_raw()
    with audio_path.open("rb") as fh:
        result = client.audio.transcriptions.create(
            file=(audio_path.name, fh.read()),
            model=model or s.asr_model,
            language=language,
            response_format="json",
        )
    return getattr(result, "text", str(result))
