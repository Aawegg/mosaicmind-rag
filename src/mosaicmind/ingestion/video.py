"""Video ingestion: extract audio + keyframes, transcribe + caption."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2

from mosaicmind.config import get_settings
from mosaicmind.ingestion.audio import _split, _transcribe
from mosaicmind.ingestion.base import IngestedChunk, IngestResult
from mosaicmind.ingestion.image import _caption_with_gemini
from mosaicmind.utils.ids import file_sha1, new_doc_id
from mosaicmind.utils.logging import logger


def _extract_audio(video_path: Path, out_path: Path) -> Path | None:
    """Use ffmpeg to extract a 16kHz mono wav from the video."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path if out_path.exists() and out_path.stat().st_size > 1024 else None
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"[video] audio extraction failed: {e}")
        return None


def _extract_keyframes(video_path: Path, out_dir: Path, every_n_seconds: int = 5) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"[video] cannot open {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(max(1, fps * every_n_seconds))
    saved: list[Path] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ts = int(idx / fps)
            p = out_dir / f"frame_{ts:06d}s.jpg"
            cv2.imwrite(str(p), frame)
            saved.append(p)
        idx += 1
    cap.release()
    logger.info(f"[video] extracted {len(saved)} keyframes from {video_path.name}")
    return saved


def ingest_video(path: Path, keyframe_every_s: int = 5, max_frames_to_caption: int = 8) -> IngestResult:
    settings = get_settings()
    path = Path(path)
    doc_id = new_doc_id()
    sha = file_sha1(path)
    logger.info(f"[video] ingesting {path.name} doc_id={doc_id}")

    workspace = settings.upload_dir / "video" / doc_id
    audio_path = _extract_audio(path, workspace / "audio.wav")
    keyframes = _extract_keyframes(path, workspace / "frames", every_n_seconds=keyframe_every_s)

    chunks: list[IngestedChunk] = []

    if audio_path is not None:
        transcript = _transcribe(audio_path).strip()
        for j, piece in enumerate(_split(transcript)):
            chunks.append(
                IngestedChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-v-a{j}",
                    modality="video",
                    content=piece,
                    metadata={
                        "source": str(path),
                        "source_name": path.name,
                        "modality": "video-transcript",
                        "chunk_index": j,
                        "sha1": sha,
                    },
                )
            )

    # Caption a sampled subset of keyframes (cost control).
    if keyframes:
        stride = max(1, len(keyframes) // max_frames_to_caption)
        sampled = keyframes[::stride][:max_frames_to_caption]
        for k, kf in enumerate(sampled):
            ts_seconds = int(kf.stem.split("_")[-1].rstrip("s"))
            cap_text = _caption_with_gemini(kf)
            chunks.append(
                IngestedChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-v-img{k}",
                    modality="image",
                    content=str(kf),
                    metadata={
                        "source": str(path),
                        "source_name": path.name,
                        "modality": "video-frame",
                        "timestamp_s": ts_seconds,
                        "caption": cap_text,
                        "sha1": sha,
                    },
                )
            )
            chunks.append(
                IngestedChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-v-cap{k}",
                    modality="text",
                    content=f"[{path.name} @ {ts_seconds}s]\n{cap_text}",
                    metadata={
                        "source": str(path),
                        "source_name": path.name,
                        "modality": "video-frame-caption",
                        "timestamp_s": ts_seconds,
                        "sha1": sha,
                    },
                )
            )

    logger.info(f"[video] {path.name}: {len(chunks)} chunks across modalities")
    return IngestResult(doc_id=doc_id, source_path=path, chunks=chunks)
