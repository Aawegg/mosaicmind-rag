"""POST /ingest — multipart upload, routed to the right modality pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from mosaicmind.api.schemas import IngestResponse
from mosaicmind.config import get_settings
from mosaicmind.indexing.store import add_ingest_result
from mosaicmind.ingestion.router import detect_kind, ingest_path
from mosaicmind.mlops.tracking import log_ingest
from mosaicmind.utils.logging import logger

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    out = dest_dir / file.filename
    with out.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return out


@router.post("", response_model=IngestResponse)
def ingest_file(
    background: BackgroundTasks,
    file: Annotated[UploadFile, File()],
) -> IngestResponse:
    settings = get_settings()
    saved = _save_upload(file, settings.upload_dir)
    try:
        kind = detect_kind(saved)
    except ValueError as e:
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail=str(e)) from e

    logger.info(f"[api/ingest] {file.filename} -> {kind}")

    result = ingest_path(saved)
    counts = add_ingest_result(result)

    background.add_task(
        log_ingest,
        doc_id=result.doc_id,
        source_name=saved.name,
        kind=kind,
        text_nodes=counts["text_nodes"],
        image_nodes=counts["image_nodes"],
    )

    return IngestResponse(
        doc_id=result.doc_id,
        source_name=saved.name,
        kind=kind,
        text_nodes=counts["text_nodes"],
        image_nodes=counts["image_nodes"],
        summary=result.summary[:500],
    )
