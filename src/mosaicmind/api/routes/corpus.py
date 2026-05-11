"""DELETE /corpus — wipe the index (and optionally uploaded files)."""
from __future__ import annotations

from fastapi import APIRouter, Query

from mosaicmind.api.schemas import ClearCorpusResponse
from mosaicmind.indexing.store import clear_all

router = APIRouter(tags=["corpus"])


@router.delete("/corpus", response_model=ClearCorpusResponse)
def clear_corpus(
    delete_files: bool = Query(default=True, description="Also wipe data/uploads/"),
) -> ClearCorpusResponse:
    counts = clear_all(also_delete_uploads=delete_files)
    return ClearCorpusResponse(
        text_deleted=counts["text"],
        image_deleted=counts["image"],
        files_deleted=counts["files"],
    )
