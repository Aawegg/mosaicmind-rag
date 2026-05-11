"""Vector indexing layer (LlamaIndex + Chroma + CLIP for images)."""

from mosaicmind.indexing.store import (
    add_ingest_result,
    get_image_store,
    get_text_store,
    retrieve_images,
    retrieve_text,
)

__all__ = [
    "add_ingest_result",
    "get_image_store",
    "get_text_store",
    "retrieve_images",
    "retrieve_text",
]
