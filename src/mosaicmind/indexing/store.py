"""Two collections in a single Chroma DB:

  - mosaic_text   : text + transcripts + captions, embedded with Gemini.
  - mosaic_images : image paths, embedded with CLIP (sentence-transformers).

This keeps a clean cross-modal index: text queries hit the text store
(including image captions, so they retrieve images too), while CLIP-based
image search lets us match on visual content directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from PIL import Image

from mosaicmind.config import get_settings
from mosaicmind.ingestion.base import IngestResult
from mosaicmind.llm.gemini import gemini_embed_model
from mosaicmind.utils.logging import logger

TEXT_COLLECTION = "mosaic_text"
IMAGE_COLLECTION = "mosaic_images"


@lru_cache(maxsize=1)
def _chroma_client() -> chromadb.api.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


@lru_cache(maxsize=1)
def get_text_store() -> ChromaVectorStore:
    client = _chroma_client()
    coll = client.get_or_create_collection(TEXT_COLLECTION)
    return ChromaVectorStore(chroma_collection=coll)


@lru_cache(maxsize=1)
def _image_collection():
    settings = get_settings()
    client = _chroma_client()
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.image_embed_model)
    return client.get_or_create_collection(IMAGE_COLLECTION, embedding_function=embed_fn)


def get_image_store():
    """The image store is queried directly via Chroma's API (CLIP embedder)."""
    return _image_collection()


@lru_cache(maxsize=1)
def _text_index() -> VectorStoreIndex:
    store = get_text_store()
    storage = StorageContext.from_defaults(vector_store=store)
    return VectorStoreIndex.from_vector_store(
        vector_store=store,
        embed_model=gemini_embed_model(),
        storage_context=storage,
    )


# ----------------------------- ingest -------------------------------------


def _to_documents(result: IngestResult) -> list[Document]:
    """Convert text-modal chunks (incl. captions, transcripts) into LlamaIndex docs."""
    docs: list[Document] = []
    for c in result.chunks:
        if c.modality in ("text", "audio", "video"):
            docs.append(
                Document(
                    text=c.content,
                    doc_id=c.chunk_id,
                    metadata={**c.metadata, "doc_id": c.doc_id, "chunk_id": c.chunk_id},
                )
            )
    return docs


def _add_images_to_clip(result: IngestResult) -> int:
    """Push image-modality chunks into the CLIP-backed Chroma collection."""
    coll = _image_collection()
    ids: list[str] = []
    embeds_input: list[Any] = []  # PIL Image objects
    metas: list[dict[str, Any]] = []
    for c in result.chunks:
        if c.modality != "image":
            continue
        try:
            img = Image.open(c.content).convert("RGB")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[index] skip image {c.content}: {e}")
            continue
        ids.append(c.chunk_id)
        embeds_input.append(img)
        metas.append({**c.metadata, "doc_id": c.doc_id, "chunk_id": c.chunk_id, "path": c.content})

    if not ids:
        return 0

    # SentenceTransformer's CLIP wrapper accepts PIL images via its embed function.
    coll.add(ids=ids, images=embeds_input, metadatas=metas)
    return len(ids)


def add_ingest_result(result: IngestResult) -> dict[str, int]:
    """Insert all chunks from one ingest result into the right collection."""
    settings = get_settings()
    settings.ensure_dirs()

    text_docs = _to_documents(result)
    splitter = SentenceSplitter(chunk_size=900, chunk_overlap=120)
    nodes = splitter.get_nodes_from_documents(text_docs) if text_docs else []

    if nodes:
        idx = _text_index()
        idx.insert_nodes(nodes)

    n_imgs = _add_images_to_clip(result)
    logger.info(f"[index] doc={result.doc_id} text_nodes={len(nodes)} image_nodes={n_imgs}")
    return {"text_nodes": len(nodes), "image_nodes": n_imgs}


# ----------------------------- retrieve -----------------------------------


@dataclass
class Hit:
    text: str
    score: float
    metadata: dict[str, Any]


def retrieve_text(query: str, top_k: int = 6) -> list[Hit]:
    idx = _text_index()
    retriever = VectorIndexRetriever(index=idx, similarity_top_k=top_k)
    nodes: list[NodeWithScore] = retriever.retrieve(query)
    return [
        Hit(text=n.get_content(), score=float(n.score or 0.0), metadata=dict(n.metadata or {}))
        for n in nodes
    ]


def retrieve_images(query: str, top_k: int = 4) -> list[Hit]:
    coll = _image_collection()
    res = coll.query(query_texts=[query], n_results=top_k, include=["metadatas", "distances"])
    out: list[Hit] = []
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    for i, meta in enumerate(metas):
        path = (meta or {}).get("path") or (meta or {}).get("source") or ids[i]
        out.append(
            Hit(
                text=f"[IMAGE {Path(path).name}] {(meta or {}).get('caption', '')}",
                score=1.0 - float(dists[i]) if i < len(dists) else 0.0,
                metadata={**(meta or {}), "path": path},
            )
        )
    return out
