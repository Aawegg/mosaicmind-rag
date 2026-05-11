"""LangChain tools that wrap the LlamaIndex retrievers."""
from __future__ import annotations

from langchain_core.tools import tool

from mosaicmind.indexing.store import retrieve_images, retrieve_text


def _format_hits(hits, kind: str) -> str:
    if not hits:
        return f"(no {kind} hits)"
    out = []
    for i, h in enumerate(hits, start=1):
        src = h.metadata.get("source_name") or h.metadata.get("source") or "?"
        loc = ""
        if "page" in h.metadata:
            loc = f" p.{h.metadata['page']}"
        elif "timestamp_s" in h.metadata:
            loc = f" @ {h.metadata['timestamp_s']}s"
        out.append(f"[{i}] ({src}{loc}, score={h.score:.3f})\n{h.text}")
    return "\n\n".join(out)


@tool("search_text", return_direct=False)
def search_text_tool(query: str, top_k: int = 6) -> str:
    """Search PDFs, transcripts, and image captions for text relevant to the query."""
    return _format_hits(retrieve_text(query, top_k=top_k), "text")


@tool("search_images", return_direct=False)
def search_images_tool(query: str, top_k: int = 4) -> str:
    """Search images and video keyframes by visual content (CLIP) for the query."""
    return _format_hits(retrieve_images(query, top_k=top_k), "image")


ALL_TOOLS = [search_text_tool, search_images_tool]
