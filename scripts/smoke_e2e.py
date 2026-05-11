"""End-to-end smoke test: build a tiny PDF, ingest, query, print the answer."""
from __future__ import annotations

from pathlib import Path

import fitz

from mosaicmind.agents.graph import answer
from mosaicmind.config import get_settings
from mosaicmind.indexing.store import add_ingest_result
from mosaicmind.ingestion.pdf import ingest_pdf
from mosaicmind.utils.logging import setup_logging


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "MosaicMind is a multimodal RAG assistant.", fontsize=12)
    p1.insert_text(
        (50, 100),
        "It uses LangGraph to orchestrate retrieval, LlamaIndex to index "
        "PDFs/images/audio/video, and Gemini 3.1 Pro for synthesis.",
        fontsize=10,
    )
    p2 = doc.new_page()
    p2.insert_text((50, 72), "The capital of France is Paris.", fontsize=12)
    p2.insert_text((50, 100), "The Eiffel Tower stands 330 meters tall.", fontsize=10)
    doc.save(path)
    doc.close()


def main() -> int:
    setup_logging()
    s = get_settings()
    pdf = s.upload_dir / "smoke.pdf"
    _make_pdf(pdf)
    print(f"created {pdf}")

    res = ingest_pdf(pdf)
    counts = add_ingest_result(res)
    print(f"ingested doc_id={res.doc_id} counts={counts}")

    for q in [
        "What is MosaicMind?",
        "How tall is the Eiffel Tower?",
    ]:
        print(f"\n=== Q: {q}")
        out = answer(q)
        print(f"--- A:\n{out.answer}")
        print(f"--- citations: {[(c.n, c.source_name, c.page) for c in out.citations]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
