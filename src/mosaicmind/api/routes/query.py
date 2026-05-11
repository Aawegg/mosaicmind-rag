"""POST /query and POST /query/stream — synchronous + streaming."""
from __future__ import annotations

import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from mosaicmind.agents.graph import answer as agent_answer
from mosaicmind.agents.graph import stream_answer
from mosaicmind.api.schemas import CitationOut, QueryRequest, QueryResponse
from mosaicmind.mlops.tracking import log_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    result = agent_answer(req.question)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    cits = [CitationOut(**c.__dict__) for c in result.citations]
    log_query(
        question=req.question,
        answer=result.answer,
        plan=result.plan,
        n_citations=len(cits),
        latency_ms=latency_ms,
    )

    return QueryResponse(
        question=req.question,
        answer=result.answer,
        citations=cits,
        plan=result.plan,
        trace=result.trace,
        latency_ms=latency_ms,
    )


def _sse(event: str, data) -> bytes:
    """Serialize one SSE frame."""
    payload = data if isinstance(data, str) else json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode()


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE endpoint that streams plan -> citations -> tokens -> done."""
    history = [m.model_dump() for m in req.history]

    async def events():
        t0 = time.perf_counter()
        full: list[str] = []
        n_citations = 0
        plan: dict = {}
        try:
            async for ev in stream_answer(req.question, history=history):
                if ev["event"] == "token" and isinstance(ev["data"], str):
                    full.append(ev["data"])
                if ev["event"] == "citations":
                    n_citations = len(ev["data"])
                if ev["event"] == "plan":
                    plan = ev["data"]
                yield _sse(ev["event"], ev["data"])
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            try:
                log_query(
                    question=req.question,
                    answer="".join(full),
                    plan=plan,
                    n_citations=n_citations,
                    latency_ms=latency_ms,
                )
            except Exception:  # noqa: BLE001
                pass
            yield _sse("latency", {"ms": latency_ms})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering on HF / nginx
        },
    )
