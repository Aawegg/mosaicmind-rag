"""POST /query — run the LangGraph agent."""
from __future__ import annotations

import time

from fastapi import APIRouter

from mosaicmind.agents.graph import answer as agent_answer
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
