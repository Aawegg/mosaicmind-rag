"""POST /eval — run an LLM-as-judge eval on a small Q/A set."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from mosaicmind.mlops.eval import EvalCase, run_eval

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRequest(BaseModel):
    cases: list[EvalCase]
    run_name: str | None = None


class EvalResponse(BaseModel):
    run_name: str
    n: int
    avg_score: float
    avg_latency_ms: float
    rows: list[dict]


@router.post("", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest) -> EvalResponse:
    out = run_eval(req.cases, run_name=req.run_name)
    return EvalResponse(**out)
