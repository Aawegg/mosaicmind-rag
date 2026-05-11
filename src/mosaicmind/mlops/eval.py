"""LLM-as-judge eval harness. Reuses the LangGraph agent end-to-end."""
from __future__ import annotations

import json
import time
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from mosaicmind.agents.graph import _msg_text
from mosaicmind.agents.graph import answer as agent_answer
from mosaicmind.config import get_settings
from mosaicmind.llm.gemini import gemini_chat_model
from mosaicmind.mlops.tracking import log_eval_run
from mosaicmind.utils.logging import logger

JUDGE_SYSTEM = """\
You are a strict but fair grader for an enterprise RAG assistant.

Score the candidate answer 0..10 against the reference answer using:
  - faithfulness to the retrieved context
  - factual alignment with the reference
  - completeness
  - citation discipline (each non-trivial claim cites a source)

Return strict JSON: {"score": float, "rationale": str}
"""


class EvalCase(BaseModel):
    question: str
    reference: str = ""
    notes: str = ""


class EvalRow(BaseModel):
    question: str
    reference: str
    candidate: str
    score: float
    rationale: str
    latency_ms: int
    n_citations: int


def _judge(question: str, candidate: str, reference: str) -> tuple[float, str]:
    settings = get_settings()
    llm = gemini_chat_model(model=settings.heavy_model, temperature=0.0)
    msg = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Reference answer:\n{reference or '(none provided)'}\n\n"
            f"Candidate answer:\n{candidate}"
        )),
    ])
    raw = _msg_text(msg).strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    try:
        d = json.loads(raw)
        return float(d.get("score", 0.0)), str(d.get("rationale", ""))
    except Exception:  # noqa: BLE001
        return 0.0, f"unparseable judge output: {raw[:200]}"


def run_eval(cases: list[EvalCase], run_name: str | None = None) -> dict:
    name = run_name or f"eval-{uuid.uuid4().hex[:8]}"
    logger.info(f"[eval] start {name} n={len(cases)}")
    rows: list[EvalRow] = []
    for case in cases:
        t0 = time.perf_counter()
        ar = agent_answer(case.question)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        score, rationale = _judge(case.question, ar.answer, case.reference)
        rows.append(EvalRow(
            question=case.question,
            reference=case.reference,
            candidate=ar.answer,
            score=score,
            rationale=rationale,
            latency_ms=latency_ms,
            n_citations=len(ar.citations),
        ))

    avg_score = sum(r.score for r in rows) / max(1, len(rows))
    avg_latency = sum(r.latency_ms for r in rows) / max(1, len(rows))
    metrics = {"avg_score": avg_score, "avg_latency_ms": avg_latency, "n": len(rows)}
    log_eval_run(name, metrics, [r.model_dump() for r in rows])
    logger.info(f"[eval] {name}: avg_score={avg_score:.2f} avg_latency_ms={avg_latency:.0f}")
    return {
        "run_name": name,
        "n": len(rows),
        "avg_score": avg_score,
        "avg_latency_ms": avg_latency,
        "rows": [r.model_dump() for r in rows],
    }
