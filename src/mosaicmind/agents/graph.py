"""LangGraph state machine: plan -> retrieve -> synthesize -> reflect (loop) -> end."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from mosaicmind.agents.prompts import PLANNER_SYSTEM, REFLECT_SYSTEM, SYNTHESIZER_SYSTEM
from mosaicmind.config import get_settings
from mosaicmind.indexing.store import Hit, retrieve_images, retrieve_text
from mosaicmind.llm.gemini import gemini_chat_model
from mosaicmind.utils.logging import logger

MAX_REFLECT_LOOPS = 1


@dataclass
class Citation:
    n: int
    source_name: str
    page: int | None = None
    timestamp_s: int | None = None
    score: float = 0.0
    modality: str = "text"


class GraphState(TypedDict, total=False):
    question: str
    plan: dict
    text_hits: list[Hit]
    image_hits: list[Hit]
    answer: str
    citations: list[Citation]
    loops: int
    trace: list[dict]


# --------------------------- nodes ----------------------------------


def _msg_text(msg) -> str:
    """Extract plain text from a LangChain message.

    Gemini 3.x via langchain-google-genai returns `content` as a list of
    typed parts (text, tool_use, thinking_signature, ...).  We concatenate
    the text parts and ignore everything else.
    """
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out: list[str] = []
        for item in c:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    out.append(str(item["text"]))
        return "".join(out)
    return str(c)


def _safe_json_load(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def plan_node(state: GraphState) -> GraphState:
    settings = get_settings()
    llm = gemini_chat_model(model=settings.fast_model, temperature=0.0)
    msg = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"User question: {state['question']}"),
    ])
    plan = _safe_json_load(_msg_text(msg))
    if not plan.get("queries"):
        plan = {"queries": [state["question"]], "use_images": False, "reasoning": "fallback"}
    state["plan"] = plan
    state.setdefault("trace", []).append({"node": "plan", "plan": plan})
    return state


def retrieve_node(state: GraphState) -> GraphState:
    plan = state.get("plan", {})
    queries: list[str] = plan.get("queries", [state["question"]])
    use_images: bool = bool(plan.get("use_images", False))

    text_seen: dict[str, Hit] = {}
    image_seen: dict[str, Hit] = {}

    for q in queries:
        for h in retrieve_text(q, top_k=6):
            key = h.metadata.get("chunk_id") or h.text[:80]
            prev = text_seen.get(key)
            if (prev is None) or (h.score > prev.score):
                text_seen[key] = h
        if use_images:
            for h in retrieve_images(q, top_k=4):
                key = h.metadata.get("chunk_id") or h.metadata.get("path") or h.text[:80]
                prev = image_seen.get(key)
                if (prev is None) or (h.score > prev.score):
                    image_seen[key] = h

    state["text_hits"] = sorted(text_seen.values(), key=lambda h: h.score, reverse=True)[:8]
    state["image_hits"] = sorted(image_seen.values(), key=lambda h: h.score, reverse=True)[:4]
    state.setdefault("trace", []).append({
        "node": "retrieve",
        "n_text": len(state["text_hits"]),
        "n_image": len(state["image_hits"]),
    })
    return state


def _build_context(state: GraphState) -> tuple[str, list[Citation]]:
    parts: list[str] = []
    cits: list[Citation] = []
    n = 0
    for h in state.get("text_hits", []):
        n += 1
        src = h.metadata.get("source_name") or "?"
        loc = ""
        if "page" in h.metadata:
            loc = f" p.{h.metadata['page']}"
        elif "timestamp_s" in h.metadata:
            loc = f" @ {h.metadata['timestamp_s']}s"
        parts.append(f"[{n}] {src}{loc}\n{h.text}")
        cits.append(Citation(
            n=n, source_name=src,
            page=h.metadata.get("page"),
            timestamp_s=h.metadata.get("timestamp_s"),
            score=h.score,
            modality=h.metadata.get("modality", "text"),
        ))
    for h in state.get("image_hits", []):
        n += 1
        src = h.metadata.get("source_name") or "?"
        cap = h.metadata.get("caption", "")
        parts.append(f"[{n}] IMAGE {src}\nCaption: {cap}")
        cits.append(Citation(n=n, source_name=src, score=h.score, modality="image"))
    return "\n\n".join(parts) or "(no context retrieved)", cits


def synthesize_node(state: GraphState) -> GraphState:
    settings = get_settings()
    context, cits = _build_context(state)
    state["citations"] = cits
    llm = gemini_chat_model(model=settings.heavy_model, temperature=0.2)
    msg = llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=f"Question: {state['question']}\n\nContext:\n{context}"),
    ])
    state["answer"] = _msg_text(msg).strip()
    state.setdefault("trace", []).append({"node": "synthesize", "answer_chars": len(state["answer"])})
    return state


def reflect_node(state: GraphState) -> GraphState:
    settings = get_settings()
    state["loops"] = state.get("loops", 0) + 1
    context, _ = _build_context(state)
    llm = gemini_chat_model(model=settings.fast_model, temperature=0.0)
    msg = llm.invoke([
        SystemMessage(content=REFLECT_SYSTEM),
        HumanMessage(content=f"Question: {state['question']}\n\nDraft:\n{state['answer']}\n\nContext:\n{context}"),
    ])
    parsed = _safe_json_load(_msg_text(msg))
    state.setdefault("trace", []).append({"node": "reflect", "result": parsed})

    if (
        not parsed.get("sufficient", True)
        and parsed.get("followup_query")
        and state["loops"] <= MAX_REFLECT_LOOPS
    ):
        state["plan"] = {
            "queries": [parsed["followup_query"]],
            "use_images": state.get("plan", {}).get("use_images", False),
            "reasoning": "reflection followup",
        }
        state["_branch"] = "retry"
    else:
        state["_branch"] = "done"
    return state


def _branch_router(state: GraphState) -> str:
    return state.get("_branch", "done")


# --------------------------- graph -----------------------------------


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("reflect", reflect_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "reflect")
    g.add_conditional_edges("reflect", _branch_router, {"retry": "retrieve", "done": END})
    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)


def answer(question: str) -> AgentResult:
    logger.info(f"[agent] q={question!r}")
    state: GraphState = {"question": question, "loops": 0, "trace": []}
    final: GraphState = _graph().invoke(state)
    return AgentResult(
        question=question,
        answer=final.get("answer", ""),
        citations=final.get("citations", []),
        trace=final.get("trace", []),
        plan=final.get("plan", {}),
    )
