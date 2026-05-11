"""Verify the LangGraph wires together correctly using monkeypatched LLM + retrievers."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage


@dataclass
class _FakeHit:
    text: str
    score: float
    metadata: dict


def test_agent_runs_offline(monkeypatch):
    from mosaicmind.agents import graph as graph_mod

    plan_msg = AIMessage(content='{"queries":["q"],"use_images":false,"reasoning":"t"}')
    synth_msg = AIMessage(content="The answer is 42.\n\nSources:\n[1] doc.pdf p.1")
    reflect_msg = AIMessage(content='{"sufficient":true,"missing":"","followup_query":""}')

    class _SeqLLM:
        def __init__(self, msgs):
            self._msgs = list(msgs)

        def invoke(self, _msgs):
            return self._msgs.pop(0)

    fake = _SeqLLM([plan_msg, synth_msg, reflect_msg])
    monkeypatch.setattr(graph_mod, "gemini_chat_model", lambda *a, **k: fake)

    monkeypatch.setattr(
        graph_mod,
        "retrieve_text",
        lambda q, top_k=6: [_FakeHit("evidence about 42", 0.9, {"source_name": "doc.pdf", "page": 1})],
    )
    monkeypatch.setattr(graph_mod, "retrieve_images", lambda q, top_k=4: [])

    graph_mod._GRAPH = None
    res = graph_mod.answer("what is the answer?")
    assert "42" in res.answer
    assert res.citations
    assert res.citations[0].source_name == "doc.pdf"


def test_safe_json_load_handles_codefences():
    from mosaicmind.agents.graph import _safe_json_load

    assert _safe_json_load("```json\n{\"a\": 1}\n```")["a"] == 1
    assert _safe_json_load("not json") == {}
