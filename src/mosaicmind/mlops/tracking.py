"""Thin MLflow wrapper. Failures are logged and swallowed so the API stays up."""
from __future__ import annotations

import json
from typing import Any

import mlflow

from mosaicmind.config import get_settings
from mosaicmind.utils.logging import logger

_INITIALIZED = False


def init_mlflow() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    s = get_settings()
    try:
        mlflow.set_tracking_uri(s.mlflow_uri)
        mlflow.set_experiment(s.mlflow_experiment)
        _INITIALIZED = True
        logger.info(f"[mlflow] tracking_uri={s.mlflow_uri} experiment={s.mlflow_experiment}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[mlflow] init failed (continuing without tracking): {e}")


def log_ingest(*, doc_id: str, source_name: str, kind: str, text_nodes: int, image_nodes: int) -> None:
    init_mlflow()
    try:
        with mlflow.start_run(run_name=f"ingest:{kind}:{source_name}"):
            mlflow.log_params({"event": "ingest", "doc_id": doc_id, "kind": kind, "source": source_name})
            mlflow.log_metrics({"text_nodes": text_nodes, "image_nodes": image_nodes})
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[mlflow] log_ingest skipped: {e}")


def log_query(*, question: str, answer: str, plan: dict, n_citations: int, latency_ms: int) -> None:
    init_mlflow()
    try:
        with mlflow.start_run(run_name="query"):
            s = get_settings()
            mlflow.log_params({
                "event": "query",
                "heavy_model": s.heavy_model,
                "fast_model": s.fast_model,
                "question_chars": len(question),
            })
            mlflow.log_metrics({"latency_ms": latency_ms, "n_citations": n_citations})
            mlflow.log_text(question, "question.txt")
            mlflow.log_text(answer, "answer.txt")
            mlflow.log_text(json.dumps(plan, indent=2), "plan.json")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[mlflow] log_query skipped: {e}")


def log_eval_run(name: str, metrics: dict[str, Any], rows: list[dict]) -> None:
    init_mlflow()
    try:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"event": "eval", "n_cases": len(rows)})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_text(json.dumps(rows, indent=2, default=str), "rows.json")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[mlflow] log_eval_run skipped: {e}")
