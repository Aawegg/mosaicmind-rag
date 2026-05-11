"""Weekly DAG: run the eval set against the agent and log to MLflow."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "mosaicmind",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

EVAL_FILE = Path("/opt/airflow/dags/../../scripts/eval_cases.json")


def _run_eval(**_):
    import json

    from mosaicmind.mlops.eval import EvalCase, run_eval

    cases_data = json.loads(EVAL_FILE.read_text())
    cases = [EvalCase(**c) for c in cases_data]
    out = run_eval(cases, run_name=f"airflow-{datetime.utcnow().strftime('%Y%m%d')}")
    print(f"[eval-sweep] avg_score={out['avg_score']:.2f} n={out['n']}")


with DAG(
    dag_id="mosaicmind_eval_sweep",
    default_args=DEFAULT_ARGS,
    description="Weekly LLM-as-judge eval sweep, logged to MLflow.",
    schedule="0 4 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mosaicmind", "eval"],
) as dag:
    PythonOperator(task_id="run_eval", python_callable=_run_eval)
