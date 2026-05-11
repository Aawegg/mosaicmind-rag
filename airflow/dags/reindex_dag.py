"""Nightly DAG: scan upload dir for new files and re-ingest them.

The DAG file imports MosaicMind lazily (inside tasks) so Airflow can parse
the DAG even when the package isn't installed in the scheduler environment.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "mosaicmind",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _scan_and_ingest(**_):
    from pathlib import Path

    from mosaicmind.config import get_settings
    from mosaicmind.indexing.store import add_ingest_result
    from mosaicmind.ingestion.router import detect_kind, ingest_path

    settings = get_settings()
    seen_marker = settings.data_dir / ".seen.txt"
    seen = set(seen_marker.read_text().splitlines()) if seen_marker.exists() else set()

    new_files: list[Path] = []
    for p in settings.upload_dir.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            detect_kind(p)
        except ValueError:
            continue
        key = str(p.relative_to(settings.upload_dir))
        if key not in seen:
            new_files.append(p)

    print(f"[reindex] found {len(new_files)} new files")
    for p in new_files:
        result = ingest_path(p)
        add_ingest_result(result)
        seen.add(str(p.relative_to(settings.upload_dir)))

    seen_marker.write_text("\n".join(sorted(seen)))


with DAG(
    dag_id="mosaicmind_reindex",
    default_args=DEFAULT_ARGS,
    description="Scan upload dir nightly and ingest new files into the index.",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mosaicmind", "rag"],
) as dag:
    PythonOperator(task_id="scan_and_ingest", python_callable=_scan_and_ingest)
