"""Run the eval set defined in scripts/eval_cases.json and print metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mosaicmind.mlops.eval import EvalCase, run_eval
from mosaicmind.utils.logging import setup_logging

CASES_PATH = Path(__file__).parent / "eval_cases.json"


def main() -> int:
    setup_logging()
    cases = [EvalCase(**c) for c in json.loads(CASES_PATH.read_text())]
    out = run_eval(cases, run_name="cli")

    c = Console()
    t = Table(title=f"Eval {out['run_name']}: avg={out['avg_score']:.2f}/10", show_lines=True)
    t.add_column("question", style="cyan", overflow="fold")
    t.add_column("score", justify="right", style="green")
    t.add_column("latency_ms", justify="right")
    t.add_column("citations", justify="right")
    for r in out["rows"]:
        t.add_row(r["question"][:80], f"{r['score']:.1f}", str(r["latency_ms"]), str(r["n_citations"]))
    c.print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
