"""Quick local query against the agent.

Usage:
    python scripts/cli_query.py "what does the report say about Q3 revenue?"
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mosaicmind.agents import answer
from mosaicmind.utils.logging import setup_logging


def main() -> int:
    setup_logging()
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    question = " ".join(sys.argv[1:])

    console = Console()
    res = answer(question)
    console.print(Panel.fit(question, title="question", border_style="cyan"))
    console.print(Panel(Markdown(res.answer or "(empty)"), title="answer", border_style="green"))
    if res.citations:
        console.rule("citations")
        for c in res.citations:
            loc = ""
            if c.page is not None:
                loc = f"  page={c.page}"
            elif c.timestamp_s is not None:
                loc = f"  t={c.timestamp_s}s"
            console.print(f"[{c.n}] {c.source_name}{loc}  ({c.modality}, score={c.score:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
