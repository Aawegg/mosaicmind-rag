"""Index every supported file in data/uploads/. Useful for the first run."""
from __future__ import annotations

import sys
from pathlib import Path

from mosaicmind.config import get_settings
from mosaicmind.indexing.store import add_ingest_result
from mosaicmind.ingestion.router import detect_kind, ingest_path
from mosaicmind.utils.logging import logger, setup_logging


def main() -> int:
    setup_logging()
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    files = [p for p in upload_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if not files:
        logger.warning(f"No files in {upload_dir}. Drop PDFs/images/audio/video there and re-run.")
        return 1
    n_ok = 0
    for p in files:
        try:
            detect_kind(p)
        except ValueError:
            logger.info(f"skip {p}")
            continue
        result = ingest_path(p)
        counts = add_ingest_result(result)
        logger.info(f"indexed {p.name}: {counts}")
        n_ok += 1
    logger.info(f"done. {n_ok} files indexed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
