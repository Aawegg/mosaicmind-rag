"""Stable id helpers."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def new_doc_id() -> str:
    return uuid.uuid4().hex[:16]
