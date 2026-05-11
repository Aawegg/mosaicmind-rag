"""Test the modality router."""
from __future__ import annotations

from pathlib import Path

import pytest

from mosaicmind.ingestion.router import detect_kind


@pytest.mark.parametrize(
    "name,kind",
    [
        ("a.pdf", "pdf"),
        ("a.PDF", "pdf"),
        ("photo.jpg", "image"),
        ("photo.PNG", "image"),
        ("song.mp3", "audio"),
        ("call.wav", "audio"),
        ("clip.mp4", "video"),
        ("movie.MOV", "video"),
    ],
)
def test_detect_kind(name, kind):
    assert detect_kind(Path(name)) == kind


def test_detect_kind_unknown():
    with pytest.raises(ValueError):
        detect_kind(Path("notes.txt"))
