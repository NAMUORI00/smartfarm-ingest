from __future__ import annotations

from pathlib import Path

from pipeline.docling_parser import DoclingParser


def test_docling_parser_text_fallback(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("tomato humidity control memo", encoding="utf-8")

    p = DoclingParser()
    p._converter = None  # type: ignore[attr-defined]
    chunks = p.parse_file(f)

    assert len(chunks) >= 1
    assert chunks[0].metadata.get("modality") == "text"


def test_docling_parser_modality_segments(tmp_path: Path) -> None:
    f = tmp_path / "sample.pdf"
    f.write_text("dummy", encoding="utf-8")

    p = DoclingParser()
    p._iter_docling_segments = lambda _path: [  # type: ignore[assignment]
        {"text": "plain text section", "modality": "text", "asset_ref": None},
        {"text": "table section", "modality": "table", "asset_ref": "page:1#tbl:0"},
        {"text": "image caption", "modality": "image", "asset_ref": "page:2#img:0"},
        {"text": "formula x+y", "modality": "formula", "asset_ref": "page:2#eq:0"},
    ]

    chunks = p.parse_file(f)
    mods = [c.metadata.get("modality") for c in chunks]

    assert "text" in mods
    assert "table" in mods
    assert "image" in mods
    assert "formula" in mods
