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


def test_docling_parser_token_chunking_for_text(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DOCLING_CHUNK_TOKEN_SIZE", "4")
    monkeypatch.setenv("DOCLING_CHUNK_TOKEN_OVERLAP", "1")

    f = tmp_path / "sample.txt"
    f.write_text("a b c d e f g h", encoding="utf-8")

    p = DoclingParser()
    p._converter = None  # type: ignore[attr-defined]
    chunks = p.parse_file(f)

    assert len(chunks) == 3
    assert chunks[0].text == "a b c d"
    assert chunks[1].text == "d e f g"
    assert chunks[2].text == "g h"


def test_docling_parser_modality_segments(tmp_path: Path) -> None:
    f = tmp_path / "sample.pdf"
    f.write_text("dummy", encoding="utf-8")

    p = DoclingParser()
    p._iter_docling_segments = lambda _path: [  # type: ignore[assignment]
        {
            "text": "plain text section",
            "modality": "text",
            "asset_ref": None,
            "table_html": None,
            "formula_latex": None,
            "image_b64_ref": None,
        },
        {
            "text": "table section with many words that should remain a single modality chunk",
            "modality": "table",
            "asset_ref": "page:1#tbl:0",
            "table_html": "<table><tr><td>a</td></tr></table>",
            "formula_latex": None,
            "image_b64_ref": None,
        },
        {
            "text": "image caption",
            "modality": "image",
            "asset_ref": "page:2#img:0",
            "table_html": None,
            "formula_latex": None,
            "image_b64_ref": "ZmFrZV9pbWFnZQ==",
        },
        {
            "text": "formula x+y",
            "modality": "formula",
            "asset_ref": "page:2#eq:0",
            "table_html": None,
            "formula_latex": "x+y",
            "image_b64_ref": None,
        },
    ]

    chunks = p.parse_file(f)
    mods = [c.metadata.get("modality") for c in chunks]

    assert "text" in mods
    assert "table" in mods
    assert "image" in mods
    assert "formula" in mods

    table_chunk = next(c for c in chunks if c.metadata.get("modality") == "table")
    image_chunk = next(c for c in chunks if c.metadata.get("modality") == "image")
    formula_chunk = next(c for c in chunks if c.metadata.get("modality") == "formula")

    assert table_chunk.metadata.get("table_html_ref")
    assert image_chunk.metadata.get("image_b64_ref")
    assert formula_chunk.metadata.get("formula_latex_ref")
    assert len([c for c in chunks if c.metadata.get("modality") == "table"]) == 1
