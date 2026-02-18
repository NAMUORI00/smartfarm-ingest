from __future__ import annotations

from pathlib import Path

from pipeline.docling_parser import ParsedChunk
from pipeline.reproducibility_check import build_repro_manifest


def test_repro_manifest_deterministic_with_injected_parser(tmp_path: Path) -> None:
    f = tmp_path / "doc1.txt"
    f.write_text("abc", encoding="utf-8")

    class _Parser:
        def parse_file(self, path):
            _ = path
            return [
                ParsedChunk(chunk_id="doc1#c0", text="tomato humidity", metadata={"modality": "text"}),
                ParsedChunk(chunk_id="doc1#c1", text="pepper ec", metadata={"modality": "text"}),
            ]

    class _Extractor:
        def extract(self, text: str):
            if "tomato" in text:
                return {"entities": [{"canonical_id": "tomato"}], "relations": []}
            return {"entities": [], "relations": [{"source": "pepper", "target": "ec", "type": "RELATED"}]}

    m1 = build_repro_manifest(input_dir=tmp_path, include_extractor=True, parser=_Parser(), extractor=_Extractor())
    m2 = build_repro_manifest(input_dir=tmp_path, include_extractor=True, parser=_Parser(), extractor=_Extractor())

    assert m1["chunk_total"] == 2
    assert m1["entity_total"] == 1
    assert m1["relation_total"] == 1
    assert m1["manifest_sha256"] == m2["manifest_sha256"]

