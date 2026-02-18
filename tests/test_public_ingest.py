from __future__ import annotations

from pathlib import Path

from pipeline.docling_parser import ParsedChunk
from pipeline.kg_writer import KGWriter
from pipeline.public_ingest_runner import iter_input_files, run_public_ingest


def test_iter_input_files_collects_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "d.json").write_text("x", encoding="utf-8")

    files = sorted([p.name for p in iter_input_files(tmp_path)])
    assert files == ["a.txt", "b.md", "c.pdf"]


def test_kg_writer_private_relation_merge_includes_farm_id(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    writer = KGWriter()
    queries = []
    monkeypatch.setattr(writer, "_run", lambda q: queries.append(q))

    writer.write_relations(
        [
            {
                "source": "e1",
                "target": "e2",
                "type": "CAUSES",
                "confidence": 0.9,
                "evidence": "x",
                "source_type": "Crop",
                "target_type": "Environment",
                "source_text": "tomato",
                "target_text": "humidity",
            }
        ],
        tier="private",
        farm_id="farm-a",
        created_at="2026-02-12T00:00:00+00:00",
    )

    assert queries
    q = queries[0]
    assert "canonical_id:'e1'" in q
    assert "canonical_id:'e2'" in q
    assert "tier:'private'" in q
    assert "farm_id:'farm-a'" in q
    assert "(s:Entity:Crop" in q
    assert "(t:Entity:Environment" in q
    assert "s.name='tomato'" in q
    assert "t.name='humidity'" in q
    assert "r.confidence" in q


def test_kg_writer_chunk_creates_document_link(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    writer = KGWriter()
    queries = []
    monkeypatch.setattr(writer, "_run", lambda q: queries.append(q))

    writer.write_chunk(
        chunk_id="doc#c0",
        text="smartfarm text",
        metadata={
            "source_type": "document",
            "source_doc": "manual.pdf",
            "created_at": "2026-02-12T00:00:00+00:00",
        },
        tier="public",
        farm_id="",
    )

    assert len(queries) >= 2
    doc_q = queries[1]
    assert "MERGE (d:Document" in doc_q
    assert "HAS_CHUNK" in doc_q
    assert "manual.pdf" in doc_q


def test_run_public_ingest_smoke_with_injected_components(tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("smartfarm sample", encoding="utf-8")

    class _Parser:
        def parse_file(self, _path):
            return [
                ParsedChunk(
                    chunk_id="doc#c0",
                    text="tomato humidity control",
                    metadata={
                        "modality": "table",
                        "asset_ref": "page:1#tbl:0",
                        "table_html_ref": "<table><tr><td>x</td></tr></table>",
                        "created_at": "2026-02-12T00:00:00+00:00",
                    },
                )
            ]

    class _Extractor:
        def extract(self, _text):
            return {
                "entities": [{"canonical_id": "tomato", "type": "Crop", "confidence": 0.9}],
                "relations": [
                    {
                        "source": "tomato",
                        "target": "humidity",
                        "type": "AFFECTS",
                        "confidence": 0.8,
                        "evidence": "table",
                    }
                ],
            }

    class _Vectors:
        def __init__(self):
            self.calls = []

        def upsert_chunk(self, *, chunk_id, text, payload):
            self.calls.append((chunk_id, text, payload))
            return True

    class _KG:
        def __init__(self):
            self.chunk_calls = []
            self.entity_calls = []
            self.relation_calls = []

        def write_chunk(self, **kwargs):
            self.chunk_calls.append(kwargs)

        def write_entities(self, entities, *, tier="public", farm_id="", created_at="", chunk_id=""):
            self.entity_calls.append(
                {
                    "entities": list(entities),
                    "tier": tier,
                    "farm_id": farm_id,
                    "created_at": created_at,
                    "chunk_id": chunk_id,
                }
            )

        def write_relations(self, relations, *, tier="public", farm_id="", created_at=""):
            self.relation_calls.append(
                {
                    "relations": list(relations),
                    "tier": tier,
                    "farm_id": farm_id,
                    "created_at": created_at,
                }
            )

    vectors = _Vectors()
    kg = _KG()

    rc = run_public_ingest(
        input_dir=tmp_path,
        qdrant_host="localhost",
        qdrant_port=6333,
        falkor_host="localhost",
        falkor_port=6379,
        parser=_Parser(),
        extractor=_Extractor(),
        vectors=vectors,
        kg=kg,
    )

    assert rc == 0
    assert len(vectors.calls) == 1
    assert vectors.calls[0][2]["table_html_ref"]
    assert vectors.calls[0][2]["canonical_doc_id"] == "doc.txt"
    assert vectors.calls[0][2]["canonical_chunk_id"] == "doc#c0"
    assert vectors.calls[0][2]["doc_id"] == "doc.txt"
    assert vectors.calls[0][2]["chunk_id"] == "doc#c0"
    assert len(kg.chunk_calls) == 1
    assert len(kg.entity_calls) == 1
    assert len(kg.relation_calls) == 1
    assert kg.entity_calls[0]["chunk_id"] == "doc#c0"
    assert kg.relation_calls[0]["tier"] == "public"
