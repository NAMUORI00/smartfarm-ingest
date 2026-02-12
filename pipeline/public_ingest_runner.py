from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pipeline.docling_parser import DoclingParser
from pipeline.kg_writer import KGWriter
from pipeline.kimi_extractor import ExtractionInput, KimiExtractor
from pipeline.vector_writer import VectorWriter


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_input_files(root: Path) -> Iterable[Path]:
    for ext in ("*.txt", "*.md", "*.pdf"):
        for p in root.rglob(ext):
            if p.is_file():
                yield p


def run_public_ingest(
    *,
    input_dir: Path,
    qdrant_host: str,
    qdrant_port: int,
    falkor_host: str,
    falkor_port: int,
    parser: DoclingParser | None = None,
    extractor: KimiExtractor | None = None,
    vectors: VectorWriter | None = None,
    kg: KGWriter | None = None,
) -> int:
    parser = parser or DoclingParser()
    extractor = extractor or KimiExtractor()
    vectors = vectors or VectorWriter(host=qdrant_host, port=qdrant_port)
    kg = kg or KGWriter(host=falkor_host, port=falkor_port)

    total_chunks = 0
    total_entities = 0
    total_relations = 0

    for file_path in iter_input_files(input_dir):
        chunks = parser.parse_file(file_path)
        for ch in chunks:
            created_at = str(ch.metadata.get("created_at") or "").strip() or _utc_now_iso()
            payload = {
                "tier": "public",
                "source_type": "document",
                "source_doc": str(file_path.name),
                "modality": ch.metadata.get("modality", "text"),
                "asset_ref": ch.metadata.get("asset_ref"),
                "table_html_ref": ch.metadata.get("table_html_ref"),
                "image_b64_ref": ch.metadata.get("image_b64_ref"),
                "formula_latex_ref": ch.metadata.get("formula_latex_ref"),
                "created_at": created_at,
            }
            vectors.upsert_chunk(chunk_id=ch.chunk_id, text=ch.text, payload=payload)
            kg.write_chunk(chunk_id=ch.chunk_id, text=ch.text, metadata=payload, tier="public", farm_id="")

            extracted = extractor.extract(
                ExtractionInput(
                    text=ch.text,
                    modality=str(ch.metadata.get("modality") or "text"),
                    table_html_ref=(str(ch.metadata.get("table_html_ref")) if ch.metadata.get("table_html_ref") else None),
                    image_b64_ref=(str(ch.metadata.get("image_b64_ref")) if ch.metadata.get("image_b64_ref") else None),
                    formula_latex_ref=(str(ch.metadata.get("formula_latex_ref")) if ch.metadata.get("formula_latex_ref") else None),
                    asset_ref=(str(ch.metadata.get("asset_ref")) if ch.metadata.get("asset_ref") else None),
                    source_doc=str(file_path.name),
                )
            )
            entities = extracted.get("entities") or []
            relations = extracted.get("relations") or []

            kg.write_entities(
                entities,
                tier="public",
                farm_id="",
                created_at=created_at,
                chunk_id=ch.chunk_id,
            )
            kg.write_relations(
                relations,
                tier="public",
                farm_id="",
                created_at=created_at,
            )

            total_chunks += 1
            total_entities += len(entities)
            total_relations += len(relations)

    print(
        f"[public-ingest] done chunks={total_chunks} entities={total_entities} relations={total_relations}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Build public knowledge artifacts (Qdrant + FalkorDB)")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--qdrant-host", default="localhost")
    p.add_argument("--qdrant-port", type=int, default=6333)
    p.add_argument("--falkor-host", default="localhost")
    p.add_argument("--falkor-port", type=int, default=6379)
    args = p.parse_args()

    return run_public_ingest(
        input_dir=Path(args.input_dir),
        qdrant_host=str(args.qdrant_host),
        qdrant_port=int(args.qdrant_port),
        falkor_host=str(args.falkor_host),
        falkor_port=int(args.falkor_port),
    )


if __name__ == "__main__":
    raise SystemExit(main())
