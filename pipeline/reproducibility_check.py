from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pipeline.docling_parser import DoclingParser, ParsedChunk
from pipeline.kimi_extractor import KimiExtractor
from pipeline.public_ingest_runner import iter_input_files


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _chunk_digest(chunk: ParsedChunk, *, extracted: Dict[str, Any]) -> str:
    payload = {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "metadata": dict(chunk.metadata or {}),
        "entities": extracted.get("entities") or [],
        "relations": extracted.get("relations") or [],
    }
    return _sha256_text(_stable_json(payload))


def _iter_chunks(parser: DoclingParser, files: Iterable[Path]) -> Iterable[tuple[Path, ParsedChunk]]:
    for f in files:
        for ch in parser.parse_file(f):
            yield f, ch


def build_repro_manifest(
    *,
    input_dir: Path,
    include_extractor: bool = False,
    parser: DoclingParser | None = None,
    extractor: KimiExtractor | None = None,
) -> Dict[str, Any]:
    parser = parser or DoclingParser()
    extractor = extractor or KimiExtractor()

    files = sorted(list(iter_input_files(input_dir)), key=lambda p: str(p))
    per_file: Dict[str, Dict[str, Any]] = {}
    all_digests: List[str] = []

    for f, chunk in _iter_chunks(parser, files):
        rel_path = str(f.relative_to(input_dir))
        entry = per_file.setdefault(rel_path, {"chunks": 0, "entities": 0, "relations": 0, "chunk_digests": []})
        extracted = extractor.extract(chunk.text) if include_extractor else {"entities": [], "relations": []}
        entities = list(extracted.get("entities") or [])
        relations = list(extracted.get("relations") or [])
        digest = _chunk_digest(chunk, extracted={"entities": entities, "relations": relations})

        entry["chunks"] = int(entry["chunks"]) + 1
        entry["entities"] = int(entry["entities"]) + len(entities)
        entry["relations"] = int(entry["relations"]) + len(relations)
        entry["chunk_digests"].append(digest)
        all_digests.append(digest)

    manifest_core = {
        "input_dir": str(input_dir.resolve()),
        "include_extractor": bool(include_extractor),
        "file_count": len(files),
        "chunk_total": sum(int(v["chunks"]) for v in per_file.values()),
        "entity_total": sum(int(v["entities"]) for v in per_file.values()),
        "relation_total": sum(int(v["relations"]) for v in per_file.values()),
        "files": dict(sorted(per_file.items(), key=lambda x: x[0])),
    }
    manifest_core["manifest_sha256"] = _sha256_text(_stable_json(manifest_core))
    return manifest_core


def main() -> int:
    p = argparse.ArgumentParser(description="Deterministic reproducibility manifest for offline ingest inputs")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", default="output/repro/public_ingest_manifest.json")
    p.add_argument("--include-extractor", action="store_true")
    p.add_argument("--compare-manifest", default="")
    args = p.parse_args()

    manifest = build_repro_manifest(
        input_dir=Path(args.input_dir),
        include_extractor=bool(args.include_extractor),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    compare_path = str(args.compare_manifest or "").strip()
    if compare_path:
        other = json.loads(Path(compare_path).read_text(encoding="utf-8"))
        if str(other.get("manifest_sha256") or "") != str(manifest.get("manifest_sha256") or ""):
            print("[FAIL] reproducibility mismatch: manifest hash differs")
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

