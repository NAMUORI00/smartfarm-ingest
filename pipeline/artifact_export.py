from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from pipeline.env_contract import DEFAULT_EMBED_MODEL


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort()
    return files


def _describe_files(base: Path, files: List[Path]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for p in files:
        rel = p.relative_to(base)
        out.append(
            {
                "path": str(rel).replace("\\", "/"),
                "size_bytes": int(p.stat().st_size),
                "sha256": _sha256(p),
            }
        )
    return out


def build_artifact_manifest(
    *,
    qdrant_dir: Path,
    falkordb_dir: Path,
    output_dir: Path,
    model_id: str,
    graph_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    qdrant_files = list(_iter_files(qdrant_dir))
    falkor_files = list(_iter_files(falkordb_dir))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "qdrant": {
                "root": str(qdrant_dir),
                "file_count": len(qdrant_files),
                "files": _describe_files(qdrant_dir, qdrant_files),
            },
            "falkordb": {
                "root": str(falkordb_dir),
                "graph_name": graph_name,
                "file_count": len(falkor_files),
                "files": _describe_files(falkordb_dir, falkor_files),
            },
        },
        "runtime": {
            "embedding_model": model_id,
        },
    }

    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest["manifest_sha256"] = manifest_sha

    path = output_dir / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Export edge-sync artifact manifest")
    p.add_argument("--qdrant-dir", default="data/index/qdrant")
    p.add_argument("--falkordb-dir", default="data/index/falkordb")
    p.add_argument("--output-dir", default="data/index/export")
    p.add_argument("--model-id", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--graph-name", default="smartfarm")
    args = p.parse_args()

    out = build_artifact_manifest(
        qdrant_dir=Path(args.qdrant_dir),
        falkordb_dir=Path(args.falkordb_dir),
        output_dir=Path(args.output_dir),
        model_id=str(args.model_id),
        graph_name=str(args.graph_name),
    )
    print(f"[artifact-export] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
