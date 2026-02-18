from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_source_root(
    *,
    explicit: Path | None,
    manifest_root: str,
    fallback_dir: Path,
    label: str,
) -> Path:
    if explicit is not None:
        return explicit

    candidate = Path(str(manifest_root or "")).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if fallback_dir.exists():
        return fallback_dir.resolve()

    raise FileNotFoundError(
        f"cannot resolve source root for {label}: "
        f"explicit=None manifest_root={manifest_root!r} fallback={fallback_dir}"
    )


def _copy_manifest_entries(
    *,
    source_root: Path,
    target_root: Path,
    entries: Iterable[Dict[str, object]],
    verify_sha256: bool,
) -> Tuple[int, int]:
    copied = 0
    copied_bytes = 0
    target_root.mkdir(parents=True, exist_ok=True)

    for e in entries:
        rel = str(e.get("path") or "").strip().replace("\\", "/")
        if not rel:
            continue
        src = (source_root / rel).resolve()
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"artifact file missing: {src}")

        if verify_sha256:
            expected = str(e.get("sha256") or "").strip().lower()
            if expected:
                got = _sha256(src).lower()
                if got != expected:
                    raise ValueError(f"sha256 mismatch for {src}: expected={expected} got={got}")

        dst = (target_root / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        copied += 1
        copied_bytes += int(dst.stat().st_size)

    return copied, copied_bytes


def import_artifact_manifest(
    *,
    manifest_path: Path,
    qdrant_target_dir: Path,
    falkordb_target_dir: Path,
    qdrant_source_dir: Path | None = None,
    falkordb_source_dir: Path | None = None,
    verify_sha256: bool = True,
) -> Dict[str, object]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = data.get("artifacts") or {}
    qdrant_block = artifacts.get("qdrant") or {}
    falkor_block = artifacts.get("falkordb") or {}

    qdrant_entries = qdrant_block.get("files") or []
    falkor_entries = falkor_block.get("files") or []
    if not isinstance(qdrant_entries, list):
        qdrant_entries = []
    if not isinstance(falkor_entries, list):
        falkor_entries = []

    qdrant_src = _resolve_source_root(
        explicit=qdrant_source_dir,
        manifest_root=str(qdrant_block.get("root") or ""),
        fallback_dir=manifest_path.parent / "qdrant",
        label="qdrant",
    )
    falkor_src = _resolve_source_root(
        explicit=falkordb_source_dir,
        manifest_root=str(falkor_block.get("root") or ""),
        fallback_dir=manifest_path.parent / "falkordb",
        label="falkordb",
    )

    q_count, q_bytes = _copy_manifest_entries(
        source_root=qdrant_src,
        target_root=qdrant_target_dir,
        entries=qdrant_entries,
        verify_sha256=bool(verify_sha256),
    )
    f_count, f_bytes = _copy_manifest_entries(
        source_root=falkor_src,
        target_root=falkordb_target_dir,
        entries=falkor_entries,
        verify_sha256=bool(verify_sha256),
    )

    return {
        "manifest_path": str(manifest_path),
        "verify_sha256": bool(verify_sha256),
        "qdrant": {"source_root": str(qdrant_src), "target_root": str(qdrant_target_dir), "copied_files": q_count, "copied_bytes": q_bytes},
        "falkordb": {"source_root": str(falkor_src), "target_root": str(falkordb_target_dir), "copied_files": f_count, "copied_bytes": f_bytes},
        "runtime": data.get("runtime") or {},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Import edge-sync artifact manifest")
    p.add_argument("--manifest", required=True, help="artifact_manifest.json path")
    p.add_argument("--qdrant-dir", default="data/index/qdrant", help="target qdrant dir")
    p.add_argument("--falkordb-dir", default="data/index/falkordb", help="target falkordb dir")
    p.add_argument("--qdrant-source-dir", default="", help="optional source qdrant dir override")
    p.add_argument("--falkordb-source-dir", default="", help="optional source falkordb dir override")
    p.add_argument("--skip-sha256-check", action="store_true", help="skip file checksum validation")
    args = p.parse_args()

    summary = import_artifact_manifest(
        manifest_path=Path(args.manifest),
        qdrant_target_dir=Path(args.qdrant_dir),
        falkordb_target_dir=Path(args.falkordb_dir),
        qdrant_source_dir=Path(args.qdrant_source_dir) if str(args.qdrant_source_dir).strip() else None,
        falkordb_source_dir=Path(args.falkordb_source_dir) if str(args.falkordb_source_dir).strip() else None,
        verify_sha256=not bool(args.skip_sha256_check),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
