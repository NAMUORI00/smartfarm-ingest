from __future__ import annotations

import json
from pathlib import Path

from pipeline.artifact_export import build_artifact_manifest
from pipeline.artifact_import import import_artifact_manifest


def test_artifact_import_roundtrip(tmp_path: Path) -> None:
    src_qdrant = tmp_path / "src" / "qdrant"
    src_falkor = tmp_path / "src" / "falkordb"
    src_qdrant.mkdir(parents=True)
    src_falkor.mkdir(parents=True)

    (src_qdrant / "segments.bin").write_bytes(b"abc")
    (src_falkor / "dump.rdb").write_bytes(b"def")

    manifest_dir = tmp_path / "manifest"
    manifest_path = build_artifact_manifest(
        qdrant_dir=src_qdrant,
        falkordb_dir=src_falkor,
        output_dir=manifest_dir,
        model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
        graph_name="smartfarm",
    )

    dst_qdrant = tmp_path / "dst" / "qdrant"
    dst_falkor = tmp_path / "dst" / "falkordb"
    summary = import_artifact_manifest(
        manifest_path=manifest_path,
        qdrant_target_dir=dst_qdrant,
        falkordb_target_dir=dst_falkor,
        qdrant_source_dir=src_qdrant,
        falkordb_source_dir=src_falkor,
        verify_sha256=True,
    )

    assert (dst_qdrant / "segments.bin").read_bytes() == b"abc"
    assert (dst_falkor / "dump.rdb").read_bytes() == b"def"
    assert int(summary["qdrant"]["copied_files"]) == 1  # type: ignore[index]
    assert int(summary["falkordb"]["copied_files"]) == 1  # type: ignore[index]


def test_artifact_import_detects_sha_mismatch(tmp_path: Path) -> None:
    src_qdrant = tmp_path / "src" / "qdrant"
    src_falkor = tmp_path / "src" / "falkordb"
    src_qdrant.mkdir(parents=True)
    src_falkor.mkdir(parents=True)

    (src_qdrant / "segments.bin").write_bytes(b"abc")
    (src_falkor / "dump.rdb").write_bytes(b"def")

    manifest_path = build_artifact_manifest(
        qdrant_dir=src_qdrant,
        falkordb_dir=src_falkor,
        output_dir=tmp_path / "manifest",
        model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
        graph_name="smartfarm",
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["artifacts"]["qdrant"]["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import_artifact_manifest(
            manifest_path=manifest_path,
            qdrant_target_dir=tmp_path / "dst" / "qdrant",
            falkordb_target_dir=tmp_path / "dst" / "falkordb",
            qdrant_source_dir=src_qdrant,
            falkordb_source_dir=src_falkor,
            verify_sha256=True,
        )
    except ValueError as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("expected sha256 mismatch")
