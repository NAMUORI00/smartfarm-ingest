from __future__ import annotations

import json
from pathlib import Path

from pipeline.artifact_export import build_artifact_manifest


def test_artifact_export_manifest(tmp_path: Path) -> None:
    qdrant = tmp_path / "qdrant"
    falkor = tmp_path / "falkordb"
    qdrant.mkdir(parents=True)
    falkor.mkdir(parents=True)

    (qdrant / "segments.bin").write_bytes(b"abc")
    (falkor / "dump.rdb").write_bytes(b"def")

    out_dir = tmp_path / "export"
    manifest_path = build_artifact_manifest(
        qdrant_dir=qdrant,
        falkordb_dir=falkor,
        output_dir=out_dir,
        model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
        graph_name="smartfarm",
    )

    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["artifacts"]["qdrant"]["file_count"] == 1
    assert data["artifacts"]["falkordb"]["file_count"] == 1
    assert data["runtime"]["embedding_model"] == "sentence-transformers/distiluse-base-multilingual-cased-v2"
    assert len(data["manifest_sha256"]) == 64
