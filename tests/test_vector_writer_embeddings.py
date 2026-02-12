from __future__ import annotations

import pipeline.vector_writer as vector_writer_module
from pipeline.vector_writer import VectorWriter


def test_vector_writer_dense_vector_dim(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _FakeEmbedder:
        name = "fake"

        def embed_texts(self, req):
            return [[0.01] * 512 for _ in req.texts]

    monkeypatch.setattr("pipeline.vector_writer.build_embedding_provider", lambda: _FakeEmbedder())
    w = VectorWriter()
    vec = w._dense_vector("test")
    assert len(vec) == 512


def test_vector_writer_upsert_uses_named_vectors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _FakeEmbedder:
        name = "fake"

        def embed_texts(self, req):
            return [[0.01] * 512 for _ in req.texts]

    calls = []

    class _Resp:
        def __init__(self, status_code=200):
            self.status_code = status_code

    class _Client:
        def __init__(self, timeout=None):  # noqa: ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def get(self, url):
            calls.append(("get", url, None))
            return _Resp(status_code=404)

        def put(self, url, json=None):
            calls.append(("put", url, json))
            return _Resp(status_code=200)

    class _Httpx:
        Client = _Client

    monkeypatch.setattr("pipeline.vector_writer.build_embedding_provider", lambda: _FakeEmbedder())
    monkeypatch.setattr(vector_writer_module, "httpx", _Httpx())

    w = VectorWriter()
    w.upsert_chunk(
        chunk_id="chunk-1",
        text="image caption text",
        payload={"tier": "public", "source_type": "document", "modality": "image", "asset_ref": "img://1"},
    )

    upserts = [c for c in calls if c[0] == "put" and "/points" in c[1]]
    assert upserts
    body = upserts[-1][2]
    vectors = body["points"][0]["vector"]
    assert "dense_text" in vectors
    assert "dense_image" in vectors
    assert "sparse" in vectors
