from __future__ import annotations

import pipeline.vector_writer as vector_writer_module
from pipeline.env_contract import DEFAULT_EMBED_DIM
from pipeline.vector_writer import VectorWriter


def test_vector_writer_dense_vector_dim(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _FakeEmbedder:
        name = "fake"

        def embed_texts(self, req):
            return [[0.01] * int(DEFAULT_EMBED_DIM) for _ in req.texts]

    monkeypatch.setattr("pipeline.vector_writer.build_embedding_provider", lambda: _FakeEmbedder())
    w = VectorWriter()
    vec = w._dense_vector("test")
    assert len(vec) == int(DEFAULT_EMBED_DIM)


def test_vector_writer_upsert_uses_named_vectors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _FakeEmbedder:
        name = "fake"

        def embed_texts(self, req):
            return [[0.01] * int(DEFAULT_EMBED_DIM) for _ in req.texts]

    class _FakeModels:
        class Distance:
            COSINE = "Cosine"

        class Modifier:
            IDF = "idf"

        class VectorParams:
            def __init__(self, *, size, distance):
                self.size = size
                self.distance = distance

        class SparseVectorParams:
            def __init__(self, *, modifier=None):
                self.modifier = modifier

        class SparseVector:
            def __init__(self, *, indices, values):
                self.indices = indices
                self.values = values

        class PointStruct:
            def __init__(self, *, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    calls = []

    class _FakeClient:
        def __init__(self, url=None, timeout=None):  # noqa: ARG002
            pass

        def collection_exists(self, _collection):
            calls.append(("collection_exists", _collection))
            return False

        def create_collection(self, **kwargs):
            calls.append(("create_collection", kwargs))
            return True

        def upsert(self, *, collection_name, points, wait=True):
            calls.append(("upsert", collection_name, points, wait))
            return True

    monkeypatch.setattr("pipeline.vector_writer.build_embedding_provider", lambda: _FakeEmbedder())
    monkeypatch.setattr(vector_writer_module, "QdrantClient", _FakeClient)
    monkeypatch.setattr(vector_writer_module, "models", _FakeModels)

    w = VectorWriter()
    ok = w.upsert_chunk(
        chunk_id="chunk-1",
        text="image caption text",
        payload={"tier": "public", "source_type": "document", "modality": "image", "asset_ref": "img://1"},
    )

    assert ok is True
    upserts = [c for c in calls if c[0] == "upsert"]
    assert upserts
    point = upserts[-1][2][0]
    vectors = point.vector
    assert "dense_text" in vectors
    assert "dense_image" in vectors
    assert "sparse" in vectors
    payload = point.payload
    assert payload["canonical_doc_id"] == "chunk-1"
    assert payload["canonical_chunk_id"] == "chunk-1"
    assert payload["doc_id"] == "chunk-1"
    assert payload["chunk_id"] == "chunk-1"


def test_vector_writer_upsert_payload_includes_canonical_ids(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _FakeEmbedder:
        name = "fake"

        def embed_texts(self, req):
            return [[0.01] * int(DEFAULT_EMBED_DIM) for _ in req.texts]

    class _FakeModels:
        class Distance:
            COSINE = "Cosine"

        class Modifier:
            IDF = "idf"

        class VectorParams:
            def __init__(self, *, size, distance):
                self.size = size
                self.distance = distance

        class SparseVectorParams:
            def __init__(self, *, modifier=None):
                self.modifier = modifier

        class SparseVector:
            def __init__(self, *, indices, values):
                self.indices = indices
                self.values = values

        class PointStruct:
            def __init__(self, *, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    calls = []

    class _FakeClient:
        def __init__(self, url=None, timeout=None):  # noqa: ARG002
            pass

        def collection_exists(self, _collection):
            return False

        def create_collection(self, **kwargs):  # noqa: ARG002
            return True

        def upsert(self, *, collection_name, points, wait=True):  # noqa: ARG002
            calls.append(points[0])
            return True

    monkeypatch.setattr("pipeline.vector_writer.build_embedding_provider", lambda: _FakeEmbedder())
    monkeypatch.setattr(vector_writer_module, "QdrantClient", _FakeClient)
    monkeypatch.setattr(vector_writer_module, "models", _FakeModels)

    w = VectorWriter()
    ok = w.upsert_chunk(
        chunk_id="manual#t3",
        text="sample text",
        payload={
            "source_doc": "manual-v2.pdf",
            "doc_id": "legacy-doc-id",
            "tier": "public",
        },
    )

    assert ok is True
    assert calls
    payload = calls[-1].payload
    assert payload["canonical_doc_id"] == "manual-v2.pdf"
    assert payload["canonical_chunk_id"] == "manual#t3"
    assert payload["doc_id"] == "legacy-doc-id"
    assert payload["chunk_id"] == "manual#t3"
