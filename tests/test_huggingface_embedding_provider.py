from __future__ import annotations

import pytest

from pipeline.embeddings.base import EmbeddingRequest
from pipeline.embeddings import huggingface_api_provider as mod
from pipeline.env_contract import DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL


def test_hf_provider_embeds_texts(monkeypatch):  # type: ignore[no-untyped-def]
    calls = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _fake_post(url, json, headers, timeout):  # noqa: ANN001
        calls.append((url, json, headers, timeout))
        return _Resp([0.2] * int(DEFAULT_EMBED_DIM))

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    provider = mod.HuggingFaceAPIEmbeddingProvider()

    vectors = provider.embed_texts(EmbeddingRequest(texts=["a", "a"]))
    assert len(vectors) == 2
    assert len(vectors[0]) == int(DEFAULT_EMBED_DIM)
    assert DEFAULT_EMBED_MODEL in calls[0][0]
    assert len(calls) == 1


def test_hf_provider_rejects_rank_mismatch(monkeypatch):  # type: ignore[no-untyped-def]
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"bad": "format"}

    monkeypatch.setattr(mod.requests, "post", lambda *args, **kwargs: _Resp())  # noqa: ARG005
    provider = mod.HuggingFaceAPIEmbeddingProvider()

    with pytest.raises(RuntimeError, match="list\\[float\\]"):
        provider.embed_texts(EmbeddingRequest(texts=["a"]))
