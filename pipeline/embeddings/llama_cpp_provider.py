from __future__ import annotations

from typing import Any, Dict, List, Sequence

import httpx

from pipeline.env_contract import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    default_llama_openai_base_url,
)
from pipeline.embeddings.base import EmbeddingRequest


class LlamaCppEmbeddingProvider:
    name = "llama_cpp"

    def __init__(self) -> None:
        self.base_url = default_llama_openai_base_url().rstrip("/")
        self.model = DEFAULT_EMBED_MODEL
        self.dim = DEFAULT_EMBED_DIM
        self.timeout = DEFAULT_TIMEOUT
        self.batch_size = DEFAULT_BATCH_SIZE
        self.max_retries = DEFAULT_MAX_RETRIES

    def _validate_dim(self, vectors: List[List[float]]) -> List[List[float]]:
        for v in vectors:
            if len(v) != self.dim:
                raise RuntimeError(f"embedding dim mismatch: expected={self.dim} got={len(v)}")
        return vectors

    def _coerce_vectors(self, payload: Dict[str, Any], expected: int) -> List[List[float]]:
        data = payload.get("data")
        if isinstance(data, list):
            vectors: List[List[float]] = []
            for item in data:
                emb = (item or {}).get("embedding")
                if isinstance(emb, list):
                    vectors.append([float(x) for x in emb])
            if vectors:
                if len(vectors) != expected:
                    raise RuntimeError(f"embedding count mismatch: expected={expected} got={len(vectors)}")
                return self._validate_dim(vectors)

        emb = payload.get("embedding")
        if expected == 1 and isinstance(emb, list):
            return self._validate_dim([[float(x) for x in emb]])

        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            vectors = [[float(x) for x in v] for v in embeddings]
            if len(vectors) != expected:
                raise RuntimeError(f"embedding count mismatch: expected={expected} got={len(vectors)}")
            return self._validate_dim(vectors)

        raise RuntimeError("llama.cpp embedding response missing vectors")

    def _embed_batch_with_endpoint(self, *, texts: Sequence[str], endpoint: str) -> List[List[float]]:
        payload = {
            "input": [str(t or "") for t in texts],
            "model": self.model,
            "encoding_format": "float",
        }
        timeout = httpx.Timeout(connect=1.5, read=self.timeout, write=5.0, pool=5.0)
        r = httpx.post(f"{self.base_url}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return self._coerce_vectors(r.json() or {}, expected=len(texts))

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            for endpoint in ("/v1/embeddings", "/embeddings"):
                try:
                    return self._embed_batch_with_endpoint(texts=texts, endpoint=endpoint)
                except Exception as exc:
                    last_err = exc
        raise RuntimeError(f"llama.cpp embedding failed: {last_err}")

    def embed_texts(self, req: EmbeddingRequest) -> List[List[float]]:
        texts = [str(t or "") for t in req.texts]
        if not texts:
            return []

        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return out
