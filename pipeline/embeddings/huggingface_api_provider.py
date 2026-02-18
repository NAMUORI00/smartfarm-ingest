from __future__ import annotations

import os
from typing import List

import requests

from pipeline.embeddings.base import EmbeddingRequest
from pipeline.env_contract import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
)


class HuggingFaceAPIEmbeddingProvider:
    """
    Strict HF Inference API feature-extraction provider.

    Reference:
    - Hugging Face Inference API (feature-extraction task)
      https://huggingface.co/docs/inference-providers/tasks/feature-extraction
    """

    name = "huggingface_api"

    def __init__(self) -> None:
        self.model = DEFAULT_EMBED_MODEL
        self.dim = DEFAULT_EMBED_DIM
        self.batch_size = DEFAULT_BATCH_SIZE
        self.max_retries = DEFAULT_MAX_RETRIES
        self.timeout = float(DEFAULT_TIMEOUT)
        self.token = str(os.getenv("HF_TOKEN", "") or "").strip()
        self.endpoint = (
            f"https://router.huggingface.co/hf-inference/models/{self.model}/pipeline/feature-extraction"
        )
        self._cache: dict[str, List[float]] = {}

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _coerce_vectors(self, payload: object, expected: int) -> List[List[float]]:
        if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
            if expected != 1:
                raise RuntimeError(f"HF embedding count mismatch: expected={expected} got=1")
            return [[float(x) for x in payload]]
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            vectors = []
            for row in payload:
                if not isinstance(row, list) or (row and not isinstance(row[0], (int, float))):
                    raise RuntimeError("HF embedding response must be a list[float] or list[list[float]]")
                vectors.append([float(x) for x in row])
            if len(vectors) != expected:
                raise RuntimeError(f"HF embedding count mismatch: expected={expected} got={len(vectors)}")
            return vectors
        raise RuntimeError("HF embedding response must be a list[float] or list[list[float]]")

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        payload = {"inputs": [str(text or "") for text in texts]}
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self._headers(),
                    timeout=(5.0, self.timeout),
                )
                response.raise_for_status()
                return self._coerce_vectors(response.json(), expected=len(texts))
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"huggingface_api embedding failed: {last_err}")

    def embed_texts(self, req: EmbeddingRequest) -> List[List[float]]:
        texts = [str(t or "") for t in req.texts]
        if not texts:
            return []

        out: List[List[float] | None] = [None] * len(texts)
        missing_index_map: dict[str, List[int]] = {}
        for idx, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                out[idx] = list(cached)
                continue
            missing_index_map.setdefault(text, []).append(idx)

        missing_texts = list(missing_index_map.keys())
        for i in range(0, len(missing_texts), max(1, self.batch_size)):
            batch = missing_texts[i : i + max(1, self.batch_size)]
            vectors = self._request_embeddings(batch)
            for text, vec in zip(batch, vectors):
                if len(vec) != self.dim:
                    raise RuntimeError(f"embedding dim mismatch: expected={self.dim} got={len(vec)}")
                self._cache[text] = list(vec)
                for idx in missing_index_map.get(text, []):
                    out[idx] = list(vec)

        if any(v is None for v in out):
            raise RuntimeError("embedding resolution failed for one or more texts")
        return [list(v) for v in out if v is not None]
