from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from pipeline.embeddings import EmbeddingRequest, build_embedding_provider

_TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힣]+")


class VectorWriter:
    def __init__(self, host: str = "localhost", port: int = 6333, collection: str = "smartfarm_chunks_v2") -> None:
        self.base = f"http://{host}:{port}"
        self.collection = collection
        self.vector_size = int(os.getenv("EMBED_DIM", "512"))
        self.embedder = build_embedding_provider()
        self.embed_model = str(os.getenv("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in _TOKEN_RE.findall(str(text or "").lower()) if t]

    def _sparse_vector(self, text: str) -> Dict[str, List[float] | List[int]]:
        tokens = self._tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        counts = Counter(tokens)
        indices: List[int] = []
        values: List[float] = []
        for tok, cnt in counts.items():
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % 65536
            indices.append(idx)
            values.append(1.0 + math.log1p(float(cnt)))

        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        values = [v / norm for v in values]
        return {"indices": indices, "values": values}

    def _dense_vector(self, text: str) -> List[float]:
        vectors = self.embedder.embed_texts(EmbeddingRequest(texts=[str(text or "")]))
        if not vectors:
            raise RuntimeError("embedding provider returned empty vector")
        vec = vectors[0]
        if len(vec) != self.vector_size:
            raise RuntimeError(f"embedding dim mismatch: expected={self.vector_size} got={len(vec)}")
        return vec

    def ensure_collection(self) -> None:
        if httpx is None:
            return
        with httpx.Client(timeout=4.0) as c:
            r = c.get(f"{self.base}/collections/{self.collection}")
            if r.status_code == 200:
                return
            c.put(
                f"{self.base}/collections/{self.collection}",
                json={
                    "vectors": {
                        "dense_text": {"size": self.vector_size, "distance": "Cosine"},
                        "dense_image": {"size": self.vector_size, "distance": "Cosine"},
                    },
                    "sparse_vectors": {"sparse": {"modifier": "idf"}},
                },
            )

    def upsert_chunk(self, *, chunk_id: str, text: str, payload: Dict[str, Any]) -> None:
        if httpx is None:
            return
        self.ensure_collection()

        point_payload = dict(payload or {})
        modality = str(point_payload.get("modality") or "text").strip().lower()
        asset_ref = str(point_payload.get("asset_ref") or "").strip()
        point_payload["text"] = text
        point_payload.setdefault("embedding_model", self.embed_model)
        point_payload["modality"] = modality
        point_payload["asset_ref"] = asset_ref or None

        dense_text = self._dense_vector(text)
        image_basis = f"{asset_ref}\n{text}".strip() if modality == "image" and asset_ref else text
        dense_image = self._dense_vector(image_basis)

        with httpx.Client(timeout=6.0) as c:
            c.put(
                f"{self.base}/collections/{self.collection}/points",
                json={
                    "points": [
                        {
                            "id": chunk_id,
                            "vector": {
                                "dense_text": dense_text,
                                "dense_image": dense_image,
                                "sparse": self._sparse_vector(text),
                            },
                            "payload": point_payload,
                        }
                    ]
                },
            )
