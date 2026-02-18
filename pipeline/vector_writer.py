from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Dict, List

from pipeline.embeddings import EmbeddingRequest, build_embedding_provider
from pipeline.env_contract import DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL

try:
    from qdrant_client import QdrantClient, models
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힣]+")


class VectorWriter:
    def __init__(self, host: str = "localhost", port: int = 6333, collection: str = "smartfarm_chunks") -> None:
        self.base = f"http://{host}:{port}"
        self.collection = collection
        self.vector_size = DEFAULT_EMBED_DIM
        self.embedder = build_embedding_provider()
        self.embed_model = DEFAULT_EMBED_MODEL
        self.client = None
        if QdrantClient is not None:
            self.client = QdrantClient(url=self.base, timeout=8.0)

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
        if self.client is None or models is None:
            return

        try:
            if self.client.collection_exists(self.collection):
                return
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense_text": models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                    "dense_image": models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )
        except Exception:
            return

    def upsert_chunk(self, *, chunk_id: str, text: str, payload: Dict[str, Any]) -> bool:
        if self.client is None or models is None:
            return False
        self.ensure_collection()

        point_payload = dict(payload or {})
        modality = str(point_payload.get("modality") or "text").strip().lower()
        asset_ref = str(point_payload.get("asset_ref") or "").strip()
        point_payload["text"] = text
        point_payload.setdefault("embedding_model", self.embed_model)
        point_payload["modality"] = modality
        point_payload["asset_ref"] = asset_ref or None
        point_payload.setdefault("table_html_ref", None)
        point_payload.setdefault("image_b64_ref", None)
        point_payload.setdefault("formula_latex_ref", None)

        dense_text = self._dense_vector(text)
        image_basis = f"{asset_ref}\n{text}".strip() if modality == "image" and asset_ref else text
        dense_image = self._dense_vector(image_basis)

        sparse = self._sparse_vector(text)
        sparse_vec = models.SparseVector(
            indices=[int(i) for i in sparse.get("indices") or []],
            values=[float(v) for v in sparse.get("values") or []],
        )

        point = models.PointStruct(
            id=str(chunk_id),
            vector={
                "dense_text": dense_text,
                "dense_image": dense_image,
                "sparse": sparse_vec,
            },
            payload=point_payload,
        )
        try:
            self.client.upsert(collection_name=self.collection, points=[point], wait=True)
            return True
        except Exception:
            return False
