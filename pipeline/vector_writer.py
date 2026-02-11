from __future__ import annotations

import hashlib
import math
from typing import Dict, List

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


class VectorWriter:
    def __init__(self, host: str = "localhost", port: int = 6333, collection: str = "smartfarm_chunks_v2") -> None:
        self.base = f"http://{host}:{port}"
        self.collection = collection
        self.vector_size = 8

    def _dummy_vector(self, text: str) -> List[float]:
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        vals = [((h[i % len(h)] / 255.0) * 2.0) - 1.0 for i in range(self.vector_size)]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def ensure_collection(self) -> None:
        if httpx is None:
            return
        with httpx.Client(timeout=4.0) as c:
            r = c.get(f"{self.base}/collections/{self.collection}")
            if r.status_code == 200:
                return
            c.put(
                f"{self.base}/collections/{self.collection}",
                json={"vectors": {"size": self.vector_size, "distance": "Cosine"}},
            )

    def upsert_chunk(self, *, chunk_id: str, text: str, payload: Dict[str, str]) -> None:
        if httpx is None:
            return
        self.ensure_collection()
        with httpx.Client(timeout=6.0) as c:
            c.put(
                f"{self.base}/collections/{self.collection}/points",
                json={
                    "points": [
                        {
                            "id": chunk_id,
                            "vector": self._dummy_vector(text),
                            "payload": {"text": text, **payload},
                        }
                    ]
                },
            )
