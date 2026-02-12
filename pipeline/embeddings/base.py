from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence


@dataclass
class EmbeddingRequest:
    texts: Sequence[str]


class EmbeddingProvider(Protocol):
    name: str

    def embed_texts(self, req: EmbeddingRequest) -> List[List[float]]:
        ...
