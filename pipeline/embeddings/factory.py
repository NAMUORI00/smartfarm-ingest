from __future__ import annotations

import os

from pipeline.embeddings.llama_cpp_provider import LlamaCppEmbeddingProvider
from pipeline.embeddings.openai_compat_provider import OpenAICompatEmbeddingProvider


def _normalize_backend(raw_backend: str) -> str:
    v = (raw_backend or "openai_compatible").strip().lower()
    if v in {"openai", "openai_compatible", "openai-compatible"}:
        return "openai_compatible"
    return "llama_cpp"


def build_embedding_provider():
    backend = _normalize_backend(str(os.getenv("EMBED_BACKEND", "openai_compatible")))
    if backend == "openai_compatible":
        return OpenAICompatEmbeddingProvider()
    return LlamaCppEmbeddingProvider()
