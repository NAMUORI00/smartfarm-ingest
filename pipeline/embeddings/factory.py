from __future__ import annotations

from pipeline.embeddings.llama_cpp_provider import LlamaCppEmbeddingProvider
from pipeline.embeddings.openai_compat_provider import OpenAICompatEmbeddingProvider
from pipeline.env_contract import normalize_llm_backend, resolve_llm_backend, warn_unused_project_env_keys


def _normalize_backend(raw_backend: str) -> str:
    return normalize_llm_backend(raw_backend)


def build_embedding_provider():
    warn_unused_project_env_keys()
    backend = _normalize_backend(resolve_llm_backend(default="llama_cpp"))
    if backend == "openai_compatible":
        return OpenAICompatEmbeddingProvider()
    return LlamaCppEmbeddingProvider()
