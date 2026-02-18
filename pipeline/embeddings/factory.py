from __future__ import annotations

from pipeline.embeddings.huggingface_api_provider import HuggingFaceAPIEmbeddingProvider
from pipeline.env_contract import warn_unused_project_env_keys


def build_embedding_provider():
    warn_unused_project_env_keys()
    return HuggingFaceAPIEmbeddingProvider()
