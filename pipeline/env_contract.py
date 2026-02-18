from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore[override]
        return False

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_WORKSPACE_ROOT / ".env")


ALLOWED_PROJECT_ENV_KEYS = {
    "LLM_BACKEND",
    "OPENAI_COMPAT_BASE_URL",
    "OPENAI_COMPAT_API_KEY",
    "OPENAI_COMPAT_MODEL",
    "JUDGE_RUNTIME",
    "RAGAS_BASE_URL",
    "RAGAS_API_KEY",
    "HF_TOKEN",
}

PROJECT_ENV_PREFIXES = (
    "LLM_",
    "OPENAI_",
    "RAGAS_",
    "JUDGE_",
    "HF_",
    "HUGGING_",
    "EXTRACTOR_",
    "EMBED_",
    "LLMLITE_",
    "QDRANT_",
    "FALKORDB_",
    "PRIVATE_",
    "RETRIEVAL_",
    "SOVEREIGNTY_",
    "UNSTRUCTURED_",
    "CHUNK_",
    "DOCLING_",
    "SENSOR_",
)

DEFAULT_OPENAI_MODEL = "Qwen/Qwen3-4B"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_EMBED_DIM = 512
DEFAULT_TIMEOUT = 30.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_RETRIES = 2

_WARNED_UNUSED_ENV = False


def normalize_llm_backend(raw_backend: str) -> str:
    v = (raw_backend or "openai_compatible").strip().lower()
    if v in {"openai", "openai_compatible", "openai-compatible"}:
        return "openai_compatible"
    return "llama_cpp"


def resolve_llm_backend(default: str = "llama_cpp") -> str:
    return normalize_llm_backend(os.getenv("LLM_BACKEND", default))


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def default_llama_openai_base_url() -> str:
    if _running_in_container():
        return "http://llama:8080/v1"
    return "http://localhost:45857/v1"


def warn_unused_project_env_keys() -> None:
    global _WARNED_UNUSED_ENV
    if _WARNED_UNUSED_ENV:
        return
    unknown = sorted(
        {
            k
            for k in os.environ.keys()
            if (k == "API_KEY" or k.startswith(PROJECT_ENV_PREFIXES)) and k not in ALLOWED_PROJECT_ENV_KEYS
        }
    )
    if unknown:
        warnings.warn(
            "Unused project env keys detected (strict env contract): " + ", ".join(unknown),
            RuntimeWarning,
            stacklevel=2,
        )
    _WARNED_UNUSED_ENV = True


def resolve_openai_compat_runtime(
    *,
    default_model: str,
    default_backend: str = "openai_compatible",
) -> Tuple[str, str, str, str]:
    """Return backend, base_url, api_key, model using strict env contract."""
    warn_unused_project_env_keys()
    backend = resolve_llm_backend(default=default_backend)
    model = str(os.getenv("OPENAI_COMPAT_MODEL", default_model) or default_model).strip()

    if backend == "openai_compatible":
        base_url = str(os.getenv("OPENAI_COMPAT_BASE_URL", "") or "").strip().rstrip("/")
        api_key = str(os.getenv("OPENAI_COMPAT_API_KEY", "") or "").strip()
        if not base_url:
            raise ValueError("OPENAI_COMPAT_BASE_URL is required when LLM_BACKEND=openai_compatible")
        if not api_key:
            raise ValueError("OPENAI_COMPAT_API_KEY is required when LLM_BACKEND=openai_compatible")
        return backend, base_url, api_key, model

    return backend, default_llama_openai_base_url().rstrip("/"), "", model
