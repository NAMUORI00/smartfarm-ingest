from __future__ import annotations

import pytest

import pipeline.env_contract as env_contract


def test_openai_backend_requires_base_and_key(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LLM_BACKEND", "openai_compatible")
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setattr(env_contract, "_WARNED_UNUSED_ENV", False)

    with pytest.raises(ValueError, match="OPENAI_COMPAT_BASE_URL"):
        env_contract.resolve_openai_compat_runtime(default_model="Qwen/Qwen3-4B")

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://example.com/v1")
    with pytest.raises(ValueError, match="OPENAI_COMPAT_API_KEY"):
        env_contract.resolve_openai_compat_runtime(default_model="Qwen/Qwen3-4B")


def test_removed_project_env_key_emits_warning(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("EXTRACTOR_API_KEY", "legacy")
    monkeypatch.setattr(env_contract, "_WARNED_UNUSED_ENV", False)

    with pytest.warns(RuntimeWarning, match="Unused project env keys"):
        env_contract.warn_unused_project_env_keys()
