from __future__ import annotations

from types import SimpleNamespace

from pipeline import llm_extractor as mod
from pipeline.llm_extractor import ExtractionInput, LLMExtractor


def test_extractor_uses_featherless_env_and_modality_payload(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("EXTRACTOR_BASE_URL", "https://api.featherless.ai/v1")
    monkeypatch.setenv("EXTRACTOR_API_KEY", "sk-test")
    monkeypatch.setenv("EXTRACTOR_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    monkeypatch.setenv("EXTRACTOR_MODEL_CANDIDATES", "Qwen/Qwen2.5-14B-Instruct")
    monkeypatch.setenv("EXTRACTOR_HTTP_REFERER", "https://smartfarm.local")
    monkeypatch.setenv("EXTRACTOR_HTTP_TITLE", "smartfarm-ingest")

    calls = []

    def _fake_post(url, json, headers, timeout):  # noqa: ANN001
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"entities":[{"text":"토마토","type":"Crop","canonical_id":"tomato","confidence":0.9}],'
                                '"relations":[]}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(mod.httpx, "post", _fake_post)

    ext = LLMExtractor()
    out = ext.extract(
        ExtractionInput(
            text="표를 보고 토마토 병충해 요인을 추출해줘",
            modality="table",
            table_html_ref="<table><tr><td>토마토</td></tr></table>",
            source_doc="sample.pdf",
        )
    )

    assert out["entities"]
    assert calls
    req = calls[0]
    assert req["url"] == "https://api.featherless.ai/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    assert req["headers"]["HTTP-Referer"] == "https://smartfarm.local"
    assert req["headers"]["X-Title"] == "smartfarm-ingest"
    prompt = str(req["json"]["messages"][0]["content"])
    assert "TABLE_HTML" in prompt
    assert "sample.pdf" in prompt


def test_extractor_model_fallback_order(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("EXTRACTOR_BASE_URL", "https://api.featherless.ai/v1")
    monkeypatch.setenv("EXTRACTOR_API_KEY", "sk-test")
    monkeypatch.setenv("EXTRACTOR_MODEL", "bad/model")
    monkeypatch.setenv("EXTRACTOR_MODEL_CANDIDATES", "good/model")

    models = []

    def _fake_post(url, json, headers, timeout):  # noqa: ANN001
        model = str(json.get("model") or "")
        models.append(model)
        if model == "bad/model":
            raise RuntimeError("model unavailable")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"entities":[],"relations":[]}'}}]},
        )

    monkeypatch.setattr(mod.httpx, "post", _fake_post)

    ext = LLMExtractor()
    out = ext.extract("simple text")
    assert out == {"entities": [], "relations": []}
    assert models[:2] == ["bad/model", "good/model"]
