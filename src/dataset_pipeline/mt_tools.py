from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Template

from .json_utils import try_parse_json
from .llm_connector import LLMConnector
from .corpus_text import extract_numbers_and_units


_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _load_prompt(name: str) -> str:
    p = _PROMPTS_DIR / name
    return p.read_text(encoding="utf-8")


@dataclass
class NumbersUnitsCheck:
    source_tokens: list[str]
    target_tokens: list[str]
    missing_in_target: list[str]
    extra_in_target: list[str]
    ok: bool


def check_numbers_units(source_text: str, translated_text: str) -> NumbersUnitsCheck:
    src = extract_numbers_and_units(source_text)
    tgt = extract_numbers_and_units(translated_text)
    missing = [t for t in src if t not in tgt]
    extra = [t for t in tgt if t not in src]
    ok = not missing
    return NumbersUnitsCheck(
        source_tokens=src,
        target_tokens=tgt,
        missing_in_target=missing,
        extra_in_target=extra,
        ok=ok,
    )


class MTTranslator:
    def __init__(self, llm: LLMConnector):
        self.llm = llm
        self._template = Template(_load_prompt("mt_translate.jinja"))

    def translate(
        self,
        text: str,
        *,
        src_lang: str = "en",
        tgt_lang: str = "ko",
        glossary: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        prompt = self._template.render(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            glossary=glossary or {},
            text=text,
        )
        out = self.llm.generate(
            prompt=prompt,
            role="generator",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (out or "").strip()


class MQMJudge:
    def __init__(self, llm: LLMConnector):
        self.llm = llm
        self._template = Template(_load_prompt("mt_mqm_judge.jinja"))

    def score(
        self,
        source_text: str,
        translated_text: str,
        *,
        src_lang: str = "en",
        tgt_lang: str = "ko",
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        prompt = self._template.render(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            source_text=source_text,
            translated_text=translated_text,
        )
        resp = self.llm.generate(
            prompt=prompt,
            role="judge",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        parsed = try_parse_json(resp)
        if isinstance(parsed, dict):
            return parsed
        return {"overall_score": 0, "errors": {}, "summary": "parse_failed", "raw": (resp or "")[:2000]}


class PostEditor:
    def __init__(self, llm: LLMConnector):
        self.llm = llm
        self._template = Template(_load_prompt("mt_postedit.jinja"))

    def post_edit(
        self,
        source_text: str,
        translated_text: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1800,
    ) -> Dict[str, Any]:
        prompt = self._template.render(
            source_text=source_text,
            translated_text=translated_text,
        )
        resp = self.llm.generate(
            prompt=prompt,
            role="judge",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = try_parse_json(resp)
        if isinstance(parsed, dict) and isinstance(parsed.get("post_edited_text"), str):
            return parsed
        return {"post_edited_text": translated_text, "changes": ["parse_failed"], "raw": (resp or "")[:2000]}


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"

