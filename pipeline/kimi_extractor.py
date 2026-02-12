from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

_ALLOWED_ENTITY_TYPES = {"Crop", "Disease", "Pest", "Environment", "Practice", "Condition", "Category"}
_ALLOWED_REL_TYPES = {"CAUSES", "TREATED_BY", "REQUIRES", "SUSCEPTIBLE_TO", "AFFECTS", "MENTIONS", "PART_OF"}
_CID_RE = re.compile(r"^[a-z0-9_\-:.]{3,120}$")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.IGNORECASE | re.DOTALL)


@dataclass
class ExtractionInput:
    text: str
    modality: str = "text"
    table_html_ref: str | None = None
    image_b64_ref: str | None = None
    formula_latex_ref: str | None = None
    asset_ref: str | None = None
    source_doc: str | None = None


class MultiLLMExtractor:
    """OpenAI-compatible structured extractor with multi-model fallback.

    Default runtime is Featherless API, but any OpenAI-compatible endpoint works.
    """

    def __init__(self) -> None:
        self.base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or os.getenv("EXTRACTOR_BASE_URL")
            or os.getenv("FEATHERLESS_BASE_URL")
            or os.getenv("KIMI_API_BASE")
            or "https://api.featherless.ai/v1"
        ).rstrip("/")
        self.api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_COMPAT_API_KEY")
            or os.getenv("EXTRACTOR_API_KEY")
            or os.getenv("FEATHERLESS_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("KIMI_API_KEY")
            or ""
        ).strip()

        # Recommended high-quality default from Featherless model catalog.
        self.primary_model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_COMPAT_MODEL")
            or os.getenv("EXTRACTOR_MODEL")
            or os.getenv("FEATHERLESS_MODEL")
            or os.getenv("KIMI_MODEL")
            or "Qwen/Qwen2.5-32B-Instruct"
        ).strip()
        self.model_candidates = self._build_model_candidates(
            os.getenv("EXTRACTOR_MODEL_CANDIDATES")
            or os.getenv("FEATHERLESS_MODEL_CANDIDATES")
            or ""
        )
        self.timeout = float(os.getenv("EXTRACTOR_TIMEOUT", "45") or "45")
        self.max_tokens = int(os.getenv("EXTRACTOR_MAX_TOKENS", "1024") or "1024")
        self.temperature = float(os.getenv("EXTRACTOR_TEMPERATURE", "0.0") or "0.0")
        self.require_success = str(os.getenv("EXTRACTOR_REQUIRE_SUCCESS", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.http_referer = (os.getenv("EXTRACTOR_HTTP_REFERER") or "").strip()
        self.http_title = (os.getenv("EXTRACTOR_HTTP_TITLE") or "").strip()

    def _build_model_candidates(self, raw: str) -> List[str]:
        # Ordered fallback chain: primary -> explicit candidates -> safe defaults.
        out: List[str] = []
        for item in [self.primary_model] + [x.strip() for x in str(raw or "").split(",") if x.strip()]:
            if item and item not in out:
                out.append(item)
        for item in [
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]:
            if item not in out:
                out.append(item)
        return out

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {"entities": [], "relations": []}
        try:
            return json.loads(text)
        except Exception:
            pass

        m = _JSON_FENCE_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        lb = text.find("{")
        rb = text.rfind("}")
        if 0 <= lb < rb:
            try:
                return json.loads(text[lb : rb + 1])
            except Exception:
                pass
        return {"entities": [], "relations": []}

    def _normalize_canonical_id(self, value: str) -> str:
        cid = str(value or "").strip().lower()
        cid = re.sub(r"\s+", "_", cid)
        cid = re.sub(r"[^a-z0-9_\-:.]", "", cid)
        if not cid:
            return ""
        if _CID_RE.match(cid):
            return cid
        return ""

    def _validate_entities(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed_lower = {x.lower(): x for x in _ALLOWED_ENTITY_TYPES}
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            etype_raw = str(it.get("type") or "").strip()
            etype = allowed_lower.get(etype_raw.lower(), "")
            if not etype:
                continue
            text = str(it.get("text") or "").strip()
            cid = self._normalize_canonical_id(str(it.get("canonical_id") or ""))
            if not text or not cid:
                continue
            conf = float(it.get("confidence", 0.0) or 0.0)
            conf = max(0.0, min(1.0, conf))
            out.append(
                {
                    "text": text,
                    "type": etype,
                    "canonical_id": cid,
                    "confidence": conf,
                }
            )
        return out

    def _validate_relations(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            rtype = str(it.get("type") or "").strip().upper()
            if rtype not in _ALLOWED_REL_TYPES:
                continue
            src = self._normalize_canonical_id(str(it.get("source") or ""))
            tgt = self._normalize_canonical_id(str(it.get("target") or ""))
            if not src or not tgt:
                continue
            conf = float(it.get("confidence", 0.0) or 0.0)
            conf = max(0.0, min(1.0, conf))
            evidence = str(it.get("evidence") or "").strip()
            out.append(
                {
                    "source": src,
                    "target": tgt,
                    "type": rtype,
                    "confidence": conf,
                    "evidence": evidence,
                }
            )
        return out

    def _normalize_input(self, data: str | Dict[str, Any] | ExtractionInput) -> ExtractionInput:
        if isinstance(data, ExtractionInput):
            return data
        if isinstance(data, dict):
            return ExtractionInput(
                text=str(data.get("text") or ""),
                modality=str(data.get("modality") or "text"),
                table_html_ref=(str(data.get("table_html_ref")) if data.get("table_html_ref") is not None else None),
                image_b64_ref=(str(data.get("image_b64_ref")) if data.get("image_b64_ref") is not None else None),
                formula_latex_ref=(str(data.get("formula_latex_ref")) if data.get("formula_latex_ref") is not None else None),
                asset_ref=(str(data.get("asset_ref")) if data.get("asset_ref") is not None else None),
                source_doc=(str(data.get("source_doc")) if data.get("source_doc") is not None else None),
            )
        return ExtractionInput(text=str(data or ""))

    def _build_prompt(self, inp: ExtractionInput) -> str:
        modality = str(inp.modality or "text").strip().lower()
        focus = "text"
        extra_context: List[str] = []
        if modality == "table":
            focus = "table"
            if inp.table_html_ref:
                extra_context.append(f"TABLE_HTML:\n{inp.table_html_ref}")
        elif modality == "image":
            focus = "image caption/meta"
            if inp.asset_ref:
                extra_context.append(f"IMAGE_ASSET_REF: {inp.asset_ref}")
            if inp.image_b64_ref:
                # Keep prompt bounded for large base64 payloads.
                b64 = inp.image_b64_ref[:1024]
                if len(inp.image_b64_ref) > 1024:
                    b64 += "...(truncated)"
                extra_context.append(f"IMAGE_B64_REF:\n{b64}")
        elif modality == "formula":
            focus = "formula/latex"
            if inp.formula_latex_ref:
                extra_context.append(f"FORMULA_LATEX:\n{inp.formula_latex_ref}")

        if inp.source_doc:
            extra_context.append(f"SOURCE_DOC: {inp.source_doc}")

        extra = "\n\n".join(extra_context).strip()
        if extra:
            extra = "\n\n" + extra

        return (
            "You are an agriculture knowledge extractor. "
            "Return ONLY JSON with keys: entities, relations. "
            "Allowed entity types: Crop, Disease, Pest, Environment, Practice, Condition, Category. "
            "Allowed relation types: CAUSES, TREATED_BY, REQUIRES, SUSCEPTIBLE_TO, AFFECTS, MENTIONS, PART_OF. "
            "Entity fields: text,type,canonical_id,confidence,scientific_name,aliases,symptom_keywords,growth_stage,metric,unit. "
            "Relation fields: source,target,type,confidence,evidence. "
            f"Focus on {focus} evidence when present.\n\n"
            f"INPUT:\n{inp.text}{extra}"
        )

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.http_title:
            headers["X-Title"] = self.http_title
        return headers

    def _call_model(self, *, model: str, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=httpx.Timeout(self.timeout, connect=8.0),
        )
        r.raise_for_status()
        msg = ((r.json() or {}).get("choices") or [{}])[0].get("message", {})
        return self._parse_json(str(msg.get("content") or ""))

    def _fallback_result(self) -> Dict[str, Any]:
        return {"entities": [], "relations": []}

    def extract(self, data: str | Dict[str, Any] | ExtractionInput) -> Dict[str, Any]:
        """Extract entities/relations with strict schema validation.

        Validation policy:
        - parse failure -> empty extraction
        - unsupported entity/relation types -> filtered
        - invalid canonical_id -> dropped
        """
        if httpx is None:
            return self._fallback_result()

        inp = self._normalize_input(data)
        if not str(inp.text or "").strip():
            return self._fallback_result()

        prompt = self._build_prompt(inp)
        parsed = self._fallback_result()
        last_error: Exception | None = None
        for model in self.model_candidates:
            try:
                parsed = self._call_model(model=model, prompt=prompt)
                break
            except Exception as exc:
                last_error = exc
                continue

        if parsed == self._fallback_result() and self.require_success and last_error is not None:
            raise RuntimeError(f"extractor call failed across all models: {last_error}") from last_error

        entities_raw = parsed.get("entities") if isinstance(parsed.get("entities"), list) else []
        rels_raw = parsed.get("relations") if isinstance(parsed.get("relations"), list) else []

        entities = self._validate_entities(entities_raw)
        relations = self._validate_relations(rels_raw)
        return {"entities": entities, "relations": relations}


# Backward compatible alias for existing imports.
KimiExtractor = MultiLLMExtractor
