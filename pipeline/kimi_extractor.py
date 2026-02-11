from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

_ALLOWED_ENTITY_TYPES = {"Crop", "Disease", "Pest", "Environment", "Practice", "Condition", "Category"}
_ALLOWED_REL_TYPES = {"CAUSES", "TREATED_BY", "REQUIRES", "SUSCEPTIBLE_TO", "AFFECTS", "MENTIONS", "PART_OF"}
_CID_RE = re.compile(r"^[a-z0-9_\-:.]{3,120}$")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.IGNORECASE | re.DOTALL)


class KimiExtractor:
    def __init__(self) -> None:
        self.base_url = os.getenv("KIMI_API_BASE", "https://api.moonshot.cn/v1").rstrip("/")
        self.api_key = os.getenv("KIMI_API_KEY", "")
        self.model = os.getenv("KIMI_MODEL", "moonshot-v1-8k")

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
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            etype = str(it.get("type") or "").strip()
            if etype not in _ALLOWED_ENTITY_TYPES:
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
            rtype = str(it.get("type") or "").strip()
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

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities/relations with strict schema validation.

        Validation policy:
        - parse failure -> empty extraction
        - unsupported entity/relation types -> filtered
        - invalid canonical_id -> dropped
        """
        prompt = (
            "You are an agriculture knowledge extractor. "
            "Return ONLY JSON with keys: entities, relations. "
            "Allowed entity types: Crop, Disease, Pest, Environment, Practice, Condition, Category. "
            "Allowed relation types: CAUSES, TREATED_BY, REQUIRES, SUSCEPTIBLE_TO, AFFECTS, MENTIONS, PART_OF. "
            "Entity fields: text,type,canonical_id,confidence. "
            "Relation fields: source,target,type,confidence,evidence.\n\n"
            f"INPUT:\n{text}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        if httpx is None:
            return {"entities": [], "relations": []}

        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
            r.raise_for_status()
            msg = ((r.json() or {}).get("choices") or [{}])[0].get("message", {})
            parsed = self._parse_json(str(msg.get("content") or ""))
        except Exception:
            parsed = {"entities": [], "relations": []}

        entities_raw = parsed.get("entities") if isinstance(parsed.get("entities"), list) else []
        rels_raw = parsed.get("relations") if isinstance(parsed.get("relations"), list) else []

        entities = self._validate_entities(entities_raw)
        relations = self._validate_relations(rels_raw)
        return {"entities": entities, "relations": relations}
