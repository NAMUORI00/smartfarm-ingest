"""Ontology Builder — auto-generates ``domain_ontology.json`` from ingest artifacts.

After the public ingest pipeline processes all documents, this module
aggregates the entity types, relation types, and textual keywords
discovered by the LLM extractor into a JSON manifest.  The resulting
file is the **Single Source of Truth** consumed by both the
``smartfarm-search`` runtime and the ``smartfarm-ingest`` pipeline on
subsequent runs.

Workflow:
    1. During ingest, ``run_public_ingest`` accumulates raw extraction
       results (entity types, relation types, entity texts).
    2. At the end of ingest, ``build_ontology_from_extractions`` merges
       these observations with any existing baseline ontology.
    3. The merged ontology is written to
       ``smartfarm-search/data/artifacts/domain_ontology.json``.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set


# ── Korean/English relation hint extraction ──────────────────────────
_KO_RELATION_RE = re.compile(
    r"(원인|관계|연관|비교|왜|이유|어떻게|영향|방지|예방|발생|"
    r"때문|결과|유발|억제|촉진|증가|감소|악화|개선)",
)
_EN_RELATION_RE = re.compile(
    r"\b(why|cause|because|relation|compare|versus|affect|prevent|how|"
    r"increase|decrease|result|trigger|inhibit|promote|worsen|improve)\b",
    re.IGNORECASE,
)


def _resolve_output_path() -> Path:
    env = os.getenv("DOMAIN_ONTOLOGY_PATH", "")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parents[1]
    search_root = here.parent / "smartfarm-search"
    return search_root / "data" / "artifacts" / "domain_ontology.json"


def _load_existing(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_relation_keywords(texts: List[str]) -> Dict[str, List[str]]:
    """Mine relation hint keywords from raw chunk texts."""
    ko_counter: Counter[str] = Counter()
    en_counter: Counter[str] = Counter()
    for text in texts:
        ko_counter.update(m.group(0) for m in _KO_RELATION_RE.finditer(text))
        en_counter.update(m.group(0).lower() for m in _EN_RELATION_RE.finditer(text.lower()))
    # Keep keywords that appeared at least once, sorted by frequency
    ko = [k for k, _ in ko_counter.most_common()]
    en = [k for k, _ in en_counter.most_common()]
    return {"ko": ko, "en": en}


def build_ontology_from_extractions(
    *,
    observed_entity_types: Set[str],
    observed_relation_types: Set[str],
    chunk_texts: List[str],
    sensor_thresholds: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    """Build a merged domain ontology manifest from ingest observations.

    Merges discovered types with the existing baseline ontology so that
    previously known types are never lost.

    Returns the full ontology dict (also written to disk).
    """
    out_path = _resolve_output_path()
    existing = _load_existing(out_path)

    # ── merge entity labels ──────────────────────────────────────────
    baseline_entities: Set[str] = set(existing.get("allowed_entity_labels") or [])
    merged_entities = sorted(baseline_entities | observed_entity_types)

    # ── merge relation types ─────────────────────────────────────────
    baseline_relations: Set[str] = set(existing.get("allowed_relation_types") or [])
    merged_relations = sorted(baseline_relations | observed_relation_types)

    # ── extract relation hint keywords from texts ────────────────────
    mined_keywords = _extract_relation_keywords(chunk_texts)
    baseline_hints = existing.get("relation_hint_tokens") or {}
    merged_ko = sorted(set(baseline_hints.get("ko", [])) | set(mined_keywords["ko"]))
    merged_en = sorted(set(baseline_hints.get("en", [])) | set(mined_keywords["en"]))

    # ── sensor thresholds (preserve existing, overlay new) ───────────
    merged_sensors: Dict[str, Dict[str, float]] = dict(existing.get("sensor_thresholds") or {})
    if sensor_thresholds:
        merged_sensors.update(sensor_thresholds)

    # ── modalities (preserve existing) ───────────────────────────────
    baseline_modalities = existing.get("allowed_modalities") or ["text", "table", "image", "formula"]

    ontology: Dict[str, Any] = {
        "schema_version": "ontology_v1",
        "domain": existing.get("domain", "smartfarm"),
        "languages": existing.get("languages", ["ko", "en"]),
        "relation_hint_tokens": {"ko": merged_ko, "en": merged_en},
        "allowed_entity_labels": merged_entities,
        "allowed_relation_types": merged_relations,
        "allowed_modalities": list(baseline_modalities),
        "sensor_thresholds": merged_sensors,
        "_generation_meta": {
            "observed_entity_count": len(observed_entity_types),
            "observed_relation_count": len(observed_relation_types),
            "mined_ko_keywords": len(mined_keywords["ko"]),
            "mined_en_keywords": len(mined_keywords["en"]),
            "chunk_texts_analyzed": len(chunk_texts),
        },
    }

    # ── write to disk ────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(ontology, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ontology
