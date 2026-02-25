"""Domain Ontology Loader for smartfarm-ingest pipeline.

Mirrors ``smartfarm-search/core/Config/ontology.py`` but resolves the JSON
path relative to the *ingest* project root.  Falls back to safe built-in
defaults when the manifest file is absent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Set

_FALLBACK_ENTITY_LABELS: Set[str] = {"Crop", "Disease", "Pest", "Environment", "Practice", "Condition", "Category"}
_FALLBACK_RELATION_TYPES: Set[str] = {"CAUSES", "TREATED_BY", "REQUIRES", "SUSCEPTIBLE_TO", "AFFECTS", "MENTIONS", "PART_OF"}


def _resolve_ontology_path() -> Path:
    env = os.getenv("DOMAIN_ONTOLOGY_PATH", "")
    if env:
        return Path(env)
    # ingest sits next to search: ../smartfarm-search/data/artifacts/
    project_root = Path(__file__).resolve().parents[1]
    search_root = project_root.parent / "smartfarm-search"
    return search_root / "data" / "artifacts" / "domain_ontology.json"


def _load_ontology() -> Dict[str, Any]:
    path = _resolve_ontology_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── one-time load at import ──────────────────────────────────────────
_RAW: Dict[str, Any] = _load_ontology()


def _build_set(raw: Dict[str, Any], key: str, fallback: Set[str]) -> Set[str]:
    items = raw.get(key)
    if isinstance(items, list) and items:
        return {str(i) for i in items if i}
    return set(fallback)


# ── public cached sets (O(1) membership test) ────────────────────────
ALLOWED_ENTITY_LABELS: Set[str] = _build_set(_RAW, "allowed_entity_labels", _FALLBACK_ENTITY_LABELS)
ALLOWED_RELATION_TYPES: Set[str] = _build_set(_RAW, "allowed_relation_types", _FALLBACK_RELATION_TYPES)

# convenience for LLM prompt builders
ALLOWED_ENTITY_LABELS_CSV: str = ", ".join(sorted(ALLOWED_ENTITY_LABELS))
ALLOWED_RELATION_TYPES_CSV: str = ", ".join(sorted(ALLOWED_RELATION_TYPES))
