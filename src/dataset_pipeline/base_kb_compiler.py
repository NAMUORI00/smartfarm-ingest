from __future__ import annotations

"""
Base KB compiler: public corpus -> `base.sqlite` (SoT).

Design intent (Asymmetric RAG):
- Public knowledge is compiled in an internet-connected environment (SOTA LLM allowed).
- On-prem runtime consumes only the compiled bundle; private/overlay data is handled separately.

Implementation choices are aligned with LightRAG-style ingestion defaults:
- Token chunking: `chunk_token_size=1200`, `chunk_token_overlap=100` (tiktoken)
- Entity/relation "gleaning": `entity_extract_max_gleaning=1` (1 refinement pass)

We intentionally avoid importing SmartFarm runtime `core.Config.Settings` here to keep the
compiler independent from on-prem runtime side effects (e.g., LLMLITE host guards, dir creation).
"""

import hashlib
import json
import sqlite3
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .corpus_io import read_jsonl


class LLMGenerate(Protocol):
    def generate(
        self,
        prompt: str,
        role: str = "generator",
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str: ...


_CANONICAL_ID_RE = r"^[a-z][a-z0-9_]{0,63}$"
_DEFAULT_TIKTOKEN_FALLBACK_ENCODING = "cl100k_base"


_GLEAN_PROMPT = """You previously extracted entities and causal relations from the following document.
Your task is to **add missing entities/relations** that are not already included.

## IMPORTANT
- Return JSON only (no markdown, no extra text).
- Use the SAME schema: {{"entities":[...], "relations":[...]}}
- Do NOT repeat items already present in the existing extraction.

## EXISTING EXTRACTION (JSON)
{existing_json}

## DOCUMENT
{document_text}

## ADDITIONAL EXTRACTION (JSON only):"""


_FALLBACK_EXTRACTION_PROMPT = """Extract agricultural entities and causal relations from the document.
Return JSON only with this schema:
{{
  "entities": [{{"text": "...", "type": "...", "canonical_id": "...", "confidence": 0.0}}],
  "relations": [{{"source": "...", "target": "...", "type": "...", "confidence": 0.0, "evidence": "..."}}]
}}

Allowed entity types:
- crop, disease, pest, environment, practice, condition, category
- cause, symptom, action

Allowed relation types:
- causes, mitigates, related, associated_with, indicates, affects, requires

Document:
{document_text}
"""


class _FallbackEntityType(str, Enum):
    CROP = "crop"
    DISEASE = "disease"
    PEST = "pest"
    ENVIRONMENT = "environment"
    PRACTICE = "practice"
    CONDITION = "condition"
    CATEGORY = "category"
    CAUSE = "cause"
    SYMPTOM = "symptom"
    ACTION = "action"


class _FallbackRelationType(str, Enum):
    CAUSES = "causes"
    MITIGATES = "mitigates"
    RELATED = "related"
    ASSOCIATED_WITH = "associated_with"
    INDICATES = "indicates"
    AFFECTS = "affects"
    REQUIRES = "requires"


def _load_extraction_contract() -> tuple[str, set[str], set[str]]:
    """
    Keep backward compatibility with legacy CausalSchema if present.
    For rebuilt runtime, use local fallback contract so compiler is standalone.
    """
    try:
        from core.Models.Schemas.CausalSchema import EXTRACTION_PROMPT, EntityType, RelationType  # type: ignore

        return (
            EXTRACTION_PROMPT,
            {e.value for e in EntityType},
            {r.value for r in RelationType},
        )
    except Exception:
        return (
            _FALLBACK_EXTRACTION_PROMPT,
            {e.value for e in _FallbackEntityType},
            {r.value for r in _FallbackRelationType},
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_canonical_id(raw: str) -> str:
    """
    Best-effort canonical_id normalization to lower_snake_case.

    Must match SmartFarm runtime normalization rule:
    - lowercase a-z, digits, underscores
    - starts with [a-z]
    - max length 64
    """
    import re

    t = (raw or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"[\s\-]+", "_", t)
    t = re.sub(r"[^a-z0-9_]+", "", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        return ""
    if not t[0].isalpha():
        t = f"id_{t}"
    if not re.fullmatch(_CANONICAL_ID_RE, t):
        return ""
    return t


def _pick_text(row: Dict[str, Any]) -> str:
    for k in ("text", "text_ko", "text_en", "content", "document", "body"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_json_dict(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty LLM response")
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("LLM JSON root is not an object")
        return obj
    except Exception:
        # Try extracting JSON from fenced code block
        import re

        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group(1))
            if not isinstance(obj, dict):
                raise ValueError("LLM JSON root is not an object (code block)")
            return obj

        # Try to extract the first complete {...} object using brace balancing
        start = raw.find("{")
        if start < 0:
            raise ValueError("no JSON object start found")
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = raw[start : i + 1]
                        obj = json.loads(cand)
                        if not isinstance(obj, dict):
                            raise ValueError("LLM JSON root is not an object (balanced)")
                        return obj
        raise ValueError("incomplete JSON object in response")


def _validate_extraction_shape(obj: Dict[str, Any]) -> None:
    ents = obj.get("entities")
    rels = obj.get("relations")
    if not isinstance(ents, list) or not isinstance(rels, list):
        raise ValueError("extraction JSON must contain 'entities' and 'relations' as lists")


def _norm_surface(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _token_chunks(
    text: str,
    *,
    chunk_token_size: int,
    chunk_token_overlap: int,
    tokenizer_model: str,
) -> List[str]:
    """
    Token-based chunking inspired by LightRAG (chunk_token_size/overlap).

    Falls back to a simple character-based splitter if tiktoken is unavailable.
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model(str(tokenizer_model or "").strip() or "gpt-4o-mini")
        except Exception:
            enc = tiktoken.get_encoding(_DEFAULT_TIKTOKEN_FALLBACK_ENCODING)

        toks = enc.encode(text)
        if not toks:
            return []
        size = max(1, int(chunk_token_size))
        overlap = max(0, int(chunk_token_overlap))

        out: List[str] = []
        start = 0
        while start < len(toks):
            end = min(len(toks), start + size)
            chunk = enc.decode(toks[start:end]).strip()
            if chunk:
                out.append(chunk)
            if end >= len(toks):
                break
            start = max(0, end - overlap)
        return out
    except Exception:
        # Fallback: rough char-based chunking (keeps behavior reasonable without extra deps).
        # Approx: 1 token ~= 4 chars (English). For mixed ko/en, this is a heuristic.
        size_chars = max(200, int(chunk_token_size) * 4)
        overlap_chars = max(0, int(chunk_token_overlap) * 4)
        out: List[str] = []
        i = 0
        while i < len(text):
            j = min(len(text), i + size_chars)
            chunk = text[i:j].strip()
            if chunk:
                out.append(chunk)
            if j >= len(text):
                break
            i = max(0, j - overlap_chars)
        return out


@dataclass(frozen=True)
class CompileStats:
    input_path: str
    output_sqlite: str
    manifest_path: str
    bundle_path: Optional[str]

    docs: int
    chunks: int
    extractions_ok: int
    extractions_error: int
    entities: int
    relations: int


class BaseKBWriter:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if self.sqlite_path.exists():
            self.sqlite_path.unlink()
        self._conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            # Ensure the output `base.sqlite` is self-contained for distribution.
            # base-sync currently downloads only `base.sqlite` (no -wal/-shm sidecars).
            try:
                self._conn.commit()
            except Exception:
                pass
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass
            try:
                self._conn.execute("PRAGMA journal_mode=DELETE;")
            except Exception:
                pass
            try:
                self._conn.commit()
            except Exception:
                pass
            self._conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        # Use WAL during build for performance, but we checkpoint+switch to DELETE on close.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                source TEXT,
                created_at TEXT,
                text TEXT,
                metadata_json TEXT,
                content_sha256 TEXT,
                sensitivity TEXT DEFAULT 'public',
                owner_id TEXT,
                deleted INTEGER DEFAULT 0
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_deleted ON chunks(deleted);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_sha ON chunks(content_sha256);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_owner_id ON chunks(owner_id);")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extractions (
                chunk_id TEXT PRIMARY KEY,
                content_sha256 TEXT,
                model_id TEXT,
                schema_version TEXT,
                extracted_json TEXT,
                status TEXT,
                created_at TEXT,
                error TEXT
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_extractions_status ON extractions(status);")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                chunk_id TEXT,
                canonical_id TEXT,
                entity_type TEXT,
                surface_text TEXT,
                confidence REAL
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_chunk_id ON entities(chunk_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_canonical_id ON entities(canonical_id);")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                chunk_id TEXT,
                src_canonical_id TEXT,
                tgt_canonical_id TEXT,
                relation_type TEXT,
                confidence REAL,
                evidence_text TEXT
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_chunk_id ON relations(chunk_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src_canonical_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_tgt ON relations(tgt_canonical_id);")
        self._conn.commit()

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        source: str,
        created_at: str,
        text: str,
        metadata: Dict[str, Any],
        content_sha256: str,
    ) -> None:
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO chunks(chunk_id, doc_id, source, created_at, text, metadata_json, content_sha256, sensitivity, owner_id, deleted)
            VALUES(?, ?, ?, ?, ?, ?, ?, 'public', NULL, 0)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id = excluded.doc_id,
                source = excluded.source,
                created_at = excluded.created_at,
                text = excluded.text,
                metadata_json = excluded.metadata_json,
                content_sha256 = excluded.content_sha256,
                sensitivity = excluded.sensitivity,
                owner_id = excluded.owner_id,
                deleted = excluded.deleted;
            """,
            (chunk_id, doc_id, source, created_at, text, meta_json, content_sha256),
        )

    def upsert_extraction(
        self,
        *,
        chunk_id: str,
        content_sha256: str,
        model_id: str,
        schema_version: str,
        extracted_json: str,
        status: str,
        created_at: str,
        error: Optional[str],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO extractions(chunk_id, content_sha256, model_id, schema_version, extracted_json, status, created_at, error)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content_sha256 = excluded.content_sha256,
                model_id = excluded.model_id,
                schema_version = excluded.schema_version,
                extracted_json = excluded.extracted_json,
                status = excluded.status,
                created_at = excluded.created_at,
                error = excluded.error;
            """,
            (chunk_id, content_sha256, model_id, schema_version, extracted_json, status, created_at, error),
        )

    def insert_entities(self, rows: Sequence[Tuple[str, str, str, str, float]]) -> None:
        self._conn.executemany(
            "INSERT INTO entities(chunk_id, canonical_id, entity_type, surface_text, confidence) VALUES(?, ?, ?, ?, ?);",
            list(rows),
        )

    def insert_relations(self, rows: Sequence[Tuple[str, str, str, str, float, str]]) -> None:
        self._conn.executemany(
            "INSERT INTO relations(chunk_id, src_canonical_id, tgt_canonical_id, relation_type, confidence, evidence_text) VALUES(?, ?, ?, ?, ?, ?);",
            list(rows),
        )

    def commit(self) -> None:
        self._conn.commit()


def _iter_docs(input_path: Path, *, limit_docs: Optional[int]) -> Iterable[Dict[str, Any]]:
    for i, row in enumerate(read_jsonl(input_path)):
        if limit_docs is not None and i >= int(limit_docs):
            break
        if isinstance(row, dict):
            yield row


def _extract_entities_relations_from_chunk(
    llm: LLMGenerate,
    *,
    prompt_template: str,
    chunk_text: str,
    llm_role: str,
    system_prompt: str,
    max_tokens: int,
    llm_retries: int,
    entity_extract_max_gleaning: int,
    allowed_entity_types: set[str],
    allowed_relation_types: set[str],
) -> Tuple[Dict[str, Any], List[Tuple[str, str, str, str, float]], List[Tuple[str, str, str, str, float, str]]]:
    def _call_extract(prompt: str) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        tries = max(0, int(llm_retries)) + 1
        for _ in range(tries):
            try:
                # Best-effort: request JSON-only output (OpenAI-compatible servers may support this).
                try:
                    raw = llm.generate(
                        prompt,
                        role=llm_role,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        max_tokens=int(max_tokens),
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    raw = llm.generate(
                        prompt,
                        role=llm_role,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        max_tokens=int(max_tokens),
                    )
                obj = _extract_json_dict(str(raw))
                _validate_extraction_shape(obj)
                return obj
            except Exception as e:
                last_err = e
        raise ValueError(f"LLM extraction failed after {tries} tries: {last_err}")

    prompt = prompt_template.format(document_text=chunk_text)
    parsed = _call_extract(prompt)

    entities: List[Tuple[str, str, str, str, float]] = []
    text_to_cid: Dict[str, str] = {}
    norm_to_cid: Dict[str, str] = {}
    known_cids: set[str] = set()

    seen_ent: set[Tuple[str, str]] = set()
    for ent in parsed.get("entities", []) or []:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text") or "").strip()
        et = str(ent.get("type") or "").strip()
        cid = normalize_canonical_id(str(ent.get("canonical_id") or ""))
        conf = float(ent.get("confidence", 0.8) or 0.0)
        if not surface or not et or not cid:
            continue
        if et not in allowed_entity_types:
            continue
        key = (cid, surface)
        if key in seen_ent:
            continue
        seen_ent.add(key)
        entities.append(("", cid, et, surface, conf))  # chunk_id filled later
        text_to_cid[surface] = cid
        norm_to_cid[_norm_surface(surface)] = cid
        known_cids.add(cid)

    relations: List[Tuple[str, str, str, str, float, str]] = []
    seen_rel: set[Tuple[str, str, str]] = set()
    for rel in parsed.get("relations", []) or []:
        if not isinstance(rel, dict):
            continue
        src_txt = str(rel.get("source") or "").strip()
        tgt_txt = str(rel.get("target") or "").strip()
        rt = str(rel.get("type") or "").strip()
        conf = float(rel.get("confidence", 0.8) or 0.0)
        evidence = str(rel.get("evidence") or rel.get("evidence_text") or "")
        if rt not in allowed_relation_types:
            continue

        src = text_to_cid.get(src_txt) or norm_to_cid.get(_norm_surface(src_txt))
        tgt = text_to_cid.get(tgt_txt) or norm_to_cid.get(_norm_surface(tgt_txt))
        if not src:
            guess = normalize_canonical_id(src_txt)
            if guess and guess in known_cids:
                src = guess
        if not tgt:
            guess = normalize_canonical_id(tgt_txt)
            if guess and guess in known_cids:
                tgt = guess
        if not src or not tgt:
            continue
        k = (src, tgt, rt)
        if k in seen_rel:
            continue
        seen_rel.add(k)
        relations.append(("", src, tgt, rt, conf, evidence))  # chunk_id filled later

    # LightRAG-style gleaning: ask the model for missing items and merge.
    max_glean = max(0, int(entity_extract_max_gleaning))
    if max_glean > 0:
        # Build maps from current parsed lists (use internal rows for merging).
        ent_by_key: Dict[Tuple[str, str], Tuple[str, str, str, str, float]] = {}
        for row in entities:
            _chunk0, cid, et, surface, conf = row
            ent_by_key[(cid, surface)] = row

        rel_by_key: Dict[Tuple[str, str, str], Tuple[str, str, str, str, float, str]] = {}
        for row in relations:
            _chunk0, src, tgt, rt, conf, evidence = row
            rel_by_key[(src, tgt, rt)] = row

        # Keep a compact existing JSON (to avoid huge prompt).
        existing_json = json.dumps(parsed, ensure_ascii=False)
        for _ in range(max_glean):
            glean_prompt = _GLEAN_PROMPT.format(existing_json=existing_json, document_text=chunk_text)
            extra = _call_extract(glean_prompt)
            # Merge by re-parsing with the same filters.
            extra_parsed, extra_entities, extra_relations = extra, [], []
            # Reuse the current parsing logic by temporarily assigning.
            # (We intentionally do a second pass to keep one normalization rule.)
            tmp = extra_parsed

            # Entities
            seen_ent2: set[Tuple[str, str]] = set(ent_by_key.keys())
            for ent in tmp.get("entities", []) or []:
                if not isinstance(ent, dict):
                    continue
                surface = str(ent.get("text") or "").strip()
                et = str(ent.get("type") or "").strip()
                cid = normalize_canonical_id(str(ent.get("canonical_id") or ""))
                conf = float(ent.get("confidence", 0.8) or 0.0)
                if not surface or not et or not cid:
                    continue
                if et not in allowed_entity_types:
                    continue
                key = (cid, surface)
                if key in seen_ent2:
                    continue
                seen_ent2.add(key)
                extra_entities.append(("", cid, et, surface, conf))

            # Update maps used for relation resolution in this glean pass
            glean_text_to_cid: Dict[str, str] = dict(text_to_cid)
            glean_norm_to_cid: Dict[str, str] = dict(norm_to_cid)
            glean_known_cids: set[str] = set(known_cids)
            for _c0, cid, _et, surface, _conf in list(ent_by_key.values()) + extra_entities:
                glean_text_to_cid[surface] = cid
                glean_norm_to_cid[_norm_surface(surface)] = cid
                glean_known_cids.add(cid)

            # Relations
            seen_rel2: set[Tuple[str, str, str]] = set(rel_by_key.keys())
            for rel in tmp.get("relations", []) or []:
                if not isinstance(rel, dict):
                    continue
                src_txt = str(rel.get("source") or "").strip()
                tgt_txt = str(rel.get("target") or "").strip()
                rt = str(rel.get("type") or "").strip()
                conf = float(rel.get("confidence", 0.8) or 0.0)
                evidence = str(rel.get("evidence") or rel.get("evidence_text") or "")
                if not src_txt or not tgt_txt or not rt:
                    continue
                if rt not in allowed_relation_types:
                    continue
                src = glean_text_to_cid.get(src_txt) or glean_norm_to_cid.get(_norm_surface(src_txt))
                tgt = glean_text_to_cid.get(tgt_txt) or glean_norm_to_cid.get(_norm_surface(tgt_txt))
                if not src:
                    guess = normalize_canonical_id(src_txt)
                    if guess and guess in glean_known_cids:
                        src = guess
                if not tgt:
                    guess = normalize_canonical_id(tgt_txt)
                    if guess and guess in glean_known_cids:
                        tgt = guess
                if not src or not tgt:
                    continue
                k = (src, tgt, rt)
                if k in seen_rel2:
                    continue
                seen_rel2.add(k)
                extra_relations.append(("", src, tgt, rt, conf, evidence))

            # Merge
            for row in extra_entities:
                _c0, cid, et, surface, conf = row
                ent_by_key.setdefault((cid, surface), row)
                text_to_cid[surface] = cid
                norm_to_cid[_norm_surface(surface)] = cid
                known_cids.add(cid)
            for row in extra_relations:
                _c0, src, tgt, rt, conf, evidence = row
                rel_by_key.setdefault((src, tgt, rt), row)

            # Update for next pass
            entities = list(ent_by_key.values())
            relations = list(rel_by_key.values())
            # Keep existing_json small-ish: don't grow unbounded.
            parsed = {"entities": [{"text": s, "type": et, "canonical_id": cid, "confidence": conf} for _c0, cid, et, s, conf in entities],
                      "relations": [{"source": src, "target": tgt, "type": rt, "confidence": conf, "evidence": ev} for _c0, src, tgt, rt, conf, ev in relations]}
            existing_json = json.dumps(parsed, ensure_ascii=False)

    return parsed, entities, relations


def compile_base_kb(
    *,
    input_jsonl: Path,
    output_sqlite: Path,
    llm: LLMGenerate,
    llm_role: str = "judge",
    system_prompt: str = "Return JSON only. Do not include any extra text.",
    chunk_token_size: int = 1200,
    chunk_token_overlap: int = 100,
    tokenizer_model: str = "gpt-4o-mini",
    entity_extract_max_gleaning: int = 1,
    schema_version: str = "kb-update-v1",
    max_tokens: int = 700,
    limit_docs: Optional[int] = None,
    limit_chunks: Optional[int] = None,
    bundle_out: Optional[Path] = None,
    llm_retries: int = 2,
    embed_model_id: str = "",
    embedding_dim: int = 0,
) -> CompileStats:
    """
    Compile a public Base KB into `base.sqlite` using a SOTA LLM (public-only).

    Notes:
    - Writes a SQLite SoT compatible with smartfarm-search runtime loader.
    - Does NOT require any on-prem/private overlay data.
    """
    from .bootstrap import ensure_search_on_path

    # We intentionally avoid importing `core.Config.Settings` here.
    # Base compiler runs in an internet-connected environment and should remain
    # independent from on-prem runtime Settings side effects (e.g., LLMLITE_HOST guard).
    ensure_search_on_path()
    extraction_prompt, allowed_entity_types, allowed_relation_types = _load_extraction_contract()

    input_jsonl = Path(input_jsonl)
    output_sqlite = Path(output_sqlite)
    manifest_path = output_sqlite.parent / "manifest.json"
    if bundle_out is not None:
        bundle_out = Path(bundle_out)
        bundle_out.parent.mkdir(parents=True, exist_ok=True)

    writer = BaseKBWriter(output_sqlite)
    docs = 0
    chunks = 0
    ok = 0
    err = 0
    ent_count = 0
    rel_count = 0

    corpus_hasher = hashlib.sha256()
    remaining_chunks = int(limit_chunks) if limit_chunks is not None else None

    try:
        for row in _iter_docs(input_jsonl, limit_docs=limit_docs):
            if remaining_chunks is not None and remaining_chunks <= 0:
                break

            doc_id = str(row.get("id") or f"doc_{docs}").strip()
            if not doc_id:
                doc_id = f"doc_{docs}"
            text = _pick_text(row)
            if not text:
                continue

            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            meta = dict(meta or {})
            source = str(meta.get("source") or row.get("source") or "public_corpus")
            meta.setdefault("source", source)

            chunk_texts = _token_chunks(
                text,
                chunk_token_size=int(chunk_token_size),
                chunk_token_overlap=int(chunk_token_overlap),
                tokenizer_model=str(tokenizer_model),
            )
            if not chunk_texts:
                continue

            docs += 1
            for i, ch in enumerate(chunk_texts):
                if remaining_chunks is not None and remaining_chunks <= 0:
                    break
                chunk_id = f"{doc_id}#c{i}"
                sha = _sha256_text(ch)
                created_at = _utc_now_iso()

                writer.upsert_chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source=source,
                    created_at=created_at,
                    text=ch,
                    metadata=meta,
                    content_sha256=sha,
                )
                corpus_hasher.update((chunk_id + "\n").encode("utf-8"))
                corpus_hasher.update((sha + "\n").encode("utf-8"))
                chunks += 1

                model_id = ""
                try:
                    if hasattr(llm, "judge_config"):
                        model_id = str(getattr(llm, "judge_config").model or "")
                except Exception:
                    model_id = ""
                model_id = model_id or "sota-llm"

                try:
                    parsed, entities, relations = _extract_entities_relations_from_chunk(
                        llm,
                        prompt_template=extraction_prompt,
                        chunk_text=ch,
                        llm_role=llm_role,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        llm_retries=int(llm_retries),
                        entity_extract_max_gleaning=int(entity_extract_max_gleaning),
                        allowed_entity_types=allowed_entity_types,
                        allowed_relation_types=allowed_relation_types,
                    )

                    writer.upsert_extraction(
                        chunk_id=chunk_id,
                        content_sha256=sha,
                        model_id=model_id,
                        schema_version=str(schema_version),
                        extracted_json=json.dumps(parsed, ensure_ascii=False),
                        status="ok",
                        created_at=_utc_now_iso(),
                        error=None,
                    )

                    # Fill chunk_id and insert
                    ent_rows = [(chunk_id, cid, et, surface, conf) for _cid0, cid, et, surface, conf in entities]
                    rel_rows = [(chunk_id, src, tgt, rt, conf, evidence) for _cid0, src, tgt, rt, conf, evidence in relations]
                    if ent_rows:
                        writer.insert_entities(ent_rows)
                        ent_count += len(ent_rows)
                    if rel_rows:
                        writer.insert_relations(rel_rows)
                        rel_count += len(rel_rows)
                    ok += 1
                except Exception as e:
                    writer.upsert_extraction(
                        chunk_id=chunk_id,
                        content_sha256=sha,
                        model_id=model_id,
                        schema_version=str(schema_version),
                        extracted_json="{}",
                        status="error",
                        created_at=_utc_now_iso(),
                        error=f"{type(e).__name__}: {e}",
                    )
                    err += 1

                if remaining_chunks is not None:
                    remaining_chunks -= 1

        writer.commit()
    finally:
        writer.close()

    corpus_sha256 = corpus_hasher.hexdigest()
    artifacts_sha256 = {"base.sqlite": _sha256_file(output_sqlite)}

    # Best-effort embed model hint (runtime will rebuild caches).
    # Keep as a hint only; base compiler should not depend on runtime Settings.
    embed_model_id = str(embed_model_id or "")
    manifest: Dict[str, Any] = {
        "schema_version": "edgekg-v3-base-sqlite",
        "created_at": _utc_now_iso(),
        "embed_model_id": embed_model_id,
        "embedding_dim": int(embedding_dim or 0),
        "corpus_sha256": corpus_sha256,
        "doc_count": int(docs),
        "chunk_count": int(chunks),
        "artifacts_sha256": dict(sorted(artifacts_sha256.items(), key=lambda x: x[0])),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if bundle_out is not None:
        with tarfile.open(bundle_out, "w:gz") as tf:
            tf.add(output_sqlite, arcname="base.sqlite")
            tf.add(manifest_path, arcname="manifest.json")

    return CompileStats(
        input_path=str(input_jsonl),
        output_sqlite=str(output_sqlite),
        manifest_path=str(manifest_path),
        bundle_path=str(bundle_out) if bundle_out is not None else None,
        docs=int(docs),
        chunks=int(chunks),
        extractions_ok=int(ok),
        extractions_error=int(err),
        entities=int(ent_count),
        relations=int(rel_count),
    )
