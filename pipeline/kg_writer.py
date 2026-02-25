from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Sequence

try:
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore[assignment]

from pipeline.ontology import ALLOWED_ENTITY_LABELS as _ALLOWED_ENTITY_LABELS
_REL_TYPE_SANITIZER = re.compile(r"[^A-Z_]")
_CANONICAL_SANITIZER = re.compile(r"[^a-z0-9_\-:.]+")


class KGWriter:
    def __init__(self, host: str = "localhost", port: int = 6379, graph: str = "smartfarm") -> None:
        self.graph = graph
        self.client = None
        if redis is not None:
            self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def _run(self, q: str) -> None:
        if self.client is None:
            return
        self.client.execute_command("GRAPH.QUERY", self.graph, q, "--compact")

    def _escape(self, value: Any) -> str:
        return str(value or "").replace("\\", "\\\\").replace("'", "\\'")

    def _norm_entity_label(self, entity_type: str) -> str:
        t = str(entity_type or "").strip()
        return t if t in _ALLOWED_ENTITY_LABELS else "Condition"

    def _norm_rel_type(self, rel_type: str) -> str:
        raw = str(rel_type or "MENTIONS").strip().upper()
        cleaned = _REL_TYPE_SANITIZER.sub("", raw)
        return cleaned or "MENTIONS"

    def _norm_canonical_id(self, value: str) -> str:
        cid = str(value or "").strip().lower()
        cid = re.sub(r"\s+", "_", cid)
        cid = _CANONICAL_SANITIZER.sub("", cid)
        return cid[:120]

    def _merge_entity_node(self, *, canonical_id: str, label: str, tier: str, farm_id: str = "") -> str:
        ecid = self._escape(canonical_id)
        etier = self._escape(tier)
        label_safe = self._norm_entity_label(label)
        if str(tier) == "private":
            efarm = self._escape(farm_id)
            return f"MERGE (e:Entity:{label_safe} {{canonical_id:'{ecid}', tier:'{etier}', farm_id:'{efarm}'}})"
        return f"MERGE (e:Entity:{label_safe} {{canonical_id:'{ecid}', tier:'{etier}'}})"

    def write_chunk(
        self,
        *,
        chunk_id: str,
        text: str,
        metadata: Dict[str, str],
        tier: str = "public",
        farm_id: str = "",
    ) -> None:
        source_type = metadata.get("source_type", "document")
        created_at = metadata.get("created_at", "")
        escaped_chunk_id = self._escape(chunk_id)
        escaped_tier = self._escape(tier)
        escaped_text = self._escape(text)
        escaped_source_type = self._escape(source_type)
        escaped_created_at = self._escape(created_at)
        escaped_farm = self._escape(farm_id)
        escaped_source_doc = self._escape(str(metadata.get("source_doc") or ""))

        if str(tier) == "private":
            q = (
                "MERGE (c:Chunk {chunk_id:'%s', tier:'%s', farm_id:'%s'}) "
                "SET c.text='%s', c.source_type='%s', c.created_at='%s', c.updated_at=datetime()"
                % (
                    escaped_chunk_id,
                    escaped_tier,
                    escaped_farm,
                    escaped_text,
                    escaped_source_type,
                    escaped_created_at,
                )
            )
        else:
            q = (
                "MERGE (c:Chunk {chunk_id:'%s', tier:'%s'}) "
                "SET c.text='%s', c.source_type='%s', c.created_at='%s', c.updated_at=datetime()"
                % (
                    escaped_chunk_id,
                    escaped_tier,
                    escaped_text,
                    escaped_source_type,
                    escaped_created_at,
                )
            )
        self._run(q)

        if escaped_source_doc:
            if str(tier) == "private":
                q_doc = (
                    "MERGE (d:Document {doc_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "SET d.source_doc='%s', d.created_at='%s', d.updated_at=datetime() "
                    "WITH d "
                    "MATCH (c:Chunk {chunk_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "MERGE (d)-[h:HAS_CHUNK {tier:'%s', farm_id:'%s'}]->(c) "
                    "SET h.created_at='%s'"
                    % (
                        escaped_source_doc,
                        escaped_tier,
                        escaped_farm,
                        escaped_source_doc,
                        escaped_created_at,
                        escaped_chunk_id,
                        escaped_tier,
                        escaped_farm,
                        escaped_tier,
                        escaped_farm,
                        escaped_created_at,
                    )
                )
            else:
                q_doc = (
                    "MERGE (d:Document {doc_id:'%s', tier:'%s'}) "
                    "SET d.source_doc='%s', d.created_at='%s', d.updated_at=datetime() "
                    "WITH d "
                    "MATCH (c:Chunk {chunk_id:'%s', tier:'%s'}) "
                    "MERGE (d)-[h:HAS_CHUNK {tier:'%s'}]->(c) "
                    "SET h.created_at='%s'"
                    % (
                        escaped_source_doc,
                        escaped_tier,
                        escaped_source_doc,
                        escaped_created_at,
                        escaped_chunk_id,
                        escaped_tier,
                        escaped_tier,
                        escaped_created_at,
                    )
                )
            self._run(q_doc)

    def write_entities(
        self,
        entities: Sequence[Dict[str, Any]],
        *,
        tier: str = "public",
        farm_id: str = "",
        created_at: str = "",
        chunk_id: str = "",
    ) -> None:
        etier = self._escape(tier)
        efarm = self._escape(farm_id)
        ecreated = self._escape(created_at)
        echunk = self._escape(chunk_id)

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            cid_raw = str(entity.get("canonical_id") or entity.get("text") or "")
            cid = self._norm_canonical_id(cid_raw)
            if not cid:
                continue

            label = self._norm_entity_label(str(entity.get("type") or "Condition"))
            name = self._escape(str(entity.get("text") or cid))
            confidence = float(entity.get("confidence") or 0.5)
            confidence = max(0.0, min(1.0, confidence))
            scientific_name = self._escape(str(entity.get("scientific_name") or ""))
            growth_stage = self._escape(str(entity.get("growth_stage") or ""))
            metric = self._escape(str(entity.get("metric") or ""))
            unit = self._escape(str(entity.get("unit") or ""))
            aliases = entity.get("aliases")
            symptom_keywords = entity.get("symptom_keywords")
            aliases_json = self._escape(json.dumps(aliases, ensure_ascii=False)) if isinstance(aliases, list) else ""
            symptom_json = (
                self._escape(json.dumps(symptom_keywords, ensure_ascii=False))
                if isinstance(symptom_keywords, list)
                else ""
            )

            head = self._merge_entity_node(canonical_id=cid, label=label, tier=tier, farm_id=farm_id)
            q = (
                f"{head} "
                "SET e.name='%s', e.type='%s', e.confidence=%.4f, e.created_at='%s', e.updated_at=datetime()"
                % (name, self._escape(label), confidence, ecreated)
            )
            if scientific_name:
                q += ", e.scientific_name='%s'" % scientific_name
            if growth_stage:
                q += ", e.growth_stage='%s'" % growth_stage
            if metric:
                q += ", e.metric='%s'" % metric
            if unit:
                q += ", e.unit='%s'" % unit
            if aliases_json:
                q += ", e.aliases_json='%s'" % aliases_json
            if symptom_json:
                q += ", e.symptom_keywords_json='%s'" % symptom_json
            if chunk_id:
                if str(tier) == "private":
                    q += (
                        " WITH e "
                        "MATCH (c:Chunk {chunk_id:'%s', tier:'%s', farm_id:'%s'}) "
                        "MERGE (c)-[m:MENTIONS {tier:'%s', farm_id:'%s'}]->(e) "
                        "SET m.confidence=%.4f, m.evidence='%s', m.created_at='%s'"
                        % (
                            echunk,
                            etier,
                            efarm,
                            etier,
                            efarm,
                            confidence,
                            self._escape("entity_extraction"),
                            ecreated,
                        )
                    )
                else:
                    q += (
                        " WITH e "
                        "MATCH (c:Chunk {chunk_id:'%s', tier:'%s'}) "
                        "MERGE (c)-[m:MENTIONS {tier:'%s'}]->(e) "
                        "SET m.confidence=%.4f, m.evidence='%s', m.created_at='%s'"
                        % (
                            echunk,
                            etier,
                            etier,
                            confidence,
                            self._escape("entity_extraction"),
                            ecreated,
                        )
                    )
            self._run(q)

    def write_relations(
        self,
        relations: Iterable[Dict[str, Any]],
        *,
        tier: str = "public",
        farm_id: str = "",
        created_at: str = "",
    ) -> None:
        escaped_tier = self._escape(tier)
        escaped_farm = self._escape(farm_id)
        escaped_created = self._escape(created_at)

        for rel in relations:
            src = self._norm_canonical_id(str(rel.get("source") or ""))
            tgt = self._norm_canonical_id(str(rel.get("target") or ""))
            rtype = self._norm_rel_type(str(rel.get("type") or "MENTIONS"))
            if not src or not tgt:
                continue

            src_label = self._norm_entity_label(str(rel.get("source_type") or "Condition"))
            tgt_label = self._norm_entity_label(str(rel.get("target_type") or "Condition"))
            src_name = self._escape(str(rel.get("source_text") or ""))
            tgt_name = self._escape(str(rel.get("target_text") or ""))

            confidence = float(rel.get("confidence") or 0.5)
            confidence = max(0.0, min(1.0, confidence))
            evidence = self._escape(str(rel.get("evidence") or ""))

            escaped_src = self._escape(src)
            escaped_tgt = self._escape(tgt)
            if str(tier) == "private":
                q = (
                    "MERGE (s:Entity:%s {canonical_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "MERGE (t:Entity:%s {canonical_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "MERGE (s)-[r:%s {tier:'%s', farm_id:'%s'}]->(t) "
                    "SET s.type='%s', t.type='%s', r.confidence=%.4f, r.evidence='%s', r.created_at='%s'"
                    % (
                        src_label,
                        escaped_src,
                        escaped_tier,
                        escaped_farm,
                        tgt_label,
                        escaped_tgt,
                        escaped_tier,
                        escaped_farm,
                        rtype,
                        escaped_tier,
                        escaped_farm,
                        self._escape(src_label),
                        self._escape(tgt_label),
                        confidence,
                        evidence,
                        escaped_created,
                    )
                )
            else:
                q = (
                    "MERGE (s:Entity:%s {canonical_id:'%s', tier:'%s'}) "
                    "MERGE (t:Entity:%s {canonical_id:'%s', tier:'%s'}) "
                    "MERGE (s)-[r:%s {tier:'%s'}]->(t) "
                    "SET s.type='%s', t.type='%s', r.confidence=%.4f, r.evidence='%s', r.created_at='%s'"
                    % (
                        src_label,
                        escaped_src,
                        escaped_tier,
                        tgt_label,
                        escaped_tgt,
                        escaped_tier,
                        rtype,
                        escaped_tier,
                        self._escape(src_label),
                        self._escape(tgt_label),
                        confidence,
                        evidence,
                        escaped_created,
                    )
                )
            if src_name:
                q += ", s.name='%s'" % src_name
            if tgt_name:
                q += ", t.name='%s'" % tgt_name
            self._run(q)
