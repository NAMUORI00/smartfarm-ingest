from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Sequence

try:
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore[assignment]

_ALLOWED_ENTITY_LABELS = {"Crop", "Disease", "Pest", "Environment", "Practice", "Condition", "Category"}
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

            head = self._merge_entity_node(canonical_id=cid, label=label, tier=tier, farm_id=farm_id)
            q = (
                f"{head} "
                "SET e.name='%s', e.type='%s', e.confidence=%.4f, e.created_at='%s', e.updated_at=datetime()"
                % (name, self._escape(label), confidence, ecreated)
            )
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

            confidence = float(rel.get("confidence") or 0.5)
            confidence = max(0.0, min(1.0, confidence))
            evidence = self._escape(str(rel.get("evidence") or ""))

            escaped_src = self._escape(src)
            escaped_tgt = self._escape(tgt)
            if str(tier) == "private":
                q = (
                    "MERGE (s:Entity {canonical_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "MERGE (t:Entity {canonical_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "MERGE (s)-[r:%s {tier:'%s', farm_id:'%s'}]->(t) "
                    "SET r.confidence=%.4f, r.evidence='%s', r.created_at='%s'"
                    % (
                        escaped_src,
                        escaped_tier,
                        escaped_farm,
                        escaped_tgt,
                        escaped_tier,
                        escaped_farm,
                        rtype,
                        escaped_tier,
                        escaped_farm,
                        confidence,
                        evidence,
                        escaped_created,
                    )
                )
            else:
                q = (
                    "MERGE (s:Entity {canonical_id:'%s', tier:'%s'}) "
                    "MERGE (t:Entity {canonical_id:'%s', tier:'%s'}) "
                    "MERGE (s)-[r:%s {tier:'%s'}]->(t) "
                    "SET r.confidence=%.4f, r.evidence='%s', r.created_at='%s'"
                    % (
                        escaped_src,
                        escaped_tier,
                        escaped_tgt,
                        escaped_tier,
                        rtype,
                        escaped_tier,
                        confidence,
                        evidence,
                        escaped_created,
                    )
                )
            self._run(q)
