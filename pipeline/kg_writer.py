from __future__ import annotations

from typing import Dict, Iterable

try:
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore[assignment]


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
        escaped_chunk_id = chunk_id.replace("'", "\\'")
        escaped_tier = str(tier).replace("'", "\\'")
        escaped_text = text.replace("'", "\\'")
        escaped_source_type = source_type.replace("'", "\\'")
        escaped_created_at = str(created_at).replace("'", "\\'")
        escaped_farm = str(farm_id or "").replace("'", "\\'")
        if str(tier) == "private":
            q = (
                "MERGE (c:Chunk {chunk_id:'%s', tier:'%s', farm_id:'%s'}) "
                "SET c.text='%s', c.source_type='%s', c.created_at='%s'"
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
                "SET c.text='%s', c.source_type='%s', c.created_at='%s'"
                % (
                    escaped_chunk_id,
                    escaped_tier,
                    escaped_text,
                    escaped_source_type,
                    escaped_created_at,
                )
            )
        self._run(q)

    def write_relations(self, relations: Iterable[Dict[str, str]], tier: str = "public", farm_id: str = "") -> None:
        escaped_tier = str(tier).replace("'", "\\'")
        escaped_farm = str(farm_id or "").replace("'", "\\'")
        for rel in relations:
            src = str(rel.get("source") or "")
            tgt = str(rel.get("target") or "")
            rtype = str(rel.get("type") or "RELATED")
            if not src or not tgt:
                continue
            escaped_src = src.replace("'", "\\'")
            escaped_tgt = tgt.replace("'", "\\'")
            escaped_type = rtype.replace("'", "\\'")
            if str(tier) == "private":
                q = (
                    "MERGE (s:Entity {canonical_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "MERGE (t:Entity {canonical_id:'%s', tier:'%s', farm_id:'%s'}) "
                    "MERGE (s)-[:%s {tier:'%s', farm_id:'%s'}]->(t)"
                    % (
                        escaped_src,
                        escaped_tier,
                        escaped_farm,
                        escaped_tgt,
                        escaped_tier,
                        escaped_farm,
                        escaped_type,
                        escaped_tier,
                        escaped_farm,
                    )
                )
            else:
                q = (
                    "MERGE (s:Entity {canonical_id:'%s', tier:'%s'}) "
                    "MERGE (t:Entity {canonical_id:'%s', tier:'%s'}) "
                    "MERGE (s)-[:%s {tier:'%s'}]->(t)"
                    % (
                        escaped_src,
                        escaped_tier,
                        escaped_tgt,
                        escaped_tier,
                        escaped_type,
                        escaped_tier,
                    )
                )
            self._run(q)
