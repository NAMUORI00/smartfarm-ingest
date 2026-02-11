from __future__ import annotations

from typing import Dict, Iterable

try:
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore[assignment]


class KGWriter:
    def __init__(self, host: str = "localhost", port: int = 6379, graph: str = "smartfarm_v2") -> None:
        self.graph = graph
        self.client = None
        if redis is not None:
            self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def _run(self, q: str) -> None:
        if self.client is None:
            return
        self.client.execute_command("GRAPH.QUERY", self.graph, q, "--compact")

    def write_chunk(self, *, chunk_id: str, text: str, metadata: Dict[str, str], tier: str = "public") -> None:
        source_type = metadata.get("source_type", "document")
        created_at = metadata.get("created_at", "")
        q = (
            "MERGE (c:Chunk {chunk_id:'%s', tier:'%s'}) "
            "SET c.text='%s', c.source_type='%s', c.created_at='%s'"
            % (chunk_id.replace("'", "\\'"), tier, text.replace("'", "\\'"), source_type, created_at)
        )
        self._run(q)

    def write_relations(self, relations: Iterable[Dict[str, str]], tier: str = "public") -> None:
        for rel in relations:
            src = str(rel.get("source") or "")
            tgt = str(rel.get("target") or "")
            rtype = str(rel.get("type") or "RELATED")
            if not src or not tgt:
                continue
            q = (
                "MERGE (s:Entity {canonical_id:'%s', tier:'%s'}) "
                "MERGE (t:Entity {canonical_id:'%s', tier:'%s'}) "
                "MERGE (s)-[:%s {tier:'%s'}]->(t)"
                % (
                    src.replace("'", "\\'"),
                    tier,
                    tgt.replace("'", "\\'"),
                    tier,
                    rtype,
                    tier,
                )
            )
            self._run(q)
