#!/usr/bin/env python3
"""Build Tri-Graph RAG indices (LLM-free) for on-prem edge deployment.

References (conceptual inspiration; **clean-room implementation**):
  - LinearRAG paper (Tri-Graph: Entity–Sentence–Passage; semantic bridging; global aggregation via PPR):
    https://arxiv.org/abs/2510.10114
  - LinearRAG GitHub (GPL-3.0; do not copy code into this repo — logic-only adoption):
    https://github.com/DEEP-PolyU/LinearRAG

This indexer produces a *single, consistent* chunk-level corpus that is shared by:
  1) Dense index (FAISS)
  2) Sparse index (BM25)
  3) Tri-Graph artifacts (Entity–Sentence–Chunk) for multi-hop retrieval

Design goals:
  - No LLM calls during indexing
  - Permissive-only stack (no GPL graph libs; no igraph)
  - Edge-friendly artifacts (memory-mappable .npy + compact json/npz)

Typical usage (from workspace root):
  python smartfarm-ingest/scripts/indexing/build_trigraph_index.py \
    --input-jsonl smartfarm-ingest/output/wasabi_en_ko_parallel.jsonl \
    --lang ko \
    --embed-model-id minilm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\n。！？])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z가-힣0-9_\-/%\.]{1,48}")
_CHUNK_ID_RE = re.compile(r"^(?P<doc>.+?)#c(?P<idx>\d+)$")


def _iter_jsonl(path: Path, limit: Optional[int] = None) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _pick_text(item: dict, lang: str) -> str:
    if lang == "ko":
        for key in ("text_ko", "text", "content", "document", "body", "text_en"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if lang == "en":
        for key in ("text_en", "text", "content", "document", "body", "text_ko"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    # both: include EN + KO if available
    text_en = item.get("text_en") or item.get("text") or ""
    text_ko = item.get("text_ko") or ""
    text_en = text_en.strip() if isinstance(text_en, str) else ""
    text_ko = text_ko.strip() if isinstance(text_ko, str) else ""
    if text_en and text_ko:
        return f"{text_en}\n\n[한국어]\n{text_ko}"
    return text_ko or text_en


def _sentence_split(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = _SENT_SPLIT_RE.split(t)
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        out.append(s)
    return out


def _window_chunks(sentences: Sequence[str], window: int, stride: int) -> List[List[str]]:
    if window <= 0:
        return []
    stride = max(1, int(stride))
    out: List[List[str]] = []
    i = 0
    while i < len(sentences):
        chunk = [s for s in sentences[i : i + window] if s]
        if chunk:
            out.append(chunk)
        i += stride
    return out


def _normalize_token(tok: str) -> str:
    t = (tok or "").strip()
    if not t:
        return ""
    # Lowercase latin tokens; keep Korean as-is.
    if re.fullmatch(r"[A-Za-z0-9_\-/%\.]+", t):
        t = t.lower()
    return t


def _extract_candidate_tokens(sentence: str, min_len: int) -> List[str]:
    """Open-world token extraction (no domain rules, no ontology filters)."""
    toks = []
    for raw in _TOKEN_RE.findall(sentence or ""):
        tok = _normalize_token(raw)
        if len(tok) < min_len:
            continue
        # Drop purely numeric tokens
        if tok.replace(".", "", 1).isdigit():
            continue
        toks.append(tok)
    return toks


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _save_csr_npz(path: Path, indptr: np.ndarray, indices: np.ndarray, shape: Tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indptr=indptr.astype(np.int64, copy=False),
        indices=indices.astype(np.int64, copy=False),
        shape=np.asarray(shape, dtype=np.int64),
    )


def _build_csr_from_lists(rows: List[List[int]], n_cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build CSR (indptr, indices) for an unweighted adjacency list."""
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indices_list: List[int] = []
    for i, cols in enumerate(rows):
        dedup = sorted(set(int(c) for c in cols if c is not None))
        indices_list.extend(dedup)
        indptr[i + 1] = len(indices_list)
    indices = np.asarray(indices_list, dtype=np.int64)
    return indptr, indices


@dataclass(frozen=True)
class TriGraphMeta:
    embed_model_id: str
    embedding_dim: int
    created_at: str
    corpus_path: str
    corpus_sha256: str
    n_docs: int
    n_chunks: int
    n_sentences: int
    n_entities: int
    chunk_size: int
    chunk_stride: int
    version: str = "trigraph"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Tri-Graph RAG indices (LLM-free).")
    ap.add_argument("--input-jsonl", required=True, type=Path, help="Corpus JSONL (id + text_* fields)")
    ap.add_argument("--lang", default="ko", choices=["ko", "en", "both"], help="Language field preference")
    ap.add_argument("--limit", type=int, default=None, help="Optional doc limit for quick iteration")

    ap.add_argument("--chunk-size", type=int, default=5, help="Sentences per chunk window")
    ap.add_argument("--chunk-stride", type=int, default=2, help="Sentence stride for windowed chunks")

    ap.add_argument("--min-entity-len", type=int, default=2, help="Minimum token length to keep")
    ap.add_argument("--min-df", type=int, default=2, help="Minimum sentence document-frequency for entities")
    ap.add_argument("--max-df-ratio", type=float, default=0.25, help="Drop entities appearing in >ratio of sentences")
    ap.add_argument("--max-entities", type=int, default=50000, help="Cap total entities for edge index size")

    ap.add_argument(
        "--embed-model-id",
        default=os.getenv("EMBED_MODEL_ID", "minilm"),
        help="Embedding model id/alias (e.g., minilm, mpnet, BAAI/bge-m3, Qwen/Qwen3-Embedding-0.6B)",
    )

    ap.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Output directory for dense/sparse indices (default: <smartfarm-search>/data/index)",
    )
    ap.add_argument(
        "--trigraph-dir",
        type=Path,
        default=None,
        help="Output directory for Tri-Graph artifacts (default: <smartfarm-search>/data/index/trigraph_edge)",
    )
    args = ap.parse_args()

    ingest_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ingest_root / "src"))
    from dataset_pipeline.bootstrap import ensure_search_on_path

    search_root = ensure_search_on_path()

    # Import core components from smartfarm-search (keeps embedder behavior consistent with runtime).
    from core.Models.Schemas import SourceDoc
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import BM25Store

    corpus_path = args.input_jsonl.expanduser().resolve()
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    index_dir = args.index_dir.expanduser().resolve() if args.index_dir else (search_root / "data" / "index").resolve()
    trigraph_dir = (
        args.trigraph_dir.expanduser().resolve()
        if args.trigraph_dir
        else (search_root / "data" / "index" / "trigraph_edge").resolve()
    )

    tmp_trigraph_dir = trigraph_dir.with_suffix(".tmp")
    if tmp_trigraph_dir.exists():
        # Avoid partial leftovers from aborted runs.
        for p in tmp_trigraph_dir.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass

    index_dir.mkdir(parents=True, exist_ok=True)
    tmp_trigraph_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) Build sentence/chunk corpus
    #
    # This indexer supports two input styles:
    #   A) Raw documents (id has no '#cN' suffix) -> we window-chunk sentences and create chunk ids: <id>#c{n}
    #   B) Already-chunked corpus (id ends with '#cN') -> we keep chunk ids *as-is* (no double-chunking)
    # ---------------------------------------------------------------------
    doc_items = list(_iter_jsonl(corpus_path, limit=args.limit))
    n_docs = len(doc_items)
    if n_docs == 0:
        raise SystemExit(f"No documents found in {corpus_path}")

    sentence_texts: List[str] = []
    sentence_chunk_idx: List[int] = []
    sentence_tokens: List[List[str]] = []
    token_df: Counter[str] = Counter()

    chunk_docs: List[SourceDoc] = []
    chunk_ids: List[str] = []

    for i, item in enumerate(doc_items):
        raw_id = str(item.get("id") or item.get("_id") or f"doc{i}")
        text = _pick_text(item, args.lang)
        if not text:
            continue

        m = _CHUNK_ID_RE.match(raw_id)
        if m is not None:
            # Already chunked: keep id stable for evaluation & traceability.
            doc_id = str(m.group("doc"))
            chunk_index = int(m.group("idx"))
            chunk_id = raw_id
            chunk_text = text.strip()
            if not chunk_text:
                continue

            chunk_idx = len(chunk_ids)
            chunk_ids.append(chunk_id)
            chunk_docs.append(
                SourceDoc(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "lang": args.lang,
                    },
                )
            )

            sents = _sentence_split(chunk_text)
            for sent in sents:
                s = sent.strip()
                if not s:
                    continue
                s_idx = len(sentence_texts)
                sentence_texts.append(s)
                sentence_chunk_idx.append(chunk_idx)
                toks = _extract_candidate_tokens(s, min_len=int(args.min_entity_len))
                uniq = sorted(set(toks))
                sentence_tokens.append(uniq)
                token_df.update(uniq)
            continue

        # Raw doc: sentence split -> window chunks
        doc_id = raw_id
        sents = _sentence_split(text)
        if not sents:
            continue

        chunks = _window_chunks(sents, window=args.chunk_size, stride=args.chunk_stride)
        for chunk_index, chunk_sents in enumerate(chunks):
            chunk_id = f"{doc_id}#c{chunk_index}"
            chunk_text = "\n".join(chunk_sents).strip()
            if not chunk_text:
                continue
            chunk_idx = len(chunk_ids)
            chunk_ids.append(chunk_id)
            chunk_docs.append(
                SourceDoc(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "lang": args.lang,
                    },
                )
            )

            for sent in chunk_sents:
                s = sent.strip()
                if not s:
                    continue
                s_idx = len(sentence_texts)
                sentence_texts.append(s)
                sentence_chunk_idx.append(chunk_idx)
                toks = _extract_candidate_tokens(s, min_len=int(args.min_entity_len))
                uniq = sorted(set(toks))
                sentence_tokens.append(uniq)
                token_df.update(uniq)

    n_chunks = len(chunk_docs)
    n_sentences = len(sentence_texts)
    if n_chunks == 0 or n_sentences == 0:
        raise SystemExit("No chunks/sentences produced. Check input corpus or chunking params.")

    # ---------------------------------------------------------------------
    # 2) Select open-world entities (df-based; no ontology filters)
    # ---------------------------------------------------------------------
    max_df = int(math.floor(float(args.max_df_ratio) * float(n_sentences)))
    if max_df <= 0:
        max_df = 1

    candidate_entities: List[Tuple[str, int]] = []
    for tok, df in token_df.items():
        if df < int(args.min_df):
            continue
        if df > max_df:
            continue
        candidate_entities.append((tok, int(df)))

    # Score: prefer moderately frequent and longer tokens (still open-world).
    candidate_entities.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    if len(candidate_entities) > int(args.max_entities):
        candidate_entities = candidate_entities[: int(args.max_entities)]

    entity_names = [t for t, _df in candidate_entities]
    entity_to_idx = {e: i for i, e in enumerate(entity_names)}
    n_entities = len(entity_names)
    if n_entities == 0:
        raise SystemExit("No entities selected. Loosen --min-df / --max-df-ratio.")

    # ---------------------------------------------------------------------
    # 3) Build Tri-Graph adjacency (Entity–Sentence) + (Sentence->Chunk mapping)
    # ---------------------------------------------------------------------
    s2e_rows: List[List[int]] = [[] for _ in range(n_sentences)]
    e2s_rows: List[List[int]] = [[] for _ in range(n_entities)]

    for s_idx, toks in enumerate(sentence_tokens):
        for tok in toks:
            e_idx = entity_to_idx.get(tok)
            if e_idx is None:
                continue
            s2e_rows[s_idx].append(e_idx)
            e2s_rows[e_idx].append(s_idx)

    s2e_indptr, s2e_indices = _build_csr_from_lists(s2e_rows, n_cols=n_entities)
    e2s_indptr, e2s_indices = _build_csr_from_lists(e2s_rows, n_cols=n_sentences)

    sentence_chunk_idx_arr = np.asarray(sentence_chunk_idx, dtype=np.int32)

    # Optional convenience mapping: entity->chunk (deduplicated)
    e2c_rows: List[List[int]] = [[] for _ in range(n_entities)]
    for e_idx in range(n_entities):
        start = int(e2s_indptr[e_idx])
        end = int(e2s_indptr[e_idx + 1])
        sent_idxs = e2s_indices[start:end]
        chunk_idxs = sentence_chunk_idx_arr[sent_idxs].tolist() if sent_idxs.size else []
        e2c_rows[e_idx] = chunk_idxs
    e2c_indptr, e2c_indices = _build_csr_from_lists(e2c_rows, n_cols=n_chunks)

    # ---------------------------------------------------------------------
    # 4) Embeddings (entities + sentences) and dense/sparse indices (chunks)
    # ---------------------------------------------------------------------
    embedder = EmbeddingRetriever(model_id=str(args.embed_model_id), cache_size=32)

    # Entity embeddings
    entity_vecs = embedder.encode(entity_names, use_cache=False)
    entity_vecs = entity_vecs.astype(np.float16, copy=False)

    # Sentence embeddings (for semantic bridging)
    # Store float16 to reduce disk and memory-mapped footprint.
    sentence_vecs = embedder.encode(sentence_texts, use_cache=False)
    sentence_vecs = sentence_vecs.astype(np.float16, copy=False)

    # Dense index over chunks
    dense = EmbeddingRetriever(model_id=str(args.embed_model_id), cache_size=32)
    dense.build(chunk_docs)
    dense_index_path = index_dir / "dense.faiss"
    dense_docs_path = index_dir / "dense_docs.jsonl"
    dense.save(str(dense_index_path), str(dense_docs_path))

    # Sparse BM25 over chunks
    sparse = BM25Store()
    sparse.index(chunk_docs)
    sparse_state_path = index_dir / "sparse_bm25.pkl"
    sparse.save(str(sparse_state_path))

    # ---------------------------------------------------------------------
    # 5) Export Tri-Graph artifacts (atomic swap)
    # ---------------------------------------------------------------------
    (tmp_trigraph_dir / "entity_names.json").write_text(
        json.dumps(entity_names, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(tmp_trigraph_dir / "entity_embeddings.npy", entity_vecs)

    (tmp_trigraph_dir / "chunk_ids.json").write_text(
        json.dumps(chunk_ids, ensure_ascii=False),
        encoding="utf-8",
    )

    np.save(tmp_trigraph_dir / "sentence_embeddings.npy", sentence_vecs)
    np.save(tmp_trigraph_dir / "sentence_chunk_idx.npy", sentence_chunk_idx_arr)

    _save_csr_npz(tmp_trigraph_dir / "s2e.npz", s2e_indptr, s2e_indices, (n_sentences, n_entities))
    _save_csr_npz(tmp_trigraph_dir / "e2s.npz", e2s_indptr, e2s_indices, (n_entities, n_sentences))
    _save_csr_npz(tmp_trigraph_dir / "e2c.npz", e2c_indptr, e2c_indices, (n_entities, n_chunks))

    meta = TriGraphMeta(
        embed_model_id=str(dense.model_id),
        embedding_dim=int(getattr(dense, "dim", int(entity_vecs.shape[1]))),
        created_at=datetime.now(timezone.utc).isoformat(),
        corpus_path=str(corpus_path),
        corpus_sha256=_sha256_file(corpus_path),
        n_docs=n_docs,
        n_chunks=n_chunks,
        n_sentences=n_sentences,
        n_entities=n_entities,
        chunk_size=int(args.chunk_size),
        chunk_stride=int(args.chunk_stride),
    )
    (tmp_trigraph_dir / "meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    # Atomic replace
    if trigraph_dir.exists():
        old = trigraph_dir.with_suffix(".old")
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        trigraph_dir.rename(old)
    tmp_trigraph_dir.rename(trigraph_dir)

    print("[trigraph] DONE")
    print(f"  corpus={corpus_path} docs={n_docs}")
    print(f"  chunks={n_chunks} sentences={n_sentences} entities={n_entities}")
    print(f"  dense={dense_index_path}")
    print(f"  dense_docs={dense_docs_path}")
    print(f"  sparse={sparse_state_path}")
    print(f"  trigraph_dir={trigraph_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
