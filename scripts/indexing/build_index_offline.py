#!/usr/bin/env python3
"""
오프라인 인덱스 빌드 스크립트 (서버 없이 직접 실행).

JSONL 말뭉치를 로드하여 Dense/Sparse 인덱스를 구축하고 디스크에 저장.

사용 예:
  python scripts/indexing/build_index_offline.py \
    --corpus /path/to/corpus.jsonl \
    --lang ko
  # Or set DATA_ROOT env var and omit --corpus

출력:
  data/index/dense.faiss
  data/index/dense_docs.jsonl
  data/index/sparse.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

if TYPE_CHECKING:
    from core.Models.Schemas import SourceDoc


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_text(row: Dict[str, Any], lang: str) -> str:
    """Extract text from row based on language preference."""
    if lang == "ko":
        return row.get("text_ko") or row.get("text", "")
    elif lang == "en":
        return row.get("text_en") or row.get("text", "")
    else:
        # Both
        text_en = row.get("text_en") or row.get("text", "")
        text_ko = row.get("text_ko", "")
        if text_ko:
            return f"{text_en}\n\n[한국어]\n{text_ko}"
        return text_en


def load_corpus(path: Path, lang: str, limit: int | None = None) -> List[SourceDoc]:
    """Load corpus from JSONL file."""
    from core.Models.Schemas import SourceDoc

    docs = []
    for i, row in enumerate(iter_jsonl(path)):
        if limit and i >= limit:
            break
        
        text = get_text(row, lang)
        if not text.strip():
            continue
        
        doc_id = row.get("id", f"doc_{i}")
        metadata = row.get("metadata", {})
        metadata["original_id"] = doc_id
        metadata["source_file"] = str(path.name)
        metadata["lang"] = lang
        
        docs.append(SourceDoc(
            id=doc_id,
            text=text,
            metadata=metadata,
        ))
    
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build index offline from JSONL corpus.")
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to JSONL corpus file (default: uses DATA_ROOT or CORPUS_PATH env).",
    )
    parser.add_argument(
        "--lang",
        choices=["ko", "en", "both"],
        default="ko",
        help="Language to index (ko, en, both).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for index files (default: <smartfarm-search>/data/index).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of docs to index (for testing).",
    )
    args = parser.parse_args()

    ingest_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ingest_root / "src"))
    from dataset_pipeline.bootstrap import ensure_search_on_path

    search_root = ensure_search_on_path()

    from core.Config.Settings import settings
    from core.Services.Ingest.GraphBuilder import build_graph_from_docs
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Services.Retrieval.Sparse import MiniStore

    # Resolve corpus path: use --corpus argument if provided, otherwise fall back to Settings
    if args.corpus:
        corpus_path = Path(args.corpus).expanduser().resolve()
    else:
        corpus_path = Path(settings.get_corpus_path()).expanduser().resolve()

    if not corpus_path.exists():
        raise SystemExit(f"Corpus file not found: {corpus_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (search_root / "data" / "index").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OFFLINE INDEX BUILD")
    print("=" * 60)
    print(f"  Corpus: {corpus_path}")
    print(f"  Language: {args.lang}")
    print(f"  Output: {output_dir}")
    print()

    # 1. Load corpus
    print("[1/4] Loading corpus...")
    start = time.perf_counter()
    docs = load_corpus(corpus_path, args.lang, args.limit)
    load_time = time.perf_counter() - start
    print(f"  Loaded {len(docs)} documents in {load_time:.2f}s")

    if not docs:
        raise SystemExit("No documents found in corpus.")

    # 2. Build Dense index
    print("\n[2/4] Building Dense index (FAISS)...")
    start = time.perf_counter()
    dense = EmbeddingRetriever()
    dense.build(docs)
    dense_time = time.perf_counter() - start
    print(f"  Built Dense index in {dense_time:.2f}s")

    # 3. Build Sparse index
    print("\n[3/4] Building Sparse index (TF-IDF)...")
    start = time.perf_counter()
    sparse = MiniStore()
    sparse.index(docs)
    sparse_time = time.perf_counter() - start
    print(f"  Built Sparse index in {sparse_time:.2f}s")

    # 4. Save indices
    print("\n[4/4] Saving indices...")
    dense_index_path = output_dir / settings.DENSE_INDEX_FILE
    dense_docs_path = output_dir / settings.DENSE_DOCS_FILE
    sparse_path = output_dir / settings.SPARSE_STATE_FILE

    dense.save(str(dense_index_path), str(dense_docs_path))
    sparse.save(str(sparse_path))
    print(f"  Saved: {dense_index_path}")
    print(f"  Saved: {dense_docs_path}")
    print(f"  Saved: {sparse_path}")

    # 5. Build TriGraph (optional)
    print("\n[Bonus] Building causal graph...")
    start = time.perf_counter()
    graph = build_graph_from_docs(docs)
    graph_time = time.perf_counter() - start
    print(f"  Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges in {graph_time:.2f}s")

    # Summary
    total_time = load_time + dense_time + sparse_time + graph_time
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Total documents: {len(docs)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"    - Corpus load: {load_time:.2f}s")
    print(f"    - Dense index: {dense_time:.2f}s")
    print(f"    - Sparse index: {sparse_time:.2f}s")
    print(f"    - Graph build: {graph_time:.2f}s")
    print(f"\n  Index files saved to: {output_dir}")


if __name__ == "__main__":
    main()
