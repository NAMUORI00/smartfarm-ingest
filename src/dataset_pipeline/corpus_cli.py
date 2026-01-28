#!/usr/bin/env python3
"""Corpus tools (additive extension) for building EN/KO corpora from CGIAR datasets.

This CLI is designed to reuse existing dataset/ pipeline components:
- Chunking settings match dataset/pipeline/rag_connector.py defaults.
- LLM calls use dataset/pipeline/llm_connector.py (OpenAI-compatible).
- JSON robustness uses dataset/pipeline/json_utils.py.

Examples:
  # 1) Export EN corpus from official CGIAR docs
  python -m dataset.pipeline.corpus_cli export-cgiar \
    --config dataset/config/settings.yaml \
    --output dataset/output/wasabi_en_corpus.jsonl \
    --limit-per-dataset 200

  # 2) Translate EN -> KO with an API LLM (generator config)
  python -m dataset.pipeline.corpus_cli translate \
    --config dataset/config/settings.yaml \
    --input dataset/output/wasabi_en_corpus.jsonl \
    --output dataset/output/wasabi_en_ko_parallel.jsonl

  # 3) MQM-style scoring using one or more judges
  python -m dataset.pipeline.corpus_cli mqm-score \
    --config dataset/config/settings.yaml \
    --input dataset/output/wasabi_en_ko_parallel.jsonl \
    --output dataset/output/wasabi_en_ko_parallel_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .chunking_utils import split_text_recursive
from .constants import CorpusFields, MetadataFields
from .corpus_io import append_jsonl, load_existing_ids, read_jsonl
from .corpus_text import join_nonempty, normalize_text
from .llm_connector import LLMConnector
from .mt_tools import MQMJudge, MTTranslator, PostEditor, check_numbers_units
from .sources.cgiar import iter_many
from .sources.web_crawler import iter_wasabi_web_docs


DEFAULT_CGIAR_DATASETS = [
    "CGIAR/gardian-ai-ready-docs",
    "CGIAR/cirad-ai-documents",
    "CGIAR/ifpri-ai-documents",
]

_DEFAULT_CONFIG_PATH = str((Path(__file__).resolve().parents[1] / "config" / "settings.yaml").resolve())


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        dataset_root = Path(__file__).resolve().parents[1]
        for candidate in (repo_root / p, dataset_root / p):
            if candidate.exists():
                p = candidate
                break
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_dotenv_files() -> None:
    """Load .env from common locations (repo root and dataset/).

    This mirrors dataset.pipeline.llm_connector behavior, but runs for all subcommands,
    including export-cgiar (which may need HF tokens).
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    try:
        repo_root = Path(__file__).resolve().parents[2]
        load_dotenv(repo_root / ".env", override=False)
        load_dotenv(repo_root / "dataset" / ".env", override=False)
    except Exception:
        return


def _glossary_from_file(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".yml", ".yaml"):
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _keyword_filter(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords if k)


def cmd_export_cgiar(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    rag_cfg = cfg.get("rag") if isinstance(cfg.get("rag"), dict) else {}
    corpus_cfg = cfg.get("corpus") if isinstance(cfg.get("corpus"), dict) else {}
    chunk_size = int(args.chunk_size or rag_cfg.get("chunk_size", 512))
    chunk_overlap = int(args.chunk_overlap or rag_cfg.get("chunk_overlap", 50))

    repo_ids = args.datasets or corpus_cfg.get("cgiar_datasets") or DEFAULT_CGIAR_DATASETS
    cfg_keywords = corpus_cfg.get("filter_keywords") if isinstance(corpus_cfg.get("filter_keywords"), list) else []
    keywords = [k.strip() for k in (cfg_keywords + (args.filter_keyword or [])) if isinstance(k, str) and k.strip()]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_ids(out_path, id_key="id") if args.resume else set()
    rows: List[Dict[str, Any]] = []
    wrote = 0

    for d in iter_many(
        repo_ids,
        split=args.split,
        streaming=not args.no_streaming,
        limit_per_dataset=args.limit_per_dataset,
        hf_token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
    ):
        text = normalize_text(d.text)
        if not text:
            continue
        if not _keyword_filter(text, keywords):
            continue

        if args.granularity == "doc":
            rid = f"{d.id}"
            if rid in existing:
                continue
            rows.append(
                {
                    CorpusFields.ID: rid,
                    CorpusFields.TEXT: text,
                    CorpusFields.METADATA: {**(d.metadata or {}), "lang": "en"},
                }
            )
            wrote += 1
        else:
            chunks = split_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, ch in enumerate(chunks):
                ch = normalize_text(ch)
                if not ch:
                    continue
                rid = f"{d.id}#c{i}"
                if rid in existing:
                    continue
                rows.append(
                    {
                        CorpusFields.ID: rid,
                        CorpusFields.TEXT: ch,
                        CorpusFields.METADATA: {
                            **(d.metadata or {}),
                            "lang": "en",
                            "doc_id": d.id,
                            "chunk_idx": i,
                        },
                    }
                )
                wrote += 1

        if rows and len(rows) >= args.flush_every:
            append_jsonl(out_path, rows)
            rows.clear()

    if rows:
        append_jsonl(out_path, rows)

    print(f"[export-cgiar] wrote={wrote} -> {out_path}")
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    llm = LLMConnector(llm_cfg)
    translator = MTTranslator(llm)

    translation_cfg = cfg.get("translation") if isinstance(cfg.get("translation"), dict) else {}
    glossary_path = args.glossary or translation_cfg.get("glossary_path") or None
    glossary = _glossary_from_file(glossary_path)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_ids(out_path, id_key="id") if args.resume else set()

    batch: List[Dict[str, Any]] = []
    wrote = 0
    for row in read_jsonl(in_path):
        rid = row.get(CorpusFields.ID)
        text = row.get(CorpusFields.TEXT)
        meta = row.get(CorpusFields.METADATA) if isinstance(row.get(CorpusFields.METADATA), dict) else {}
        if not isinstance(rid, str) or not rid:
            continue
        if rid in existing:
            continue
        if not isinstance(text, str) or not text.strip():
            continue

        ko = translator.translate(
            text,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            glossary=glossary,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        ko = normalize_text(ko)

        nu = check_numbers_units(text, ko)
        out = {
            CorpusFields.ID: rid,
            CorpusFields.TEXT_EN: text,
            CorpusFields.TEXT_KO: ko,
            CorpusFields.METADATA: meta,
            "translation": {
                "backend": "api_llm",
                "src_lang": args.src_lang,
                "tgt_lang": args.tgt_lang,
                "model": llm_cfg.get("generator", {}).get("model"),
                "base_url": llm_cfg.get("generator", {}).get("base_url"),
                "numbers_units_ok": nu.ok,
                "missing_numbers_units": nu.missing_in_target,
            },
        }
        batch.append(out)
        wrote += 1

        if args.sleep:
            time.sleep(args.sleep)

        if len(batch) >= args.flush_every:
            append_jsonl(out_path, batch)
            batch.clear()

    if batch:
        append_jsonl(out_path, batch)

    print(f"[translate] wrote={wrote} -> {out_path}")
    return 0


def _load_mqm_judge_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    mqm_cfg = cfg.get("mqm") if isinstance(cfg.get("mqm"), dict) else {}
    judges = mqm_cfg.get("judges")
    if isinstance(judges, list) and judges:
        out: List[Dict[str, Any]] = []
        for j in judges:
            if not isinstance(j, dict):
                continue
            if not j.get("model"):
                continue
            out.append(j)
        if out:
            return out

    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    judge_cfg = llm_cfg.get("judge") if isinstance(llm_cfg.get("judge"), dict) else {}
    return [{"name": "judge", **judge_cfg}] if judge_cfg else []


def _mk_llm_from_single(cfg: Dict[str, Any]) -> LLMConnector:
    # Reuse LLMConnector by providing the same config for generator/judge.
    return LLMConnector({"generator": cfg, "judge": cfg})


def cmd_mqm_score(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    judge_cfgs = _load_mqm_judge_configs(cfg)
    if not judge_cfgs:
        raise RuntimeError("No MQM judges configured. Add `mqm.judges` or `llm.judge` in settings.yaml.")

    judges = []
    for jcfg in judge_cfgs:
        name = str(jcfg.get("name") or jcfg.get("model") or "judge")
        llm = _mk_llm_from_single(jcfg)
        judges.append((name, MQMJudge(llm)))

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_ids(out_path, id_key="id") if args.resume else set()

    batch: List[Dict[str, Any]] = []
    wrote = 0

    for row in read_jsonl(in_path):
        rid = row.get(CorpusFields.ID)
        if not isinstance(rid, str) or not rid:
            continue
        if rid in existing:
            continue

        src = row.get(CorpusFields.TEXT_EN)
        tgt = row.get(CorpusFields.TEXT_KO)
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue

        nu = check_numbers_units(src, tgt)
        per_judge: Dict[str, Any] = {}
        scores: List[float] = []
        for name, judge in judges:
            res = judge.score(
                src,
                tgt,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                temperature=args.temperature,
            )
            per_judge[name] = res
            s = res.get("overall_score")
            if isinstance(s, (int, float)):
                scores.append(float(s))
            if args.sleep:
                time.sleep(args.sleep)

        score_mean = sum(scores) / len(scores) if scores else 0.0
        out = dict(row)
        out["mqm"] = {
            "src_lang": args.src_lang,
            "tgt_lang": args.tgt_lang,
            "numbers_units_ok": nu.ok,
            "missing_numbers_units": nu.missing_in_target,
            "judges": per_judge,
            "overall_score_mean": score_mean,
            "num_judges": len(judges),
        }

        batch.append(out)
        wrote += 1

        if len(batch) >= args.flush_every:
            append_jsonl(out_path, batch)
            batch.clear()

    if batch:
        append_jsonl(out_path, batch)

    print(f"[mqm-score] wrote={wrote} -> {out_path}")
    return 0


def cmd_crawl_wasabi(args: argparse.Namespace) -> int:
    """Crawl wasabi cultivation documents from curated web sources."""
    cfg = _load_config(args.config)
    rag_cfg = cfg.get("rag") if isinstance(cfg.get("rag"), dict) else {}
    chunk_size = int(args.chunk_size or rag_cfg.get("chunk_size", 512))
    chunk_overlap = int(args.chunk_overlap or rag_cfg.get("chunk_overlap", 50))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_ids(out_path, id_key="id") if args.resume else set()
    rows: List[Dict[str, Any]] = []
    wrote = 0

    for doc in iter_wasabi_web_docs(limit=args.limit, delay=args.delay):
        text = normalize_text(doc.text)
        if not text:
            continue

        if args.granularity == "doc":
            rid = f"web_{doc.id}"
            if rid in existing:
                continue
            rows.append(
                {
                    CorpusFields.ID: rid,
                    CorpusFields.TEXT: text,
                    CorpusFields.METADATA: {
                        **doc.metadata,
                        "title": doc.title,
                        "url": doc.url,
                        "lang": "en",
                    },
                }
            )
            wrote += 1
        else:
            chunks = split_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, ch in enumerate(chunks):
                ch = normalize_text(ch)
                if not ch:
                    continue
                rid = f"web_{doc.id}#c{i}"
                if rid in existing:
                    continue
                rows.append(
                    {
                        CorpusFields.ID: rid,
                        CorpusFields.TEXT: ch,
                        CorpusFields.METADATA: {
                            **doc.metadata,
                            "title": doc.title,
                            "url": doc.url,
                            "lang": "en",
                            "doc_id": doc.id,
                            "chunk_idx": i,
                        },
                    }
                )
                wrote += 1

        if rows and len(rows) >= args.flush_every:
            append_jsonl(out_path, rows)
            rows.clear()

    if rows:
        append_jsonl(out_path, rows)

    print(f"[crawl-wasabi] wrote={wrote} -> {out_path}")
    return 0


def cmd_postedit(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    llm = LLMConnector(llm_cfg)
    post_editor = PostEditor(llm)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_ids(out_path, id_key="id") if args.resume else set()

    batch: List[Dict[str, Any]] = []
    wrote = 0
    for row in read_jsonl(in_path):
        rid = row.get(CorpusFields.ID)
        if not isinstance(rid, str) or not rid:
            continue
        if rid in existing:
            continue

        src = row.get(CorpusFields.TEXT_EN)
        tgt = row.get(CorpusFields.TEXT_KO)
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue

        pe = post_editor.post_edit(src, tgt, temperature=args.temperature, max_tokens=args.max_tokens)
        out = dict(row)
        out["post_edit"] = {
            "model": llm_cfg.get("judge", {}).get("model"),
            "base_url": llm_cfg.get("judge", {}).get("base_url"),
            "changes": pe.get("changes", []),
        }
        out["text_ko_postedit"] = pe.get("post_edited_text", tgt)
        batch.append(out)
        wrote += 1

        if args.sleep:
            time.sleep(args.sleep)

        if len(batch) >= args.flush_every:
            append_jsonl(out_path, batch)
            batch.clear()

    if batch:
        append_jsonl(out_path, batch)

    print(f"[postedit] wrote={wrote} -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help="YAML config path")

    p = argparse.ArgumentParser(
        description="Corpus tools for CGIAR -> EN/KO corpus + MQM scoring.",
        parents=[common],
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # export-cgiar
    s = sub.add_parser(
        "export-cgiar",
        help="Export EN corpus from official CGIAR datasets (JSON -> JSONL).",
        parents=[common],
    )
    s.add_argument("--datasets", nargs="*", default=None, help="HF dataset repo_ids (default: core CGIAR docs)")
    s.add_argument("--split", default="train")
    s.add_argument("--no-streaming", action="store_true", help="Disable streaming (downloads full dataset).")
    s.add_argument("--limit-per-dataset", type=int, default=None)
    s.add_argument("--output", required=True)
    s.add_argument("--granularity", choices=["chunk", "doc"], default="chunk")
    s.add_argument("--chunk-size", type=int, default=None)
    s.add_argument("--chunk-overlap", type=int, default=None)
    s.add_argument("--filter-keyword", action="append", default=[], help="Keep docs containing keyword (repeatable).")
    s.add_argument("--resume", action="store_true", help="Skip ids already present in output.")
    s.add_argument("--flush-every", type=int, default=200)
    s.set_defaults(func=cmd_export_cgiar)

    # translate
    s = sub.add_parser(
        "translate",
        help="Translate EN corpus JSONL to KO using API LLM (generator config).",
        parents=[common],
    )
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--src-lang", default="en")
    s.add_argument("--tgt-lang", default="ko")
    s.add_argument("--glossary", default=None, help="YAML/JSON glossary mapping for constrained translation.")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", type=int, default=2048)
    s.add_argument("--sleep", type=float, default=0.0)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--flush-every", type=int, default=20)
    s.set_defaults(func=cmd_translate)

    # mqm-score
    s = sub.add_parser(
        "mqm-score",
        help="MQM-style scoring for EN/KO parallel JSONL (LLM-as-a-judge).",
        parents=[common],
    )
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--src-lang", default="en")
    s.add_argument("--tgt-lang", default="ko")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--sleep", type=float, default=0.0)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--flush-every", type=int, default=20)
    s.set_defaults(func=cmd_mqm_score)

    # postedit
    s = sub.add_parser(
        "postedit",
        help="Post-edit KO translations (judge config) and append post-edited text.",
        parents=[common],
    )
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", type=int, default=1800)
    s.add_argument("--sleep", type=float, default=0.0)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--flush-every", type=int, default=20)
    s.set_defaults(func=cmd_postedit)

    # crawl-wasabi
    s = sub.add_parser(
        "crawl-wasabi",
        help="Crawl wasabi cultivation documents from curated web sources.",
        parents=[common],
    )
    s.add_argument("--output", required=True)
    s.add_argument("--limit", type=int, default=None, help="Max documents to crawl.")
    s.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds).")
    s.add_argument("--granularity", choices=["chunk", "doc"], default="chunk")
    s.add_argument("--chunk-size", type=int, default=None)
    s.add_argument("--chunk-overlap", type=int, default=None)
    s.add_argument("--resume", action="store_true", help="Skip ids already present in output.")
    s.add_argument("--flush-every", type=int, default=50)
    s.set_defaults(func=cmd_crawl_wasabi)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    _load_dotenv_files()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
