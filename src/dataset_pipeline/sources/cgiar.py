from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class CgiarDoc:
    id: str
    text: str
    metadata: Dict[str, Any]


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _as_list(x: Any) -> List[Any]:
    return list(x) if isinstance(x, list) else []


def _get_nested(d: Any, dotted: str, default=None):
    try:
        cur = d
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur
    except Exception:
        return default


def _flatten_gardian_ai_ready(row: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    doc_id = _safe_str(row.get("sieverID") or _get_nested(row, "metadata.id") or row.get("title") or "")
    title = _safe_str(row.get("title") or "")
    abstract = _safe_str(row.get("abstract") or "")

    chapters = _as_list(row.get("chapters"))
    chapter_parts: List[str] = []
    for ch in chapters:
        head = _safe_str(_get_nested(ch, "head", "")).strip()
        paragraphs = _as_list(_get_nested(ch, "paragraphs", []))
        para_texts = [_safe_str(_get_nested(p, "text", "")).strip() for p in paragraphs]
        para_texts = [t for t in para_texts if t]
        if not (head or para_texts):
            continue
        if head:
            chapter_parts.append(head)
        chapter_parts.extend(para_texts)

    figures = _as_list(row.get("figures"))
    figure_texts = [_safe_str(_get_nested(f, "text", "")).strip() for f in figures]
    figure_texts = [t for t in figure_texts if t]

    parts: List[str] = []
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    if chapter_parts:
        parts.append("\n".join(chapter_parts))
    if figure_texts:
        parts.append("\n".join(figure_texts))
    text = "\n\n".join([p for p in parts if p.strip()])

    meta = {
        "source_dataset": "CGIAR/gardian-ai-ready-docs",
        "title": title,
        "keywords": row.get("keywords") if isinstance(row.get("keywords"), list) else [],
        "url": _get_nested(row, "metadata.url"),
        "raw_metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "page_count": row.get("pageCount"),
    }
    return doc_id, text, meta


def _flatten_content_dataset(dataset_id: str, row: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    doc_id = _safe_str(row.get("sieverID") or _get_nested(row, "metadata.gardian_id") or _get_nested(row, "metadata.id") or "")
    text = _safe_str(row.get("content") or "")
    meta = {
        "source_dataset": dataset_id,
        "keywords": row.get("keywords") if isinstance(row.get("keywords"), list) else [],
        "url": _get_nested(row, "metadata.url"),
        "raw_metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "page_count": row.get("pagecount"),
        "token_count": row.get("tokenCount"),
    }
    return doc_id, text, meta


def iter_cgiar_docs(
    repo_id: str,
    split: str = "train",
    *,
    streaming: bool = True,
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Iterator[CgiarDoc]:
    """Stream text documents from official CGIAR Hugging Face datasets.

    Supported (tested):
    - CGIAR/gardian-ai-ready-docs
    - CGIAR/cirad-ai-documents
    - CGIAR/ifpri-ai-documents
    """
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, streaming=streaming, token=hf_token)
    n = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        if repo_id == "CGIAR/gardian-ai-ready-docs":
            doc_id, text, meta = _flatten_gardian_ai_ready(row)
        elif repo_id in ("CGIAR/cirad-ai-documents", "CGIAR/ifpri-ai-documents"):
            doc_id, text, meta = _flatten_content_dataset(repo_id, row)
        else:
            # Best-effort fallback for other CGIAR datasets that have 'content'.
            doc_id, text, meta = _flatten_content_dataset(repo_id, row)
            meta["note"] = "fallback_mapping_used"

        doc_id = (doc_id or "").strip()
        text = (text or "").strip()
        if not doc_id or not text:
            continue
        yield CgiarDoc(id=doc_id, text=text, metadata=meta)
        n += 1
        if limit and n >= limit:
            break


def iter_many(
    repo_ids: Iterable[str],
    split: str = "train",
    *,
    streaming: bool = True,
    limit_per_dataset: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Iterator[CgiarDoc]:
    for rid in repo_ids:
        yield from iter_cgiar_docs(
            rid,
            split=split,
            streaming=streaming,
            limit=limit_per_dataset,
            hf_token=hf_token,
        )

