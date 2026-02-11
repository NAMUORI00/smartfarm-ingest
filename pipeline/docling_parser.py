from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ParsedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]


class DoclingParser:
    """Offline parser wrapper with Docling-first behavior.

    Priority:
    1) Docling parser (if installed)
    2) Plain text fallback
    """

    def __init__(self) -> None:
        self._converter = self._build_docling_converter()

    def _build_docling_converter(self):
        try:
            from docling.document_converter import DocumentConverter

            return DocumentConverter()
        except Exception:
            return None

    def _chunk_text(self, *, doc_stem: str, text: str, source_doc: str, modality: str = "text") -> List[ParsedChunk]:
        chunks: List[ParsedChunk] = []
        stride = 900
        overlap = 160
        i = 0
        idx = 0
        while i < len(text):
            seg = text[i : i + stride].strip()
            if seg:
                chunks.append(
                    ParsedChunk(
                        chunk_id=f"{doc_stem}#c{idx}",
                        text=seg,
                        metadata={"source_doc": source_doc, "modality": modality},
                    )
                )
            i += max(1, stride - overlap)
            idx += 1
        return chunks

    def _iter_docling_segments(self, path: Path) -> Iterable[str]:
        if self._converter is None:
            return []
        try:
            result = self._converter.convert(str(path))
            doc = getattr(result, "document", None)
            if doc is None:
                return []

            segments: List[str] = []
            iterate = getattr(doc, "iterate_items", None)
            if callable(iterate):
                for item in iterate():
                    obj = item[0] if isinstance(item, tuple) else item
                    text = str(getattr(obj, "text", "") or "").strip()
                    if not text:
                        text = str(getattr(obj, "caption", "") or "").strip()
                    if text:
                        segments.append(text)
            if segments:
                return segments

            # Final fallback to markdown export when available.
            to_md = getattr(doc, "export_to_markdown", None)
            if callable(to_md):
                md = str(to_md() or "").strip()
                if md:
                    return [md]
        except Exception:
            return []
        return []

    def parse_file(self, path: str | Path) -> List[ParsedChunk]:
        p = Path(path)

        # Docling path
        segments = list(self._iter_docling_segments(p))
        if segments:
            merged = "\n\n".join(segments)
            return self._chunk_text(doc_stem=p.stem, text=merged, source_doc=p.name, modality="multimodal")

        # Plain text fallback
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""

        if not text.strip():
            return []
        return self._chunk_text(doc_stem=p.stem, text=text, source_doc=p.name, modality="text")
