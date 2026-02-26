from __future__ import annotations

"""Docling-based document parser with unstructured fallback.

Docling internally uses MinerU and other VLM pipelines to produce
high-fidelity document conversion.  When Docling is unavailable
(e.g. missing native dependencies), falls back to unstructured-based
DocumentParser transparently.
"""

from pathlib import Path
from typing import List

from pipeline.document_parser import DocumentParser, ParsedChunk

_HAS_DOCLING = False
_DocumentConverter = None

try:
    from docling.document_converter import DocumentConverter as _DC  # type: ignore[import-untyped]

    _HAS_DOCLING = True
    _DocumentConverter = _DC
except Exception:  # pragma: no cover - optional heavy dependency
    pass


class DoclingParser(DocumentParser):
    """VLM-backed parser using Docling (MinerU pipeline).

    Falls back to ``DocumentParser`` (unstructured) when the ``docling``
    package cannot be loaded.
    """

    def __init__(self) -> None:
        super().__init__()
        self._docling_converter = None
        if _HAS_DOCLING and _DocumentConverter is not None:
            try:
                self._docling_converter = _DocumentConverter()
            except Exception:
                self._docling_converter = None

    # ------------------------------------------------------------------
    # Override parse_file: prefer Docling, fallback to unstructured
    # ------------------------------------------------------------------
    def parse_file(self, path: str | Path) -> List[ParsedChunk]:
        p = Path(path)

        # Primary path: Docling VLM pipeline
        if self._docling_converter is not None:
            docling_chunks = self._parse_via_docling(p)
            if docling_chunks:
                return docling_chunks

        # Fallback: unstructured-based parent parser
        return super().parse_file(p)

    def _parse_via_docling(self, p: Path) -> List[ParsedChunk]:
        """Convert a document via Docling and produce ParsedChunk list."""
        if self._docling_converter is None:
            return []

        try:
            result = self._docling_converter.convert(str(p))
        except Exception:
            return []

        doc = getattr(result, "document", None)
        if doc is None:
            return []

        chunks: List[ParsedChunk] = []
        seed = 0

        # Iterate Docling document items (text blocks, tables, figures, etc.)
        items = []
        try:
            items = list(doc.iterate_items())
        except Exception:
            # Fallback: try export_to_markdown for simpler conversion
            try:
                md_text = doc.export_to_markdown()
                if md_text and str(md_text).strip():
                    return self._chunk_text(
                        doc_stem=p.stem,
                        text=str(md_text),
                        source_doc=p.name,
                        modality="text",
                    )
            except Exception:
                pass
            return []

        for item in items:
            item_obj = item[1] if isinstance(item, tuple) and len(item) > 1 else item
            text = ""
            modality = "text"
            table_html = None
            formula_latex = None
            image_b64 = None

            # Extract text
            try:
                text = str(getattr(item_obj, "text", "") or "").strip()
            except Exception:
                text = ""

            # Detect modality from Docling item type
            item_type = type(item_obj).__name__.lower()
            if "table" in item_type:
                modality = "table"
                try:
                    table_html = str(getattr(item_obj, "export_to_html", lambda: "")() or "").strip() or None
                except Exception:
                    pass
                if not text and table_html:
                    text = table_html
            elif "picture" in item_type or "figure" in item_type or "image" in item_type:
                modality = "image"
                try:
                    img_data = getattr(item_obj, "image", None)
                    if img_data is not None:
                        import base64
                        if isinstance(img_data, (bytes, bytearray)):
                            image_b64 = base64.b64encode(bytes(img_data)).decode("utf-8")
                        elif isinstance(img_data, str) and img_data.strip():
                            image_b64 = img_data.strip()
                except Exception:
                    pass
            elif "formula" in item_type or "equation" in item_type:
                modality = "formula"
                try:
                    formula_latex = str(getattr(item_obj, "latex", "") or "").strip() or None
                except Exception:
                    pass
                if not text and formula_latex:
                    text = formula_latex

            if not text:
                continue

            seg_chunks = self._chunk_text(
                doc_stem=p.stem,
                text=text,
                source_doc=p.name,
                modality=modality,
                chunk_index_seed=seed,
                table_html_ref=table_html,
                formula_latex_ref=formula_latex,
                image_b64_ref=image_b64,
            )
            chunks.extend(seg_chunks)
            seed += len(seg_chunks)

        return chunks


__all__ = ["DoclingParser", "DocumentParser", "ParsedChunk"]
