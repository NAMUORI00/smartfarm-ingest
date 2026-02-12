from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ParsedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Optional[str]]


class DoclingParser:
    """Docling-first parser with modality-aware chunking and text fallback."""

    def __init__(self) -> None:
        self.enable_vlm = str(os.getenv("DOCLING_ENABLE_VLM", "true")).lower() in {"1", "true", "yes"}
        self.vlm_model = str(os.getenv("DOCLING_VLM_MODEL", "")).strip()
        self._converter = self._build_docling_converter()

    def _build_docling_converter(self):
        try:
            from docling.document_converter import DocumentConverter
        except Exception:
            return None

        if not self.enable_vlm:
            try:
                return DocumentConverter()
            except Exception:
                return None

        # Best-effort VLM profile setup: if pipeline options API differs by
        # Docling version, gracefully fall back to default converter.
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            options = PdfPipelineOptions()
            if hasattr(options, "do_ocr"):
                setattr(options, "do_ocr", True)
            if hasattr(options, "do_table_structure"):
                setattr(options, "do_table_structure", True)
            if hasattr(options, "do_formula_enrichment"):
                setattr(options, "do_formula_enrichment", True)
            if self.vlm_model and hasattr(options, "vlm_model"):
                setattr(options, "vlm_model", self.vlm_model)
            try:
                return DocumentConverter(pipeline_options=options)
            except TypeError:
                return DocumentConverter()
        except Exception:
            try:
                return DocumentConverter()
            except Exception:
                return None

    def _chunk_text(
        self,
        *,
        doc_stem: str,
        text: str,
        source_doc: str,
        modality: str = "text",
        asset_ref: str | None = None,
        chunk_index_seed: int = 0,
    ) -> List[ParsedChunk]:
        chunks: List[ParsedChunk] = []
        stride = 900
        overlap = 160
        i = 0
        idx = chunk_index_seed
        while i < len(text):
            seg = text[i : i + stride].strip()
            if seg:
                chunks.append(
                    ParsedChunk(
                        chunk_id=f"{doc_stem}#{modality[:1]}{idx}",
                        text=seg,
                        metadata={
                            "source_doc": source_doc,
                            "modality": modality,
                            "asset_ref": asset_ref,
                        },
                    )
                )
            i += max(1, stride - overlap)
            idx += 1
        return chunks

    def _extract_text(self, obj: Any) -> str:
        for key in ("text", "caption", "alt_text", "formula"):
            val = getattr(obj, key, None)
            if val:
                text = str(val).strip()
                if text:
                    return text
        return ""

    def _extract_asset_ref(self, obj: Any, fallback_index: int) -> str | None:
        for key in ("asset_ref", "ref", "self_ref", "id", "name"):
            val = getattr(obj, key, None)
            if val:
                ref = str(val).strip()
                if ref:
                    return ref
        page_no = getattr(obj, "page_no", None)
        if page_no is not None:
            return f"page:{page_no}"
        return f"asset:{fallback_index}"

    def _detect_modality(self, obj: Any) -> str:
        kind = str(type(obj).__name__ or "").lower()
        if "table" in kind:
            return "table"
        if "figure" in kind or "image" in kind or "picture" in kind:
            return "image"
        if "formula" in kind or "equation" in kind:
            return "formula"
        label = str(getattr(obj, "label", "") or "").lower()
        if "table" in label:
            return "table"
        if "image" in label or "figure" in label:
            return "image"
        if "formula" in label or "equation" in label:
            return "formula"
        return "text"

    def _iter_docling_segments(self, path: Path) -> Iterable[Dict[str, str | None]]:
        if self._converter is None:
            return []
        try:
            result = self._converter.convert(str(path))
            doc = getattr(result, "document", None)
            if doc is None:
                return []

            segments: List[Dict[str, str | None]] = []
            iterate = getattr(doc, "iterate_items", None)
            if callable(iterate):
                index = 0
                for item in iterate():
                    obj = item[0] if isinstance(item, tuple) else item
                    text = self._extract_text(obj)
                    if not text:
                        continue
                    modality = self._detect_modality(obj)
                    asset_ref = self._extract_asset_ref(obj, index)
                    segments.append(
                        {
                            "text": text,
                            "modality": modality,
                            "asset_ref": asset_ref if modality in {"image", "table", "formula"} else None,
                        }
                    )
                    index += 1
            if segments:
                return segments

            # Fallback to markdown export when available.
            to_md = getattr(doc, "export_to_markdown", None)
            if callable(to_md):
                md = str(to_md() or "").strip()
                if md:
                    return [{"text": md, "modality": "text", "asset_ref": None}]
        except Exception:
            return []
        return []

    def parse_file(self, path: str | Path) -> List[ParsedChunk]:
        p = Path(path)

        segments = list(self._iter_docling_segments(p))
        if segments:
            chunks: List[ParsedChunk] = []
            seed = 0
            for seg in segments:
                chunks.extend(
                    self._chunk_text(
                        doc_stem=p.stem,
                        text=str(seg.get("text") or ""),
                        source_doc=p.name,
                        modality=str(seg.get("modality") or "text"),
                        asset_ref=seg.get("asset_ref"),
                        chunk_index_seed=seed,
                    )
                )
                seed += 1
            return chunks

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""

        if not text.strip():
            return []
        return self._chunk_text(doc_stem=p.stem, text=text, source_doc=p.name, modality="text")
