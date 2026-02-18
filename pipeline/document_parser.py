from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_TOKEN_RE = re.compile(r"\S+")


@dataclass
class ParsedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Optional[str]]


class DocumentParser:
    """Unstructured-first parser with text fallback.

    For compatibility with older tests/code paths, this class keeps `_converter`
    attribute semantics: setting `_converter=None` forces text fallback.
    """

    def __init__(self) -> None:
        self.chunk_token_size = 1200
        self.chunk_token_overlap = 100
        self.strategy = "hi_res"
        self.extract_images = True
        self.infer_table_structure = True
        self.languages = ["kor", "eng"]

        # Legacy compatibility toggle used by tests.
        self._converter = self._build_unstructured()

    def _build_unstructured(self):
        try:
            from unstructured.partition.auto import partition
        except Exception:
            return None
        return partition

    def _chunk_text(
        self,
        *,
        doc_stem: str,
        text: str,
        source_doc: str,
        modality: str = "text",
        asset_ref: str | None = None,
        chunk_index_seed: int = 0,
        table_html_ref: str | None = None,
        formula_latex_ref: str | None = None,
        image_b64_ref: str | None = None,
    ) -> List[ParsedChunk]:
        chunks: List[ParsedChunk] = []
        base_meta: Dict[str, Optional[str]] = {
            "source_doc": source_doc,
            "modality": modality,
            "asset_ref": asset_ref,
            "table_html_ref": table_html_ref,
            "formula_latex_ref": formula_latex_ref,
            "image_b64_ref": image_b64_ref,
        }

        raw = str(text or "").strip()
        if not raw:
            return chunks

        if modality in {"table", "image", "formula"}:
            return [
                ParsedChunk(
                    chunk_id=f"{doc_stem}#{modality[:1]}{chunk_index_seed}",
                    text=raw,
                    metadata=base_meta,
                )
            ]

        tokens = _TOKEN_RE.findall(raw)
        if not tokens:
            return chunks

        token_size = int(self.chunk_token_size)
        overlap = min(int(self.chunk_token_overlap), max(0, token_size - 1))
        step = max(1, token_size - overlap)
        idx = chunk_index_seed

        for start in range(0, len(tokens), step):
            window = tokens[start : start + token_size]
            if not window:
                break
            seg = " ".join(window).strip()
            if seg:
                chunks.append(
                    ParsedChunk(
                        chunk_id=f"{doc_stem}#t{idx}",
                        text=seg,
                        metadata=base_meta,
                    )
                )
                idx += 1
            if start + token_size >= len(tokens):
                break
        return chunks

    def _detect_modality(self, category: str) -> str:
        c = str(category or "").lower()
        if "table" in c:
            return "table"
        if "image" in c or "figure" in c or "picture" in c:
            return "image"
        if "formula" in c or "equation" in c or "math" in c:
            return "formula"
        return "text"

    def _safe_attr(self, obj, name: str, default=None):
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    def _to_image_b64(self, value) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("utf-8")
        return None

    def _iter_unstructured_segments(self, path: Path) -> List[Dict[str, str | None]]:
        if self._converter is None:
            return []

        try:
            elements = self._converter(
                filename=str(path),
                strategy=self.strategy,
                extract_images_in_pdf=bool(self.extract_images),
                infer_table_structure=bool(self.infer_table_structure),
                languages=self.languages,
            )
        except Exception:
            return []

        segments: List[Dict[str, str | None]] = []
        for idx, e in enumerate(elements or []):
            text = str(self._safe_attr(e, "text", "") or "").strip()
            category = str(self._safe_attr(e, "category", "") or "")
            meta = self._safe_attr(e, "metadata", None)
            modality = self._detect_modality(category)

            table_html = None
            formula_latex = None
            image_b64 = None
            page_no = ""
            if meta is not None:
                try:
                    table_html = str(getattr(meta, "text_as_html", "") or "").strip() or None
                except Exception:
                    table_html = None
                try:
                    formula_latex = str(getattr(meta, "text_as_latex", "") or "").strip() or None
                except Exception:
                    formula_latex = None
                try:
                    image_b64 = self._to_image_b64(getattr(meta, "image_base64", None))
                except Exception:
                    image_b64 = None
                try:
                    page_no = str(getattr(meta, "page_number", "") or "")
                except Exception:
                    page_no = ""

            if not text:
                if modality == "table" and table_html:
                    text = table_html
                elif modality == "formula" and formula_latex:
                    text = formula_latex

            if not text:
                continue

            asset_ref = f"page:{page_no}#idx:{idx}" if page_no else f"asset:{idx}"
            segments.append(
                {
                    "text": text,
                    "modality": modality,
                    "asset_ref": asset_ref if modality in {"table", "image", "formula"} else None,
                    "table_html": table_html,
                    "formula_latex": formula_latex,
                    "image_b64_ref": image_b64,
                }
            )
        return segments

    def parse_file(self, path: str | Path) -> List[ParsedChunk]:
        p = Path(path)

        segments = self._iter_unstructured_segments(p)
        if segments:
            chunks: List[ParsedChunk] = []
            seed = 0
            for seg in segments:
                seg_chunks = self._chunk_text(
                    doc_stem=p.stem,
                    text=str(seg.get("text") or ""),
                    source_doc=p.name,
                    modality=str(seg.get("modality") or "text"),
                    asset_ref=seg.get("asset_ref"),
                    chunk_index_seed=seed,
                    table_html_ref=str(seg.get("table_html") or "") or None,
                    formula_latex_ref=str(seg.get("formula_latex") or "") or None,
                    image_b64_ref=str(seg.get("image_b64_ref") or "") or None,
                )
                chunks.extend(seg_chunks)
                seed += len(seg_chunks)
            return chunks

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""

        if not text.strip():
            return []
        return self._chunk_text(doc_stem=p.stem, text=text, source_doc=p.name, modality="text")
