from __future__ import annotations

import base64
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
    """Docling-first parser with explicit VLM profile + text fallback."""

    def __init__(self) -> None:
        profile = str(os.getenv("DOCLING_VLM_PROFILE", "")).strip().lower()
        if profile in {"enabled", "disabled"}:
            self.enable_vlm = profile == "enabled"
        else:
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

        # Explicit VLM pipeline path (no PdfPipelineOptions fallback).
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import VlmPipelineOptions
            from docling.datamodel.pipeline_options_vlm_model import InferenceFramework, InlineVlmOptions
            from docling.document_converter import PdfFormatOption
            from docling.pipeline.vlm_pipeline import VlmPipeline

            model_id = self.vlm_model or "Qwen/Qwen2.5-VL-3B-Instruct"
            vlm_options = InlineVlmOptions(
                repo_id=model_id,
                prompt="Convert this page to markdown with all text, tables, formulas and figure captions.",
                inference_framework=InferenceFramework.TRANSFORMERS,
            )
            pipeline_options = VlmPipelineOptions(vlm_options=vlm_options)
            format_options = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
            return DocumentConverter(format_options=format_options)
        except Exception:
            # If explicit VLM pipeline is not available in current runtime,
            # degrade to default converter and keep text fallback behavior.
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
        table_html_ref: str | None = None,
        formula_latex_ref: str | None = None,
        image_b64_ref: str | None = None,
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
                            "table_html_ref": table_html_ref,
                            "formula_latex_ref": formula_latex_ref,
                            "image_b64_ref": image_b64_ref,
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

        table_html = self._extract_table_html(obj)
        if table_html:
            return table_html

        formula_latex = self._extract_formula_latex(obj)
        if formula_latex:
            return formula_latex

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

    def _extract_table_html(self, obj: Any) -> str | None:
        for key in ("table_html", "html"):
            val = getattr(obj, key, None)
            if val:
                text = str(val).strip()
                if text:
                    return text

        to_html = getattr(obj, "to_html", None)
        if callable(to_html):
            try:
                text = str(to_html() or "").strip()
                if text:
                    return text
            except Exception:
                pass
        return None

    def _extract_formula_latex(self, obj: Any) -> str | None:
        for key in ("formula_latex", "latex", "math"):
            val = getattr(obj, key, None)
            if val:
                text = str(val).strip()
                if text:
                    return text

        to_latex = getattr(obj, "to_latex", None)
        if callable(to_latex):
            try:
                text = str(to_latex() or "").strip()
                if text:
                    return text
            except Exception:
                pass
        return None

    def _extract_image_b64(self, obj: Any) -> str | None:
        for key in ("image_base64", "base64"):
            val = getattr(obj, key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()

        for key in ("image_bytes", "bytes"):
            raw = getattr(obj, key, None)
            if isinstance(raw, (bytes, bytearray)) and raw:
                return base64.b64encode(bytes(raw)).decode("utf-8")

        raw_image = getattr(obj, "image", None)
        if isinstance(raw_image, (bytes, bytearray)) and raw_image:
            return base64.b64encode(bytes(raw_image)).decode("utf-8")
        return None

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
                    table_html = self._extract_table_html(obj) if modality == "table" else None
                    formula_latex = self._extract_formula_latex(obj) if modality == "formula" else None
                    image_b64_ref = self._extract_image_b64(obj) if modality == "image" else None

                    segments.append(
                        {
                            "text": text,
                            "modality": modality,
                            "asset_ref": asset_ref if modality in {"image", "table", "formula"} else None,
                            "table_html": table_html,
                            "formula_latex": formula_latex,
                            "image_b64_ref": image_b64_ref,
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
                    return [
                        {
                            "text": md,
                            "modality": "text",
                            "asset_ref": None,
                            "table_html": None,
                            "formula_latex": None,
                            "image_b64_ref": None,
                        }
                    ]
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
                        table_html_ref=str(seg.get("table_html") or "") or None,
                        formula_latex_ref=str(seg.get("formula_latex") or "") or None,
                        image_b64_ref=str(seg.get("image_b64_ref") or "") or None,
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
