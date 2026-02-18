from __future__ import annotations

# Backward-compatible shim.
from pipeline.document_parser import DocumentParser, ParsedChunk


class DoclingParser(DocumentParser):
    pass


__all__ = ["DoclingParser", "DocumentParser", "ParsedChunk"]
