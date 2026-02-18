__all__ = [
    "DocumentParser",
    "DoclingParser",
    "ParsedChunk",
    "KGWriter",
    "LLMExtractor",
    "MultiLLMExtractor",
    "ExtractionInput",
    "VectorWriter",
]


def __getattr__(name: str):
    if name in {"DocumentParser", "ParsedChunk"}:
        from .document_parser import DocumentParser, ParsedChunk

        return {"DocumentParser": DocumentParser, "ParsedChunk": ParsedChunk}[name]
    if name == "DoclingParser":
        from .docling_parser import DoclingParser

        return DoclingParser
    if name == "KGWriter":
        from .kg_writer import KGWriter

        return KGWriter
    if name in {"LLMExtractor", "MultiLLMExtractor", "ExtractionInput"}:
        from .llm_extractor import ExtractionInput, LLMExtractor, MultiLLMExtractor

        return {
            "LLMExtractor": LLMExtractor,
            "MultiLLMExtractor": MultiLLMExtractor,
            "ExtractionInput": ExtractionInput,
        }[name]
    if name == "VectorWriter":
        from .vector_writer import VectorWriter

        return VectorWriter
    raise AttributeError(name)
