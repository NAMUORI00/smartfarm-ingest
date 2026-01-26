from __future__ import annotations

from typing import List


def split_text_recursive(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text using the same splitter defaults as dataset.pipeline.rag_connector."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text or "")

