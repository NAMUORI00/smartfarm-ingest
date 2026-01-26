#!/usr/bin/env python3
"""
RAG grounding smoke test.

Goal:
- Verify that RAG retrieval actually injects corpus knowledge into the LLM prompt.
- Verify that the generated answer reflects a fact that only exists in the indexed document.

Usage:
  python dataset/test_rag_grounding.py
"""

from __future__ import annotations

import random
import sys
import tempfile
import uuid
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dataset_pipeline.llm_connector import LLMConnector
from dataset_pipeline.rag_connector import RAGConnector
from dataset_pipeline.refiner import AnswerRefiner


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


def main() -> int:
    _configure_console()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((repo_root / "config" / "settings.yaml").read_text(encoding="utf-8"))

    llm = LLMConnector(cfg["llm"])
    rag = RAGConnector(cfg["rag"])

    token = f"ragfact_{uuid.uuid4().hex[:10]}"
    value = f"{random.random() * 1000:.6f}"

    test_doc = Path(tempfile.gettempdir()) / f"rag_grounding_{token}.txt"
    test_doc.write_text(
        "\n".join(
            [
                "RAG Grounding Test Document",
                f"token: {token}",
                f"value: {value}",
                "",
                f"The ONLY correct value for {token} is {value}.",
            ]
        ),
        encoding="utf-8",
    )

    print("[1/3] Indexing test document:", test_doc)
    chunks = rag.load_documents(str(test_doc))
    if not chunks:
        print("FAIL: load_documents() returned 0 chunks")
        return 1
    rag.index_documents(chunks, collection_name="rag_grounding_test")

    # NOTE: SimpleVectorDB는 공백 기준 키워드 포함 여부로만 검색합니다.
    # 토큰이 조사(의/는 등)와 붙으면 매칭이 안 되므로 공백으로 분리합니다.
    question = f"{token} 의 value 는 무엇인가요? 숫자만 답변 내용에 포함해 주세요."

    print("\n[2/3] Retrieving context (RAG)...")
    retrieved = rag.retrieve(question, top_k=3)
    if not retrieved:
        print("FAIL: retrieve() returned 0 contexts")
        return 1

    retrieved_text = "\n\n".join([c.get("content", "") for c in retrieved])
    has_token = token in retrieved_text
    has_value = value in retrieved_text
    print("retrieved.contains_token:", has_token)
    print("retrieved.contains_value:", has_value)
    print("retrieved.preview:", retrieved_text[:200].replace("\n", " "))

    if not (has_token and has_value):
        print("FAIL: retrieved context does not contain the injected fact")
        return 1

    print("\n[3/3] Generating answer with/without RAG...")
    with_rag = AnswerRefiner(
        llm_connector=llm,
        rag_connector=rag,
        config={"use_rag": True, "temperature": 0.0, "min_answer_length": 1, "max_iterations": 1},
    ).generate_answer(question)

    without_rag = AnswerRefiner(
        llm_connector=llm,
        rag_connector=None,
        config={"use_rag": False, "temperature": 0.0, "min_answer_length": 1, "max_iterations": 1},
    ).generate_answer(question)

    ans_with = (with_rag.get("answer") or "").strip()
    ans_without = (without_rag.get("answer") or "").strip()

    print("answer_with_rag.contains_value:", value in ans_with)
    print("answer_without_rag.contains_value:", value in ans_without)
    print("answer_with_rag.preview:", ans_with[:200].replace("\n", " "))
    print("answer_without_rag.preview:", ans_without[:200].replace("\n", " "))

    if value not in ans_with:
        print("FAIL: answer (with RAG) did not include the injected fact")
        return 1

    if value in ans_without:
        print("WARN: answer (without RAG) also included the injected fact (unexpected, but possible).")
        print("PASS (weak): RAG path works; re-run to confirm.")
        return 0

    print("PASS: Answer reflects retrieved context (RAG grounding ok).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
