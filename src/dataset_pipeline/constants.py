"""
Dataset Pipeline Constants

Centralizes field names, magic numbers, and defaults.
Import these instead of using string literals.
"""

from typing import Final

class CorpusFields:
    """Field names for corpus JSONL documents."""
    ID: Final[str] = "id"
    TEXT: Final[str] = "text"
    TEXT_EN: Final[str] = "text_en"
    TEXT_KO: Final[str] = "text_ko"
    METADATA: Final[str] = "metadata"

class MetadataFields:
    """Field names for document metadata."""
    LANG: Final[str] = "lang"
    SOURCE: Final[str] = "source"
    DOC_ID: Final[str] = "doc_id"
    CHUNK_IDX: Final[str] = "chunk_idx"
    URL: Final[str] = "url"
    TITLE: Final[str] = "title"

class QAFields:
    """Field names for QA dataset JSONL documents."""
    ID: Final[str] = "id"
    QUESTION: Final[str] = "question"
    QUESTION_KO: Final[str] = "question_ko"
    ANSWER: Final[str] = "answer"
    ANSWER_KO: Final[str] = "answer_ko"
    CONTEXT: Final[str] = "context"
    CATEGORY: Final[str] = "category"
    COMPLEXITY: Final[str] = "complexity"
    SOURCE_IDS: Final[str] = "source_ids"

class ComplexityLevels:
    """Valid complexity level values."""
    BASIC: Final[str] = "basic"
    INTERMEDIATE: Final[str] = "intermediate"
    ADVANCED: Final[str] = "advanced"
    ALL: Final[tuple] = (BASIC, INTERMEDIATE, ADVANCED)

class Defaults:
    """Default values for pipeline operations."""
    # Chunking
    CHUNK_SIZE: Final[int] = 512
    CHUNK_OVERLAP: Final[int] = 50

    # Translation
    SRC_LANG: Final[str] = "en"
    TGT_LANG: Final[str] = "ko"
    TRANSLATION_TEMPERATURE: Final[float] = 0.0
    TRANSLATION_MAX_TOKENS: Final[int] = 2048

    # QA Generation
    NUM_QUESTIONS: Final[int] = 220
    QUESTIONS_PER_CHUNK: Final[int] = 3
    EVOLUTION_PROBABILITY: Final[float] = 0.3

    # Batch Processing
    BATCH_SIZE: Final[int] = 10
    FLUSH_EVERY: Final[int] = 20
    MAX_RETRIES: Final[int] = 3
    SLEEP_BETWEEN_REQUESTS: Final[float] = 0.3

    # MQM Scoring
    MQM_PASS_THRESHOLD: Final[float] = 4.5

class EnvVars:
    """Environment variable names used by the pipeline."""
    DATA_ROOT: Final[str] = "DATA_ROOT"
    PROJECT_ROOT: Final[str] = "PIPELINE_PROJECT_ROOT"
    INPUT_FILE: Final[str] = "INPUT_FILE"
    BATCH_SIZE: Final[str] = "BATCH_SIZE"
    SLEEP_BETWEEN: Final[str] = "SLEEP_BETWEEN"
    MAX_RETRIES: Final[str] = "MAX_RETRIES"
    API_KEY: Final[str] = "API_KEY"
    OPENAI_API_KEY: Final[str] = "OPENAI_API_KEY"
    OPENAI_BASE_URL: Final[str] = "OPENAI_BASE_URL"
    HF_TOKEN: Final[str] = "HF_TOKEN"
