"""
RAG 기반 LLM-as-a-Judge 데이터셋 구축 시스템

참고 연구:
- Self-Instruct (Wang et al., 2023, ACL)
- Evol-Instruct (Xu et al., 2023, ICLR)
- RAFT (Zhang et al., 2024, COLM)
- LLM-as-a-Judge (Zheng et al., 2024, NeurIPS)
- Prometheus (Kim et al., 2024, NeurIPS)
"""

__version__ = "1.0.0"
__author__ = "Smart Farm RAG Team"

from .config import ConfigManager, get_config
from .main import DatasetPipeline
from .llm_connector import LLMConnector
from .rag_connector import RAGConnector
from .generator import QuestionGenerator
from .judge import LLMJudge
from .refiner import AnswerRefiner

__all__ = [
    "ConfigManager",
    "get_config",
    "DatasetPipeline",
    "LLMConnector",
    "RAGConnector",
    "QuestionGenerator",
    "LLMJudge",
    "AnswerRefiner",
]