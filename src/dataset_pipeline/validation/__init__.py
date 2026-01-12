"""
Dataset Validation Module

학술적 표준에 따른 데이터셋 자동 검증 시스템.

주요 기능:
- Diversity: 질문 다양성 측정 (ROUGE-L, BERTScore)
- Groundedness: 답변 근거성 검증 (NLI)
- Consistency: Judge 일관성 측정 (Self-consistency)
- Ensemble: Multiple judge agreement

References:
- Self-Instruct (Wang et al., 2023, ACL)
- LLM-as-a-Judge (Zheng et al., 2024, NeurIPS)
- Prometheus (Kim et al., 2024, NeurIPS)
"""

from .diversity_metrics import DiversityValidator
from .groundedness import GroundednessValidator
from .judge_consistency import JudgeConsistencyValidator
from .validator import DatasetValidator

__all__ = [
    "DiversityValidator",
    "GroundednessValidator",
    "JudgeConsistencyValidator",
    "DatasetValidator",
]
