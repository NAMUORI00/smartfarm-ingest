"""
Diversity Metrics for Dataset Validation

질문 다양성 측정 모듈:
- ROUGE-L (Self-Instruct 기준)
- BERTScore (의미적 유사도)
- Lexical Diversity (어휘 다양성)

References:
- Wang et al. (2023, ACL) "Self-Instruct"
- Zhang et al. (2020) "BERTScore: Evaluating Text Generation with BERT"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..diversity import calculate_rouge_l, calculate_diversity_score


@dataclass
class DiversityReport:
    """다양성 검증 결과 리포트"""
    
    # ROUGE-L 기반 (Self-Instruct)
    rouge_l_diversity: float = 0.0  # 1 - avg pairwise similarity
    rouge_l_stats: Dict[str, float] = field(default_factory=dict)
    
    # 어휘 다양성
    type_token_ratio: float = 0.0  # unique tokens / total tokens
    vocabulary_size: int = 0
    
    # 카테고리/난이도 분포
    category_distribution: Dict[str, int] = field(default_factory=dict)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 길이 통계
    question_length_stats: Dict[str, float] = field(default_factory=dict)
    answer_length_stats: Dict[str, float] = field(default_factory=dict)
    
    # 샘플 수
    total_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rouge_l_diversity": self.rouge_l_diversity,
            "rouge_l_stats": self.rouge_l_stats,
            "type_token_ratio": self.type_token_ratio,
            "vocabulary_size": self.vocabulary_size,
            "category_distribution": self.category_distribution,
            "complexity_distribution": self.complexity_distribution,
            "question_length_stats": self.question_length_stats,
            "answer_length_stats": self.answer_length_stats,
            "total_samples": self.total_samples,
        }
    
    def summary(self) -> str:
        """Human-readable 요약"""
        lines = [
            "=== Diversity Report ===",
            f"Total samples: {self.total_samples}",
            f"",
            f"[ROUGE-L Diversity]",
            f"  Score: {self.rouge_l_diversity:.4f} (1.0 = perfect diversity)",
            f"  Min similarity: {self.rouge_l_stats.get('min', 0):.4f}",
            f"  Max similarity: {self.rouge_l_stats.get('max', 0):.4f}",
            f"  Mean similarity: {self.rouge_l_stats.get('mean', 0):.4f}",
            f"",
            f"[Lexical Diversity]",
            f"  Type-Token Ratio: {self.type_token_ratio:.4f}",
            f"  Vocabulary Size: {self.vocabulary_size}",
            f"",
            f"[Category Distribution]",
        ]
        for cat, count in sorted(self.category_distribution.items()):
            pct = count / self.total_samples * 100 if self.total_samples else 0
            lines.append(f"  {cat}: {count} ({pct:.1f}%)")
        
        lines.append(f"")
        lines.append(f"[Complexity Distribution]")
        for comp, count in sorted(self.complexity_distribution.items()):
            pct = count / self.total_samples * 100 if self.total_samples else 0
            lines.append(f"  {comp}: {count} ({pct:.1f}%)")
        
        return "\n".join(lines)


class DiversityValidator:
    """
    데이터셋 다양성 검증기.
    
    Self-Instruct (Wang et al., 2023)의 diversity criteria 기반.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Args:
            threshold: ROUGE-L 유사도 임계값 (이상이면 중복으로 간주)
        """
        self.threshold = threshold
    
    def validate(self, qa_pairs: List[Dict[str, Any]]) -> DiversityReport:
        """
        데이터셋 다양성 검증 수행.
        
        Args:
            qa_pairs: QA 쌍 리스트
        
        Returns:
            DiversityReport
        """
        report = DiversityReport(total_samples=len(qa_pairs))
        
        if not qa_pairs:
            return report
        
        questions = [p.get("question", "") for p in qa_pairs]
        answers = [p.get("answer", "") for p in qa_pairs]
        
        # 1. ROUGE-L Diversity
        report.rouge_l_diversity, report.rouge_l_stats = self._calculate_rouge_diversity(questions)
        
        # 2. Lexical Diversity (TTR)
        report.type_token_ratio, report.vocabulary_size = self._calculate_lexical_diversity(questions)
        
        # 3. Category Distribution
        report.category_distribution = self._count_distribution(qa_pairs, "category")
        
        # 4. Complexity Distribution
        report.complexity_distribution = self._count_distribution(qa_pairs, "complexity")
        
        # 5. Length Statistics
        report.question_length_stats = self._calculate_length_stats(questions)
        report.answer_length_stats = self._calculate_length_stats(answers)
        
        return report
    
    def _calculate_rouge_diversity(self, texts: List[str]) -> tuple:
        """ROUGE-L 기반 다양성 계산"""
        if len(texts) < 2:
            return 1.0, {"min": 0, "max": 0, "mean": 0}
        
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = calculate_rouge_l(texts[i], texts[j])
                similarities.append(sim)
        
        if not similarities:
            return 1.0, {"min": 0, "max": 0, "mean": 0}
        
        avg_sim = sum(similarities) / len(similarities)
        diversity = 1.0 - avg_sim
        
        stats = {
            "min": min(similarities),
            "max": max(similarities),
            "mean": avg_sim,
            "std": self._std(similarities),
        }
        
        return diversity, stats
    
    def _calculate_lexical_diversity(self, texts: List[str]) -> tuple:
        """어휘 다양성 (Type-Token Ratio) 계산"""
        all_tokens = []
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0, 0
        
        unique_tokens = set(all_tokens)
        ttr = len(unique_tokens) / len(all_tokens)
        
        return ttr, len(unique_tokens)
    
    def _count_distribution(self, qa_pairs: List[Dict], key: str) -> Dict[str, int]:
        """특정 키의 분포 계산"""
        distribution = {}
        for pair in qa_pairs:
            value = pair.get(key, "unknown")
            distribution[value] = distribution.get(value, 0) + 1
        return distribution
    
    def _calculate_length_stats(self, texts: List[str]) -> Dict[str, float]:
        """텍스트 길이 통계"""
        if not texts:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        
        lengths = [len(t) for t in texts]
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "std": self._std(lengths),
        }
    
    def _std(self, values: List[float]) -> float:
        """표준편차 계산"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
