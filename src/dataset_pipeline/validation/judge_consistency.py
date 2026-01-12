"""
Judge Consistency Validation Module

LLM Judge의 신뢰성 검증:
1. Self-Consistency: 동일 입력에 대한 점수 일관성
2. Multi-Judge Ensemble: 여러 Judge 간 agreement

References:
- Zheng et al. (2024, NeurIPS) "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
- Fleiss' Kappa for inter-rater agreement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import random


@dataclass
class ConsistencyReport:
    """Judge Consistency 검증 결과 리포트"""
    
    self_consistency_score: float = 0.0
    coefficient_of_variation: float = 0.0
    
    inter_judge_agreement: float = 0.0
    judges_used: List[str] = field(default_factory=list)
    
    total_evaluations: int = 0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    
    per_sample_consistency: List[Dict[str, Any]] = field(default_factory=list)
    disagreement_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "self_consistency_score": self.self_consistency_score,
            "coefficient_of_variation": self.coefficient_of_variation,
            "inter_judge_agreement": self.inter_judge_agreement,
            "judges_used": self.judges_used,
            "total_evaluations": self.total_evaluations,
            "score_distribution": self.score_distribution,
            "disagreement_samples": self.disagreement_samples[:10],
        }
    
    def summary(self) -> str:
        lines = [
            "=== Judge Consistency Report ===",
            f"Total evaluations: {self.total_evaluations}",
            "",
            "[Self-Consistency]",
            f"  Score: {self.self_consistency_score:.4f} (1.0 = perfect)",
            f"  Coefficient of Variation: {self.coefficient_of_variation:.4f} (< 0.1 recommended)",
            "",
            "[Multi-Judge Agreement]",
            f"  Inter-Judge Agreement: {self.inter_judge_agreement:.4f}",
            f"  Judges: {', '.join(self.judges_used) if self.judges_used else 'N/A'}",
            "",
            "[Score Distribution]",
        ]
        
        for score, count in sorted(self.score_distribution.items()):
            lines.append(f"  {score}: {count}")
        
        if self.disagreement_samples:
            lines.append("")
            lines.append("[Top Disagreement Samples]")
            for i, sample in enumerate(self.disagreement_samples[:5]):
                lines.append(f"  {i+1}. ID: {sample.get('id', 'N/A')}")
                lines.append(f"     Score variance: {sample.get('variance', 0):.4f}")
        
        return "\n".join(lines)


class JudgeConsistencyValidator:
    """LLM Judge 일관성 검증기."""
    
    def __init__(
        self,
        judge_fn: Optional[Callable] = None,
        n_trials: int = 3,
        judges: Optional[List[Dict[str, Any]]] = None,
    ):
        self.judge_fn = judge_fn
        self.n_trials = n_trials
        self.judges = judges or []
    
    def measure_self_consistency(
        self, 
        qa_pairs: List[Dict[str, Any]],
        judge_fn: Optional[Callable] = None,
        sample_size: int = 20,
    ) -> ConsistencyReport:
        """Self-consistency 측정: 동일 QA를 여러 번 평가하여 점수 일관성 확인."""
        report = ConsistencyReport()
        judge = judge_fn or self.judge_fn
        
        if not judge:
            report.self_consistency_score = -1
            return report
        
        samples = random.sample(qa_pairs, min(sample_size, len(qa_pairs)))
        
        all_cvs = []
        
        for pair in samples:
            scores = []
            for _ in range(self.n_trials):
                try:
                    score = judge(pair)
                    if isinstance(score, dict):
                        score = score.get("overall_score", 0)
                    scores.append(float(score))
                except Exception:
                    continue
            
            if len(scores) >= 2:
                mean_score = sum(scores) / len(scores)
                std_score = self._std(scores)
                cv = std_score / mean_score if mean_score > 0 else 0
                
                all_cvs.append(cv)
                report.per_sample_consistency.append({
                    "id": pair.get("id", "unknown"),
                    "scores": scores,
                    "mean": mean_score,
                    "std": std_score,
                    "cv": cv,
                })
                
                if cv > 0.15:
                    report.disagreement_samples.append({
                        "id": pair.get("id", "unknown"),
                        "variance": std_score ** 2,
                        "scores": scores,
                    })
        
        report.total_evaluations = len(samples) * self.n_trials
        
        if all_cvs:
            avg_cv = sum(all_cvs) / len(all_cvs)
            report.coefficient_of_variation = avg_cv
            report.self_consistency_score = max(0, 1 - avg_cv)
        
        return report
    
    def measure_inter_judge_agreement(
        self,
        qa_pairs: List[Dict[str, Any]],
        judge_fns: List[Callable],
        judge_names: Optional[List[str]] = None,
        sample_size: int = 30,
    ) -> ConsistencyReport:
        """Multi-judge agreement 측정."""
        report = ConsistencyReport()
        report.judges_used = judge_names or [f"Judge_{i}" for i in range(len(judge_fns))]
        
        if len(judge_fns) < 2:
            report.inter_judge_agreement = -1
            return report
        
        samples = random.sample(qa_pairs, min(sample_size, len(qa_pairs)))
        
        all_scores_per_sample = []
        
        for pair in samples:
            sample_scores = []
            for judge in judge_fns:
                try:
                    score = judge(pair)
                    if isinstance(score, dict):
                        score = score.get("overall_score", 0)
                    sample_scores.append(float(score))
                except Exception:
                    sample_scores.append(None)
            
            valid_scores = [s for s in sample_scores if s is not None]
            if len(valid_scores) >= 2:
                all_scores_per_sample.append(valid_scores)
                
                for s in valid_scores:
                    score_key = f"{int(s)}"
                    report.score_distribution[score_key] = report.score_distribution.get(score_key, 0) + 1
                
                score_range = max(valid_scores) - min(valid_scores)
                if score_range > 1.5:
                    report.disagreement_samples.append({
                        "id": pair.get("id", "unknown"),
                        "scores_by_judge": dict(zip(report.judges_used, sample_scores)),
                        "variance": self._std(valid_scores) ** 2,
                    })
        
        report.total_evaluations = len(samples) * len(judge_fns)
        
        if all_scores_per_sample:
            report.inter_judge_agreement = self._calculate_agreement(all_scores_per_sample)
        
        return report
    
    def _calculate_agreement(self, scores_matrix: List[List[float]]) -> float:
        """Judge 간 agreement 계산 (simplified)."""
        if not scores_matrix:
            return 0.0
        
        variances = []
        for scores in scores_matrix:
            if len(scores) >= 2:
                mean = sum(scores) / len(scores)
                var = sum((s - mean) ** 2 for s in scores) / len(scores)
                variances.append(var)
        
        if not variances:
            return 0.0
        
        avg_variance = sum(variances) / len(variances)
        max_possible_variance = 4.0
        agreement = 1 - (avg_variance / max_possible_variance)
        
        return max(0, min(1, agreement))
    
    def _std(self, values: List[float]) -> float:
        """표준편차 계산"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
