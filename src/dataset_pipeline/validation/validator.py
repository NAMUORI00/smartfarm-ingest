"""
Unified Dataset Validator

모든 검증 모듈을 통합하여 종합 검증 보고서 생성.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .diversity_metrics import DiversityValidator, DiversityReport
from .groundedness import GroundednessValidator, GroundednessReport
from .judge_consistency import JudgeConsistencyValidator, ConsistencyReport


@dataclass
class ValidationReport:
    """종합 검증 보고서"""
    
    dataset_name: str = ""
    validation_date: str = ""
    total_samples: int = 0
    
    diversity: Optional[DiversityReport] = None
    groundedness: Optional[GroundednessReport] = None
    consistency: Optional[ConsistencyReport] = None
    
    overall_quality_score: float = 0.0
    
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "dataset_name": self.dataset_name,
                "validation_date": self.validation_date,
                "total_samples": self.total_samples,
            },
            "diversity": self.diversity.to_dict() if self.diversity else None,
            "groundedness": self.groundedness.to_dict() if self.groundedness else None,
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "overall_quality_score": self.overall_quality_score,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATASET VALIDATION REPORT",
            "=" * 60,
            f"Dataset: {self.dataset_name}",
            f"Date: {self.validation_date}",
            f"Samples: {self.total_samples}",
            "",
            f"OVERALL QUALITY SCORE: {self.overall_quality_score:.2f} / 1.00",
            "=" * 60,
        ]
        
        if self.diversity:
            lines.append("")
            lines.append(self.diversity.summary())
        
        if self.groundedness:
            lines.append("")
            lines.append(self.groundedness.summary())
        
        if self.consistency:
            lines.append("")
            lines.append(self.consistency.summary())
        
        if self.warnings:
            lines.append("")
            lines.append("=== WARNINGS ===")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        
        if self.recommendations:
            lines.append("")
            lines.append("=== RECOMMENDATIONS ===")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """리포트 저장"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(output) + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        with open(str(output) + '.txt', 'w', encoding='utf-8') as f:
            f.write(self.summary())


class DatasetValidator:
    """
    통합 데이터셋 검증기.
    
    Usage:
        validator = DatasetValidator()
        report = validator.validate(qa_pairs, dataset_name="wasabi_qa")
        report.save("validation_report")
    """
    
    def __init__(
        self,
        diversity_threshold: float = 0.7,
        groundedness_threshold: float = 0.3,
        judge_fn: Optional[Any] = None,
    ):
        self.diversity_validator = DiversityValidator(threshold=diversity_threshold)
        self.groundedness_validator = GroundednessValidator(keyword_threshold=groundedness_threshold)
        self.consistency_validator = JudgeConsistencyValidator(judge_fn=judge_fn)
    
    def validate(
        self,
        qa_pairs: List[Dict[str, Any]],
        dataset_name: str = "unnamed",
        skip_consistency: bool = True,
    ) -> ValidationReport:
        """종합 검증 수행."""
        report = ValidationReport(
            dataset_name=dataset_name,
            validation_date=datetime.now().isoformat(),
            total_samples=len(qa_pairs),
        )
        
        report.diversity = self.diversity_validator.validate(qa_pairs)
        report.groundedness = self.groundedness_validator.validate(qa_pairs)
        
        if not skip_consistency and self.consistency_validator.judge_fn:
            report.consistency = self.consistency_validator.measure_self_consistency(
                qa_pairs, sample_size=min(20, len(qa_pairs))
            )
        
        report.overall_quality_score = self._calculate_overall_score(report)
        report.recommendations, report.warnings = self._generate_recommendations(report)
        
        return report
    
    def _calculate_overall_score(self, report: ValidationReport) -> float:
        """종합 점수 계산 (가중 평균)"""
        scores = []
        weights = []
        
        if report.diversity:
            scores.append(report.diversity.rouge_l_diversity)
            weights.append(0.3)
        
        if report.groundedness:
            scores.append(report.groundedness.overall_groundedness)
            weights.append(0.5)
        
        if report.consistency and report.consistency.self_consistency_score >= 0:
            scores.append(report.consistency.self_consistency_score)
            weights.append(0.2)
        
        if not scores:
            return 0.0
        
        total_weight = sum(weights[:len(scores)])
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, report: ValidationReport) -> tuple:
        """권장사항 및 경고 생성"""
        recommendations = []
        warnings = []
        
        if report.diversity:
            if report.diversity.rouge_l_diversity < 0.7:
                warnings.append(f"Low diversity score ({report.diversity.rouge_l_diversity:.2f}). Consider filtering similar questions.")
            
            if report.diversity.category_distribution:
                counts = list(report.diversity.category_distribution.values())
                if counts and max(counts) > sum(counts) * 0.5:
                    warnings.append("Category imbalance detected. One category dominates > 50%.")
                    recommendations.append("Generate more questions from underrepresented categories.")
        
        if report.groundedness:
            if report.groundedness.overall_groundedness < 0.7:
                warnings.append(f"Low groundedness score ({report.groundedness.overall_groundedness:.2f}). Hallucination risk.")
                recommendations.append("Review answers with low keyword coverage.")
            
            if report.groundedness.hallucination_count > report.groundedness.total_samples * 0.1:
                warnings.append(f"{report.groundedness.hallucination_count} potential hallucinations detected (>10%).")
        
        if report.consistency and report.consistency.coefficient_of_variation > 0.15:
            warnings.append(f"High judge variance (CV={report.consistency.coefficient_of_variation:.2f}). Consider using multiple judges.")
        
        if not recommendations:
            recommendations.append("Dataset quality looks good. Consider human evaluation for final verification.")
        
        return recommendations, warnings
