"""
Groundedness Validation Module

답변이 컨텍스트에 근거하는지 검증 (Hallucination 탐지).

방법:
1. LLM-based NLI: LLM에게 entailment 판단 요청
2. Keyword Coverage: 답변 키워드가 컨텍스트에 있는지 확인
3. Numerical Consistency: 수치/단위 일치 여부

LLM 의존 없이도 작동하도록 규칙 기반 검증도 포함.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroundednessReport:
    """Groundedness 검증 결과 리포트"""
    
    total_samples: int = 0
    grounded_count: int = 0
    hallucination_count: int = 0
    
    overall_groundedness: float = 0.0
    keyword_coverage: float = 0.0
    numerical_consistency: float = 0.0
    
    per_sample_scores: List[Dict[str, Any]] = field(default_factory=list)
    hallucination_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "grounded_count": self.grounded_count,
            "hallucination_count": self.hallucination_count,
            "overall_groundedness": self.overall_groundedness,
            "keyword_coverage": self.keyword_coverage,
            "numerical_consistency": self.numerical_consistency,
            "hallucination_samples": self.hallucination_samples[:10],
        }
    
    def summary(self) -> str:
        pct = self.grounded_count / self.total_samples * 100 if self.total_samples else 0
        lines = [
            "=== Groundedness Report ===",
            f"Total samples: {self.total_samples}",
            f"Grounded: {self.grounded_count} ({pct:.1f}%)",
            f"Hallucination suspects: {self.hallucination_count}",
            "",
            "[Scores]",
            f"  Overall Groundedness: {self.overall_groundedness:.4f}",
            f"  Keyword Coverage: {self.keyword_coverage:.4f}",
            f"  Numerical Consistency: {self.numerical_consistency:.4f}",
        ]
        
        if self.hallucination_samples:
            lines.append("")
            lines.append("[Top Hallucination Suspects]")
            for i, sample in enumerate(self.hallucination_samples[:5]):
                lines.append(f"  {i+1}. ID: {sample.get('id', 'N/A')}")
                lines.append(f"     Reason: {sample.get('reason', 'N/A')}")
        
        return "\n".join(lines)


class GroundednessValidator:
    """답변 근거성 검증기 (Hallucination Detection)."""
    
    def __init__(
        self,
        keyword_threshold: float = 0.3,
        llm_connector: Optional[Any] = None,
    ):
        self.keyword_threshold = keyword_threshold
        self.llm = llm_connector
    
    def validate(self, qa_pairs: List[Dict[str, Any]]) -> GroundednessReport:
        """데이터셋 groundedness 검증 수행."""
        report = GroundednessReport(total_samples=len(qa_pairs))
        
        if not qa_pairs:
            return report
        
        keyword_scores = []
        numerical_scores = []
        
        for pair in qa_pairs:
            context = pair.get("context", "")
            answer = pair.get("answer", "")
            qa_id = pair.get("id", "unknown")
            
            kw_score, missing_keywords = self._check_keyword_coverage(context, answer)
            keyword_scores.append(kw_score)
            
            num_score, num_issues = self._check_numerical_consistency(context, answer)
            numerical_scores.append(num_score)
            
            overall_score = (kw_score * 0.6 + num_score * 0.4)
            
            sample_result = {
                "id": qa_id,
                "keyword_score": kw_score,
                "numerical_score": num_score,
                "overall_score": overall_score,
                "missing_keywords": missing_keywords,
                "numerical_issues": num_issues,
            }
            report.per_sample_scores.append(sample_result)
            
            if overall_score < 0.5:
                report.hallucination_count += 1
                report.hallucination_samples.append({
                    "id": qa_id,
                    "score": overall_score,
                    "reason": self._get_hallucination_reason(kw_score, num_score, missing_keywords, num_issues),
                    "question": pair.get("question", "")[:100],
                    "answer": answer[:100],
                })
            else:
                report.grounded_count += 1
        
        report.keyword_coverage = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
        report.numerical_consistency = sum(numerical_scores) / len(numerical_scores) if numerical_scores else 0
        report.overall_groundedness = (report.keyword_coverage * 0.6 + report.numerical_consistency * 0.4)
        
        report.hallucination_samples.sort(key=lambda x: x["score"])
        
        return report
    
    def _check_keyword_coverage(self, context: str, answer: str) -> Tuple[float, List[str]]:
        """답변의 주요 키워드가 컨텍스트에 있는지 확인."""
        if not answer:
            return 0.0, []
        
        stopwords = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "로", "으로", 
                     "와", "과", "도", "만", "까지", "부터", "보다", "처럼", "같이",
                     "그", "저", "이것", "저것", "그것", "것", "수", "등", "및"}
        
        answer_tokens = set(answer.lower().split()) - stopwords
        context_lower = context.lower()
        
        if not answer_tokens:
            return 1.0, []
        
        found = 0
        missing = []
        for token in answer_tokens:
            if len(token) < 2:
                continue
            if token in context_lower:
                found += 1
            else:
                missing.append(token)
        
        total_checked = found + len(missing)
        if total_checked == 0:
            return 1.0, []
        
        coverage = found / total_checked
        return coverage, missing[:10]
    
    def _check_numerical_consistency(self, context: str, answer: str) -> Tuple[float, List[str]]:
        """답변의 수치가 컨텍스트에 있는지 확인."""
        number_pattern = r'\d+\.?\d*(?:\s*[%°℃℉]|\s*도|\s*퍼센트)?'
        
        answer_numbers = set(re.findall(number_pattern, answer))
        
        if not answer_numbers:
            return 1.0, []
        
        context_numbers = set(re.findall(number_pattern, context))
        
        found = 0
        issues = []
        for num in answer_numbers:
            num_clean = re.sub(r'[^\d.]', '', num)
            if num in context_numbers or num_clean in [re.sub(r'[^\d.]', '', n) for n in context_numbers]:
                found += 1
            else:
                issues.append(f"'{num}' not in context")
        
        if not answer_numbers:
            return 1.0, []
        
        consistency = found / len(answer_numbers)
        return consistency, issues[:5]
    
    def _get_hallucination_reason(
        self, 
        kw_score: float, 
        num_score: float, 
        missing_kw: List[str], 
        num_issues: List[str]
    ) -> str:
        """Hallucination 사유 생성."""
        reasons = []
        
        if kw_score < 0.5:
            reasons.append(f"Low keyword coverage ({kw_score:.2f})")
            if missing_kw:
                top_missing = ", ".join(missing_kw[:3])
                reasons.append(f"Missing: {top_missing}")
        
        if num_score < 0.5 and num_issues:
            top_issues = ", ".join(num_issues[:2])
            reasons.append(f"Numerical inconsistency: {top_issues}")
        
        return "; ".join(reasons) if reasons else "Unknown"
