"""
LLM-as-a-Judge 모듈

Prometheus (Kim et al., 2024) 방법론 기반:
- 절대적 평가 기준
- 단계별 평가 프로세스
- 피드백 기반 개선
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from .llm_connector import LLMConnector
from .json_utils import try_parse_json

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

@dataclass
class JudgeConfig:
    """판정관 설정 데이터 클래스"""
    evaluation_criteria: List[str] = None
    score_range: tuple = (1, 5)
    temperature: float = 0.3
    
    def __post_init__(self):
        if self.evaluation_criteria is None:
            self.evaluation_criteria = [
                "정확성: 답변이 사실적으로 정확한가?",
                "완전성: 답변이 질문을 완전히 해결하는가?",
                "관련성: 답변이 스마트팜 도메인과 관련되는가?",
                "명확성: 답변이 명확하고 이해하기 쉬운가?",
                "유용성: 답변이 실용적이고 적용 가능한가?",
            ]


class LLMJudge:
    """
    LLM 기반 판정관
    
    Prometheus 방법론을 적용하여 QA 쌍의 품질을 평가합니다.
    """
    
    def __init__(self, llm_connector: LLMConnector, config: Dict[str, Any]):
        """
        Args:
            llm_connector: LLM 커넥터 인스턴스
            config: settings.yaml의 judge 섹션
        """
        self.llm = llm_connector
        self.config = JudgeConfig(
            evaluation_criteria=config.get("evaluation_criteria", []),
            score_range=config.get("score_range", (1, 5)),
            temperature=config.get("temperature", 0.3),
        )
        
        # 프롬프트 템플릿 로드
        self.judge_prompt = self._load_judge_prompt()
        self.feedback_prompt = self._load_feedback_prompt()
        self.judge_template = Template(self.judge_prompt)
        self.feedback_template = Template(self.feedback_prompt)
    
    def _load_judge_prompt(self) -> str:
        """평가 프롬프트 로드"""
        prompt_file = _PROMPTS_DIR / "judge.jinja"
        return prompt_file.read_text(encoding="utf-8")
    
    def _load_feedback_prompt(self) -> str:
        """피드백 프롬프트 로드"""
        prompt_file = _PROMPTS_DIR / "refine.jinja"
        return prompt_file.read_text(encoding="utf-8")
    
    def evaluate_qa_pair(self, question: str, answer: str, context: str = "") -> Dict[str, Any]:
        """
        QA 쌍 평가
        
        Args:
            question: 질문
            answer: 답변
            context: 관련 컨텍스트
        
        Returns:
            평가 결과
        """
        prompt = self.judge_template.render(
            criteria=self.config.evaluation_criteria,
            min_score=self.config.score_range[0],
            max_score=self.config.score_range[1],
            question=question,
            answer=answer,
            context=context,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            role="judge",
            temperature=self.config.temperature,
            max_tokens=1000,
        )
        
        parsed = try_parse_json(response)
        if isinstance(parsed, dict):
            return parsed

        # JSON 파싱 실패 시 기본 평가 반환
        return {
            "criteria_scores": {},
            "overall_score": 3.0,
            "feedback": "평가 파싱 실패",
        }
    
    def evaluate_batch(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        배치 평가
        
        Args:
            qa_pairs: QA 쌍 리스트
        
        Returns:
            평가 결과가 추가된 QA 쌍 리스트
        """
        evaluated_pairs = []
        
        for qa_pair in qa_pairs:
            evaluation = self.evaluate_qa_pair(
                question=qa_pair["question"],
                answer=qa_pair["answer"],
                context=qa_pair.get("context", ""),
            )
            
            qa_pair_with_eval = qa_pair.copy()
            qa_pair_with_eval["evaluation"] = evaluation
            evaluated_pairs.append(qa_pair_with_eval)
        
        return evaluated_pairs
    
    def generate_feedback(self, question: str, answer: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        개선 피드백 생성
        
        Args:
            question: 질문
            answer: 답변
            evaluation: 평가 결과
        
        Returns:
            개선 제안
        """
        prompt = self.feedback_template.render(
            question=question,
            current_answer=answer,
            feedback=json.dumps(evaluation, ensure_ascii=False, indent=2),
        )
        
        response = self.llm.generate(
            prompt=prompt,
            role="judge",
            temperature=self.config.temperature,
            max_tokens=800,
        )
        
        parsed = try_parse_json(response)
        if isinstance(parsed, dict):
            return parsed

        return {
            "improved_answer": answer,
            "improvement_reason": "피드백 생성 실패",
        }
    
    def filter_high_quality(self, evaluated_pairs: List[Dict[str, Any]], threshold: float = 3.5) -> List[Dict[str, Any]]:
        """
        고품질 QA 쌍 필터링
        
        Args:
            evaluated_pairs: 평가된 QA 쌍 리스트
            threshold: 품질 임계값
        
        Returns:
            고품질 QA 쌍만
        """
        high_quality = []
        
        for pair in evaluated_pairs:
            overall_score = pair["evaluation"].get("overall_score", 0)
            if overall_score >= threshold:
                high_quality.append(pair)
        
        return high_quality
