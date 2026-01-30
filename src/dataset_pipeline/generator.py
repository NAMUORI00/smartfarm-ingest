"""
질문 생성기 모듈

Self-Instruct (Wang et al., 2022) 및 Evol-Instruct (Xu et al., 2023) 방법론 기반:
- 시드 질문으로부터 새로운 질문 생성
- 진화적 복잡성 증가
- 도메인 특화 질문 생성
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from .constants import Defaults
from .json_utils import try_parse_json
from .llm_connector import LLMConnector

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@dataclass
class GeneratorConfig:
    """생성기 설정 데이터 클래스"""
    num_questions: int = Defaults.NUM_QUESTIONS
    max_iterations: int = 3
    temperature: float = 0.7
    seed_questions: List[str] = None
    
    def __post_init__(self):
        if self.seed_questions is None:
            self.seed_questions = [
                "스마트팜에서 토마토 재배 시 최적의 온도는 얼마인가?",
                "스마트팜의 자동 관수 시스템은 어떻게 작동하는가?",
                "스마트팜에서 병충해를 예방하는 방법은 무엇인가?",
            ]


class QuestionGenerator:
    """
    스마트팜 도메인 질문 생성기
    
    Self-Instruct 및 Evol-Instruct 방법론을 적용하여
    고품질 QA 쌍을 생성합니다.
    """
    
    def __init__(self, llm_connector: LLMConnector, config: Dict[str, Any]):
        """
        Args:
            llm_connector: LLM 커넥터 인스턴스
            config: settings.yaml의 generator 섹션
        """
        self.llm = llm_connector
        self.config = GeneratorConfig(
            num_questions=config.get("num_questions", 100),
            max_iterations=config.get("max_iterations", 3),
            temperature=config.get("temperature", 0.7),
            seed_questions=config.get("seed_questions", []),
        )
        
        # 프롬프트 템플릿 로드
        self.generation_prompt = self._load_generation_prompt()
        self.evolution_prompt = self._load_evolution_prompt()
        self.generation_template = Template(self.generation_prompt)
        self.evolution_template = Template(self.evolution_prompt)
    
    def _load_generation_prompt(self) -> str:
        """질문 생성 프롬프트 로드"""
        prompt_file = _PROMPTS_DIR / "generation.jinja"
        return prompt_file.read_text(encoding="utf-8")
    
    def _load_evolution_prompt(self) -> str:
        """질문 진화 프롬프트 로드"""
        prompt_file = _PROMPTS_DIR / "evolution.jinja"
        return prompt_file.read_text(encoding="utf-8")
    
    def generate_from_context(self, context: str, existing_questions: List[str] = None) -> List[Dict[str, Any]]:
        """
        컨텍스트로부터 질문 생성
        
        Args:
            context: RAG에서 검색된 컨텍스트
            existing_questions: 이미 생성된 질문들
        
        Returns:
            생성된 질문 리스트
        """
        if existing_questions is None:
            existing_questions = []
        
        prompt = self.generation_template.render(
            context=context,
            existing_questions="\n".join(existing_questions[-10:]),  # 최근 10개만
        )
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=1000,
        )
        
        parsed = try_parse_json(response)
        if isinstance(parsed, list):
            normalized: list[dict[str, Any]] = []
            for item in parsed:
                if isinstance(item, dict) and item.get("question"):
                    normalized.append({
                        "question": str(item["question"]).strip(),
                        "difficulty": item.get("difficulty", "medium"),
                    })
                elif isinstance(item, str) and item.strip():
                    normalized.append({"question": item.strip(), "difficulty": "medium"})
            if normalized:
                return normalized

        # JSON 파싱 실패 시에도, JSON 조각(예: ```json ... ```)에서 질문만 추출 시도
        q_with_diff = re.findall(
            r'\{\s*"question"\s*:\s*"([^"]+)"\s*,\s*"difficulty"\s*:\s*"([^"]+)"\s*\}',
            response,
            flags=re.DOTALL,
        )
        if q_with_diff:
            return [{"question": q.strip(), "difficulty": d.strip()} for q, d in q_with_diff if q.strip()]

        q_only = re.findall(r'"question"\s*:\s*"([^"]+)"', response)
        if q_only:
            return [{"question": q.strip(), "difficulty": "medium"} for q in q_only if q.strip()]

        # 최후 폴백: 라인 기반
        questions: list[dict[str, Any]] = []
        for line in response.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("```"):
                continue
            if s in {"[", "]", "{", "}", "..."}:
                continue
            if s.startswith(("-", "*")):
                s = s.lstrip("-*").strip()
            if not s or s in {"json", "JSON"}:
                continue
            questions.append({"question": s, "difficulty": "medium"})
        return questions
    
    def evolve_question(self, question: str) -> Dict[str, Any]:
        """
        질문 진화 (복잡성 증가)
        
        Args:
            question: 원본 질문
        
        Returns:
            진화된 질문 정보
        """
        prompt = self.evolution_template.render(question=question)
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=500,
        )
        
        parsed = try_parse_json(response)
        if isinstance(parsed, dict) and parsed.get("evolved_question"):
            return parsed

        return {
            "evolved_question": question + " (진화 실패)",
            "evolution_type": "원본 유지"
        }
    
    def generate_dataset(self, contexts: List[str], seed_questions: List[str] = None) -> List[Dict[str, Any]]:
        """
        전체 데이터셋 생성
        
        Args:
            contexts: RAG 컨텍스트 리스트
            seed_questions: 시드 질문들
        
        Returns:
            QA 쌍 리스트
        """
        if seed_questions is None:
            seed_questions = self.config.seed_questions.copy()
        
        all_questions = seed_questions.copy()
        qa_pairs = []
        
        # 반복적 생성
        for iteration in range(self.config.max_iterations):
            print(f"반복 {iteration + 1}/{self.config.max_iterations}")
            
            # 각 컨텍스트에서 질문 생성
            for context in contexts:
                if len(all_questions) >= self.config.num_questions:
                    break
                
                new_questions = self.generate_from_context(context, all_questions)
                
                for q_data in new_questions:
                    question = q_data["question"]
                    
                    # 중복 체크
                    if question not in all_questions:
                        all_questions.append(question)
                        
                        # 진화 적용 (일부 질문만)
                        if random.random() < Defaults.EVOLUTION_PROBABILITY:  # 진화 확률
                            evolved = self.evolve_question(question)
                            question = evolved["evolved_question"]
                        
                        qa_pairs.append({
                            "question": question,
                            "answer": "",  # 나중에 생성
                            "difficulty": q_data.get("difficulty", "medium"),
                            "context": context,
                            "iteration": iteration + 1,
                        })
                    
                    if len(qa_pairs) >= self.config.num_questions:
                        break
                
                if len(qa_pairs) >= self.config.num_questions:
                    break
        
        return qa_pairs[:self.config.num_questions]
