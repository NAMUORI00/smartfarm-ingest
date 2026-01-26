"""
답변 정제기 모듈

RAFT (Zhang et al., 2024) 방법론 기반:
- RAG 컨텍스트를 활용한 답변 생성
- 반복적 정제 및 개선
- 품질 검증
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from .llm_connector import LLMConnector
from .rag_connector import RAGConnector
from .json_utils import try_parse_json


@dataclass
class RefinerConfig:
    """정제기 설정 데이터 클래스"""
    max_iterations: int = 3
    temperature: float = 0.5
    use_rag: bool = True
    min_answer_length: int = 50


class AnswerRefiner:
    """
    답변 정제기
    
    RAG 컨텍스트를 활용하여 질문에 대한 고품질 답변을 생성합니다.
    """
    
    def __init__(self, llm_connector: LLMConnector, rag_connector: Optional[RAGConnector], config: Dict[str, Any]):
        """
        Args:
            llm_connector: LLM 커넥터 인스턴스
            rag_connector: RAG 커넥터 인스턴스 (선택)
            config: settings.yaml의 refiner 섹션
        """
        self.llm = llm_connector
        self.rag = rag_connector
        self.config = RefinerConfig(
            max_iterations=config.get("max_iterations", 3),
            temperature=config.get("temperature", 0.5),
            use_rag=config.get("use_rag", True),
            min_answer_length=config.get("min_answer_length", 50),
        )
        
        # 프롬프트 템플릿 로드
        self.answer_prompt = self._load_answer_prompt()
        self.refine_prompt = self._load_refine_prompt()
        self.answer_template = Template(self.answer_prompt)
        self.refine_template = Template(self.refine_prompt)
    
    def _load_answer_prompt(self) -> str:
        """답변 생성 프롬프트 로드"""
        prompt_file = Path(__file__).parent.parent / "prompts" / "answer.jinja"
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_refine_prompt(self) -> str:
        """답변 정제 프롬프트 로드"""
        prompt_file = Path(__file__).parent.parent / "prompts" / "refine.jinja"
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_answer(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 질문
            context: 관련 컨텍스트 (RAG에서 검색된)
        
        Returns:
            답변 정보
        """
        # RAG 컨텍스트 검색 (활성화된 경우)
        if self.config.use_rag and self.rag:
            retrieved_contexts = self.rag.retrieve(question, top_k=3)
            context += "\n\n".join([c["content"] for c in retrieved_contexts])
        
        prompt = self.answer_template.render(
            question=question,
            context=context,
            min_length=self.config.min_answer_length,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=1500,
        )
        
        parsed = try_parse_json(response)
        if isinstance(parsed, dict) and parsed.get("answer"):
            return parsed

        # JSON 파싱 실패 시 기본 형식 반환
        return {
            "answer": response.strip(),
            "confidence": 0.5,
            "sources": [],
        }
    
    def refine_answer(self, question: str, current_answer: str, feedback: str) -> Dict[str, Any]:
        """
        답변 정제
        
        Args:
            question: 질문
            current_answer: 현재 답변
            feedback: 개선 피드백
        
        Returns:
            정제된 답변
        """
        prompt = self.refine_template.render(
            question=question,
            current_answer=current_answer,
            feedback=feedback,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=1200,
        )
        
        parsed = try_parse_json(response)
        if isinstance(parsed, dict) and parsed.get("refined_answer"):
            return parsed

        return {
            "refined_answer": current_answer,
            "improvements": ["정제 실패"],
            "confidence": 0.5,
        }
    
    def generate_answers_batch(self, questions: List[str], contexts: List[str] = None) -> List[Dict[str, Any]]:
        """
        배치 답변 생성
        
        Args:
            questions: 질문 리스트
            contexts: 컨텍스트 리스트 (질문과 1:1 매핑)
        
        Returns:
            답변 리스트
        """
        if contexts is None:
            contexts = [""] * len(questions)
        
        answers = []
        for question, context in zip(questions, contexts):
            answer_data = self.generate_answer(question, context)
            answers.append({
                "question": question,
                "answer": answer_data["answer"],
                "confidence": answer_data.get("confidence", 0.5),
                "sources": answer_data.get("sources", []),
                "context": context,
            })
        
        return answers
    
    def refine_answers_batch(self, qa_pairs: List[Dict[str, Any]], feedbacks: List[str]) -> List[Dict[str, Any]]:
        """
        배치 답변 정제
        
        Args:
            qa_pairs: QA 쌍 리스트
            feedbacks: 피드백 리스트
        
        Returns:
            정제된 QA 쌍 리스트
        """
        refined_pairs = []
        
        for qa_pair, feedback in zip(qa_pairs, feedbacks):
            refined_data = self.refine_answer(
                question=qa_pair["question"],
                current_answer=qa_pair["answer"],
                feedback=feedback,
            )
            
            refined_pair = qa_pair.copy()
            refined_pair["answer"] = refined_data["refined_answer"]
            refined_pair["confidence"] = refined_data.get("confidence", qa_pair.get("confidence", 0.5))
            refined_pair["refinements"] = refined_data.get("improvements", [])
            
            refined_pairs.append(refined_pair)
        
        return refined_pairs
