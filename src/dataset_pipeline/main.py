"""
스마트팜 QA 데이터셋 구축 파이프라인 메인 모듈

RAFT (Zhang et al., 2024) 워크플로우 구현:
1. 문서 로드 및 인덱싱 (RAG)
2. 질문 생성 (Self-Instruct + Evol-Instruct)
3. 답변 생성 (RAG 기반)
4. 품질 평가 (LLM-as-a-Judge)
5. 반복적 정제 및 필터링
"""

import os
import json
import yaml
from typing import Dict, Any, List
from pathlib import Path

from .llm_connector import LLMConnector
from .rag_connector import RAGConnector
from .generator import QuestionGenerator
from .judge import LLMJudge
from .refiner import AnswerRefiner


class DatasetPipeline:
    """
    스마트팜 QA 데이터셋 구축 파이프라인
    
    연구 방법론을 통합하여 고품질 데이터셋을 생성합니다.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self._setup_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        config_file = Path(__file__).parent.parent / config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_components(self):
        """파이프라인 컴포넌트 초기화"""
        # LLM 커넥터 (generator/judge 역할 모두 포함)
        # NOTE: LLMConnector는 settings.yaml의 llm 섹션 전체( generator/judge 포함 )를 기대합니다.
        self.llm = LLMConnector(self.config["llm"])
        # 하위 호환/가독성용 별칭
        self.llm_generator = self.llm
        self.llm_judge = self.llm
        
        # RAG 커넥터
        self.rag = RAGConnector(self.config["rag"])
        
        # 질문 생성기
        generator_config = {
            "num_questions": 100,
            "max_iterations": self.config["pipeline"]["max_iterations"],
            "temperature": self.config["llm"]["generator"]["temperature"],
            "seed_questions": self.config["domain"]["seed_questions"],
        }
        self.generator = QuestionGenerator(self.llm_generator, generator_config)
        
        # 판정관
        judge_config = {
            "evaluation_criteria": [c["description"] for c in self.config["pipeline"]["evaluation_criteria"]],
            "score_range": (1, 5),
            "temperature": self.config["llm"]["judge"]["temperature"],
        }
        self.judge = LLMJudge(self.llm_judge, judge_config)
        
        # 답변 정제기
        refiner_config = {
            "max_iterations": self.config["pipeline"]["max_iterations"],
            "temperature": self.config["llm"]["generator"]["temperature"],
            "use_rag": True,
            "min_answer_length": 50,
        }
        self.refiner = AnswerRefiner(self.llm_generator, self.rag, refiner_config)
    
    def build_dataset(self, document_path: str, output_path: str = "dataset.jsonl") -> str:
        """
        전체 데이터셋 구축 파이프라인 실행
        
        Args:
            document_path: 문서 경로 (파일 또는 디렉토리)
            output_path: 출력 파일 경로
        
        Returns:
            출력 파일 경로
        """
        print("=== 스마트팜 QA 데이터셋 구축 시작 ===")
        
        # 1. 문서 로드 및 인덱싱
        print("1. 문서 로드 및 인덱싱...")
        chunks = self.rag.load_documents(document_path)
        self.rag.index_documents(chunks)
        print(f"   {len(chunks)}개 청크 인덱싱 완료")
        
        # 2. 질문 생성
        print("2. 질문 생성...")
        contexts = [chunk["content"] for chunk in self.rag.get_all_chunks()]
        qa_pairs = self.generator.generate_dataset(contexts)
        print(f"   {len(qa_pairs)}개 질문 생성 완료")
        
        # 3. 답변 생성
        print("3. 답변 생성...")
        questions = [pair["question"] for pair in qa_pairs]
        answers_data = self.refiner.generate_answers_batch(questions)
        
        # QA 쌍 업데이트
        for i, pair in enumerate(qa_pairs):
            pair["answer"] = answers_data[i]["answer"]
            pair["confidence"] = answers_data[i]["confidence"]
            pair["sources"] = answers_data[i]["sources"]
        
        print(f"   {len(qa_pairs)}개 답변 생성 완료")
        
        # 4. 품질 평가
        print("4. 품질 평가...")
        evaluated_pairs = self.judge.evaluate_batch(qa_pairs)
        print(f"   {len(evaluated_pairs)}개 QA 쌍 평가 완료")
        
        # 5. 고품질 데이터 필터링
        print("5. 고품질 데이터 필터링...")
        high_quality_pairs = self.judge.filter_high_quality(evaluated_pairs, threshold=3.5)
        print(f"   {len(high_quality_pairs)}개 고품질 QA 쌍 필터링 완료")
        
        # 6. 결과 저장
        print("6. 결과 저장...")
        self._save_dataset(high_quality_pairs, output_path)
        print(f"   데이터셋 저장 완료: {output_path}")
        
        print("=== 데이터셋 구축 완료 ===")
        return output_path
    
    def _save_dataset(self, qa_pairs: List[Dict[str, Any]], output_path: str):
        """데이터셋 저장 (JSONL 형식)"""
        output_file = Path(__file__).parent.parent / output_path
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in qa_pairs:
                json_line = {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "difficulty": pair.get("difficulty", "medium"),
                    "confidence": pair.get("confidence", 0.5),
                    "evaluation_score": pair.get("evaluation", {}).get("overall_score", 0),
                    "sources": pair.get("sources", []),
                    "metadata": {
                        "iteration": pair.get("iteration", 1),
                        "context_length": len(pair.get("context", "")),
                    }
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    
    def evaluate_existing_dataset(self, dataset_path: str, output_path: str = "evaluation_results.json"):
        """
        기존 데이터셋 평가
        
        Args:
            dataset_path: 평가할 데이터셋 파일 경로
            output_path: 평가 결과 출력 경로
        """
        print("=== 기존 데이터셋 평가 시작 ===")
        
        # 데이터셋 로드
        qa_pairs = self._load_dataset(dataset_path)
        print(f"   {len(qa_pairs)}개 QA 쌍 로드 완료")
        
        # 평가 실행
        evaluated_pairs = self.judge.evaluate_batch(qa_pairs)
        
        # 결과 저장
        self._save_evaluation_results(evaluated_pairs, output_path)
        print(f"   평가 결과 저장 완료: {output_path}")
        
        print("=== 데이터셋 평가 완료 ===")
    
    def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """데이터셋 로드"""
        dataset_file = Path(__file__).parent.parent / dataset_path
        qa_pairs = []
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line.strip()))
        
        return qa_pairs
    
    def _save_evaluation_results(self, evaluated_pairs: List[Dict[str, Any]], output_path: str):
        """평가 결과 저장"""
        output_file = Path(__file__).parent.parent / output_path
        
        results = {
            "total_pairs": len(evaluated_pairs),
            "average_score": sum(p["evaluation"]["overall_score"] for p in evaluated_pairs) / len(evaluated_pairs),
            "high_quality_count": len([p for p in evaluated_pairs if p["evaluation"]["overall_score"] >= 3.5]),
            "details": evaluated_pairs,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="스마트팜 QA 데이터셋 구축 파이프라인")
    parser.add_argument("--config", default="config/settings.yaml", help="설정 파일 경로")
    parser.add_argument("--documents", required=True, help="문서 경로 (파일 또는 디렉토리)")
    parser.add_argument("--output", default="smartfarm_qa_dataset.jsonl", help="출력 파일 경로")
    parser.add_argument("--evaluate", help="기존 데이터셋 평가 모드")
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = DatasetPipeline(args.config)
    
    if args.evaluate:
        # 평가 모드
        pipeline.evaluate_existing_dataset(args.evaluate, f"{args.evaluate}_evaluation.json")
    else:
        # 구축 모드
        pipeline.build_dataset(args.documents, args.output)


if __name__ == "__main__":
    main()
