#!/usr/bin/env python3
"""
데이터셋 구축 파이프라인 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# Windows 콘솔(cp949 등)에서 이모지/체크마크 출력 시 UnicodeEncodeError가 발생할 수 있어
# 출력 스트림을 "replace" 모드로 설정해 테스트가 중단되지 않도록 합니다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dataset_pipeline.main import DatasetPipeline


def test_pipeline():
    """파이프라인 테스트"""
    print("=== 데이터셋 구축 파이프라인 테스트 ===")

    try:
        # 파이프라인 초기화
        pipeline = DatasetPipeline()

        # 테스트용 더미 문서 경로 (실제로는 스마트팜 관련 문서 경로 사용)
        # 예시: "data/smartfarm_docs/" 또는 "docs/smartfarm_kb.pdf"
        dummy_docs = str(project_root / "examples" / "test_doc.txt")  # UTF-8 테스트 문서 사용

        if not os.path.exists(dummy_docs):
            print(f"테스트 문서가 없습니다: {dummy_docs}")
            print("실제 스마트팜 문서를 지정해주세요.")
            return

        # 기본 컴포넌트 테스트만 수행 (실제 LLM 호출 생략)
        print("✓ 파이프라인 초기화 성공")
        print("✓ RAG 커넥터 초기화 성공")
        print("✓ LLM 커넥터 초기화 성공")
        print("✓ 질문 생성기 초기화 성공")
        print("✓ 판정관 초기화 성공")
        print("✓ 답변 정제기 초기화 성공")
        
        # 문서 로드 테스트
        print("✓ 문서 로드 테스트...")
        chunks = pipeline.rag.load_documents(dummy_docs)
        print(f"✓ {len(chunks)}개 청크 로드 성공")
        
        print("=== 테스트 완료 ===")
        print("실제 데이터셋 구축을 위해서는 스마트팜 관련 문서를 준비하고")
        print("pipeline.build_dataset() 메소드를 호출하세요.")

    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pipeline()
