# RAG 기반 LLM-as-a-Judge 데이터셋 구축 시스템

스마트팜 도메인을 위한 고품질 QA 데이터셋 구축 파이프라인.  
**Self-Instruct**, **Evol-Instruct**, **RAFT**, **LLM-as-a-Judge** 등 검증된 연구 방법론을 계승.

## 참고 연구

| 연구 | 출처 | 핵심 기법 | 본 시스템 적용 |
|------|------|----------|---------------|
| **Self-Instruct** | Wang et al., 2023, ACL | Seed 기반 instruction 자동 생성 | 질문 생성 프롬프트 구조 |
| **Evol-Instruct** | Xu et al., 2023, ICLR | 난이도 점진적 진화 | 질문 복잡도 단계적 확장 |
| **RAFT** | Zhang et al., 2024, COLM | RAG + Fine-tuning 데이터셋 | 문서 컨텍스트 기반 QA 생성 |
| **LLM-as-a-Judge** | Zheng et al., 2024, NeurIPS | SOTA 모델 평가 체계 | Judge 프롬프트 및 루브릭 |
| **Prometheus** | Kim et al., 2024, NeurIPS | 세분화된 평가 기준 | 다차원 점수 체계 |

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Question Generation (Self-Instruct + RAFT)             │
│  - Seed questions에서 다양한 질문 자동 생성                      │
│  - RAG 컨텍스트 기반으로 도메인 특화 질문 확장                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: Answer Generation (RAFT)                               │
│  - 검색된 문서 컨텍스트 + 질문 → 초기 답변 생성                  │
│  - Chain-of-Thought 추론 과정 포함                              │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: Judge Evaluation (LLM-as-a-Judge + Prometheus)         │
│  - Groundedness, Accuracy, Completeness 다차원 평가             │
│  - 구체적 피드백 및 개선점 제시                                  │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: Iterative Refinement (Evol-Instruct)                   │
│  - 피드백 기반 답변 개선                                         │
│  - 임계값 도달까지 반복 (최대 N회)                               │
└─────────────────────────────────────────────────────────────────┘
```

## 빠른 시작

### v2 Public Artifact Build (Docling + Kimi + Qdrant/FalkorDB)

```bash
cd smartfarm-ingest
python -m pipeline.public_ingest_runner \
  --input-dir ./data/public_docs \
  --qdrant-host localhost --qdrant-port 6333 \
  --falkor-host localhost --falkor-port 6379
```

설명:
- Docling 사용 가능 시 멀티모달 파싱 우선
- Kimi 추출 결과는 스키마 검증 후 반영
- 결과는 Qdrant(벡터) + FalkorDB(KG)에 public tier로 적재

### v2 Reproducibility Check (Offline Ingest Manifest)

```bash
cd smartfarm-ingest
python -m pipeline.reproducibility_check \
  --input-dir ./data/public_docs \
  --out ../output/repro/public_ingest_manifest.json
```

비교 검증:

```bash
python -m pipeline.reproducibility_check \
  --input-dir ./data/public_docs \
  --compare-manifest ../output/repro/public_ingest_manifest.json
```

### 1. 설치

```bash
cd smartfarm-ingest
pip install -r requirements.txt
pip install -e .  # 개발 모드 설치
```

### 2. 설정

**방법 A: 환경 변수 (권장)**
```bash
export API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export HF_TOKEN="your-hf-token"  # CGIAR 데이터셋용 (선택)
```

**방법 B: secrets.yaml 사용**
```bash
# 템플릿 복사
cp config/secrets.yaml.example config/secrets.yaml

# secrets.yaml 편집하여 API 키 입력
```

### 3. 설정 확인

```bash
cd src
python -m dataset_pipeline config
```

## CLI 사용법

통합 CLI를 통해 모든 파이프라인 기능을 사용할 수 있습니다.

```bash
cd src
python -m dataset_pipeline --help
```

### 주요 명령어

#### 웹 크롤링 (와사비)
```bash
python -m dataset_pipeline crawl-wasabi \
  --output ../output/wasabi_web_en.jsonl \
  --limit 100
```

#### EN→KO 번역
```bash
python -m dataset_pipeline translate \
  --input ../output/wasabi_web_en.jsonl \
  --output ../output/wasabi_en_ko.jsonl \
  --resume  # 중단된 곳에서 이어서 진행
```

#### QA 데이터셋 생성
```bash
python -m dataset_pipeline generate-qa \
  --input ../output/wasabi_en_ko.jsonl \
  --output ../output/wasabi_qa.jsonl \
  --num-questions 220 \
  --resume
```

#### MQM 번역 품질 평가
```bash
python -m dataset_pipeline mqm-score \
  --input ../output/wasabi_en_ko.jsonl \
  --output ../output/wasabi_scored.jsonl
```

#### CGIAR 데이터셋 추출
```bash
python -m dataset_pipeline export-cgiar \
  --output ../output/cgiar_en.jsonl \
  --limit-per-dataset 200
```

#### 설정 확인
```bash
# 기본 설정 확인
python -m dataset_pipeline config

# 전체 설정 출력
python -m dataset_pipeline config --show-all

# 특정 키 조회
python -m dataset_pipeline config --key llm.generator.model
```

### 전체 파이프라인 예시

```bash
cd src

# 1. 웹에서 와사비 문서 크롤링 (영문)
python -m dataset_pipeline crawl-wasabi \
  --output ../output/wasabi_web_en.jsonl

# 2. 영문 → 한국어 번역
python -m dataset_pipeline translate \
  --input ../output/wasabi_web_en.jsonl \
  --output ../output/wasabi_en_ko.jsonl

# 3. QA 데이터셋 생성
python -m dataset_pipeline generate-qa \
  --input ../output/wasabi_en_ko.jsonl \
  --output ../output/wasabi_qa.jsonl \
  --num-questions 220

# 4. (선택) 번역 품질 평가
python -m dataset_pipeline mqm-score \
  --input ../output/wasabi_en_ko.jsonl \
  --output ../output/wasabi_scored.jsonl
```

## 설정 파일

### config/settings.yaml

```yaml
llm:
  generator:  # 질문/답변 생성용 (저비용 모델 권장)
    base_url: "${OPENAI_BASE_URL}"
    model: "gemini-2.5-flash"
    api_key: "${API_KEY}"
    temperature: 0.7
    max_tokens: 2048
  
  judge:  # SOTA 평가용 (고성능 모델 권장)
    base_url: "${OPENAI_BASE_URL}"
    model: "claude-sonnet-4-5"
    api_key: "${API_KEY}"
    temperature: 0.0
    max_tokens: 1024

rag:
  chunk_size: 512
  chunk_overlap: 50
```

### config/secrets.yaml (민감 정보)

`secrets.yaml`은 `.gitignore`에 포함되어 절대 커밋되지 않습니다.

```yaml
llm:
  generator:
    api_key: "your-actual-api-key"
  judge:
    api_key: "your-actual-api-key"

huggingface:
  token: "your-hf-token"
```

### 설정 우선순위

1. **환경 변수** (최우선): `API_KEY`, `OPENAI_BASE_URL`, `HF_TOKEN`
2. **secrets.yaml**: 민감 정보 저장
3. **settings.yaml**: 기본 설정

## Python API

CLI 외에도 Python 코드에서 직접 사용할 수 있습니다.

```python
from dataset_pipeline.config import ConfigManager, get_config
from dataset_pipeline.llm_connector import LLMConnector

# ConfigManager 사용 (권장)
config = ConfigManager()

# 또는 전역 싱글톤 사용
config = get_config()

# LLM 설정 조회
generator_config = config.get_llm_config("generator")
judge_config = config.get_llm_config("judge")

# 점 표기법으로 설정 접근
model = config.get("llm.generator.model")
chunk_size = config.get("rag.chunk_size", default=512)

# LLMConnector 생성 (새로운 방식)
connector = LLMConnector.from_config(config)
response = connector.generate("와사비의 최적 재배 온도는?", role="generator")
```

## Docker 사용법

### 빠른 시작 (Docker)

```bash
# 이미지 빌드
docker-compose build

# 설정 확인
docker-compose --profile config run --rm config

# 전체 파이프라인 실행
docker-compose --profile crawl run --rm crawler          # 1. 웹 크롤링
docker-compose --profile translate run --rm translator   # 2. 번역
docker-compose --profile qa run --rm qa-generator        # 3. QA 생성
docker-compose --profile validate run --rm validator     # 4. 검증
```

### 사용 가능한 프로파일

| 프로파일 | 명령어 | 설명 |
|----------|--------|------|
| `config` | `docker-compose --profile config run config` | 설정 확인 |
| `crawl` | `docker-compose --profile crawl run crawler` | 와사비 웹 크롤링 |
| `translate` | `docker-compose --profile translate run translator` | EN→KO 번역 |
| `qa` | `docker-compose --profile qa run qa-generator` | QA 데이터셋 생성 |
| `validate` | `docker-compose --profile validate run validator` | 데이터셋 검증 |
| `mqm` | `docker-compose --profile mqm run mqm-scorer` | 번역 품질 평가 |
| `cgiar` | `docker-compose --profile cgiar run cgiar-exporter` | CGIAR 데이터 추출 |
| `test` | `docker-compose --profile test run test` | 테스트 실행 |
| `shell` | `docker-compose --profile shell run shell` | 대화형 셸 |

### 환경 변수

Docker 실행 시 다음 환경 변수가 필요합니다:

```bash
# .env 파일 또는 export
export API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 선택
export HF_TOKEN="your-hf-token"  # CGIAR 추출 시 필요
```

### 직접 실행 (단일 명령)

```bash
# CLI 직접 실행
docker run --rm -e API_KEY=$API_KEY smartfarm-ingest:latest config

# 볼륨 마운트와 함께
docker run --rm \
  -e API_KEY=$API_KEY \
  -v $(pwd)/output:/app/output \
  smartfarm-ingest:latest generate-qa \
  --input /app/output/wasabi_en_ko.jsonl \
  --output /app/output/wasabi_qa.jsonl
```

### Legacy Docker (병렬 번역)

4-worker 병렬 번역이 필요한 경우:

```bash
docker-compose -f docker-compose.translate.yml up --build
```

## 테스트

```bash
# 파이프라인 스모크 테스트 (LLM 호출 없음)
python tests/test_pipeline.py

# ConfigManager 테스트
python scripts/test_config.py

# pytest 사용
pip install pytest
pytest tests/ -v
```

## 출력 형식

### QA 데이터셋 (JSONL)

```json
{
  "id": "wasabi_qa_0001",
  "question": "와사비 재배 시 최적 수온은?",
  "answer": "와사비의 최적 수온은 13-17°C입니다...",
  "context": "와사비는 냉수성 작물로...",
  "category": "환경조건",
  "complexity": "basic",
  "source_ids": ["web_wiki_wasabi#c42"],
  "metadata": {
    "model": "gemini-2.5-flash",
    "answer_hint": "13-17°C"
  }
}
```

### 번역 코퍼스 (JSONL)

```json
{
  "id": "web_wiki_wasabi#c42",
  "text_en": "Wasabi grows best in water temperatures of 13-17°C...",
  "text_ko": "와사비는 13-17°C의 수온에서 최적으로 생육합니다...",
  "metadata": {
    "source": "wikipedia",
    "url": "https://..."
  },
  "translation": {
    "backend": "api_llm",
    "model": "gemini-2.5-flash"
  }
}
```

## 프로젝트 구조

```
smartfarm-ingest/
├── Dockerfile                # 통합 Docker 이미지
├── docker-compose.yml        # 프로파일 기반 Docker Compose
├── requirements.txt          # Python 의존성 (전체)
├── requirements-docker.txt   # Python 의존성 (Docker용 경량)
├── pyproject.toml            # Python 패키지 설정
├── config/
│   ├── settings.yaml         # 기본 설정
│   ├── secrets.yaml          # 민감 정보 (gitignored)
│   └── secrets.yaml.example  # secrets 템플릿
├── prompts/                  # 프롬프트 템플릿 (Jinja2)
│   ├── generation.jinja      # Self-Instruct 기반 질문 생성
│   ├── evolution.jinja       # Evol-Instruct 기반 질문 진화
│   ├── judge.jinja           # Prometheus 기반 품질 평가
│   ├── answer.jinja          # RAFT 기반 답변 생성
│   └── refine.jinja          # 피드백 기반 답변 개선
├── src/
│   └── dataset_pipeline/
│       ├── __init__.py
│       ├── __main__.py       # python -m dataset_pipeline 엔트리포인트
│       ├── cli.py            # Click 기반 통합 CLI
│       ├── config.py         # ConfigManager
│       ├── llm_connector.py  # LLM API 커넥터
│       ├── generator.py      # 질문 생성기
│       ├── generator_enhanced.py  # Diversity 필터 통합 생성기
│       ├── diversity.py      # ROUGE-L 기반 다양성 필터링
│       ├── judge.py          # LLM-as-a-Judge
│       ├── reproducibility.py # 재현성 (Random seed 관리)
│       └── validation/       # 자동 검증 시스템
│           ├── diversity_metrics.py
│           ├── groundedness.py
│           ├── judge_consistency.py
│           └── validator.py
├── tests/
│   └── test_pipeline.py
├── output/                   # 출력 파일 (gitignored)
│   ├── wasabi_qa_dataset.jsonl
│   ├── validation_report.json
│   ├── DATASET_CARD.md
│   ├── LIMITATIONS.md
│   └── ACADEMIC_VALIDATION_REPORT.md
└── docker/                   # Legacy Docker (병렬 번역용)
    ├── Dockerfile.translate
    └── Dockerfile.qa_gen
```

## 라이선스

MIT License
