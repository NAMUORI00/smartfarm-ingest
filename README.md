# SmartFarm Ingest

Offline ingest 파이프라인입니다.

## Components
- `pipeline/document_parser.py`: Unstructured 기반 멀티모달 파싱
- `pipeline/llm_extractor.py`: OpenAI-compatible 엔티티/관계 추출
- `pipeline/vector_writer.py`: Qdrant dense+sparse 업서트
- `pipeline/kg_writer.py`: FalkorDB 엔티티/관계/청크 업서트
- `pipeline/public_ingest_runner.py`: E2E 공개 코퍼스 ingest
- `pipeline/artifact_export.py`: 엣지 동기화용 아티팩트 매니페스트 생성
- `pipeline/artifact_import.py`: 매니페스트 검증 + 엣지 인덱스 복원

## Public Ingest
```bash
cd smartfarm-ingest
python3 -m pipeline.public_ingest_runner \
  --input-dir ./data/public_docs \
  --qdrant-host localhost --qdrant-port 6333 \
  --falkor-host localhost --falkor-port 6379
```

## Artifact Export
```bash
python3 -m pipeline.artifact_export \
  --qdrant-dir ../data/index/qdrant \
  --falkordb-dir ../data/index/falkordb \
  --output-dir ../data/index/export
```

## Artifact Import
```bash
python3 -m pipeline.artifact_import \
  --manifest ../data/index/export/artifact_manifest.json \
  --qdrant-dir ../data/index/qdrant \
  --falkordb-dir ../data/index/falkordb
```

## Console Scripts (editable install)
```bash
pip install -e .
public-ingest --help
artifact-export --help
artifact-import --help
```

## Key Env (Strict Minimal)
- `LLM_BACKEND=llama_cpp|openai_compatible`
- `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_MODEL`
- `JUDGE_RUNTIME=api|self_host`
- `RAGAS_BASE_URL`, `RAGAS_API_KEY`
- `HF_TOKEN`
