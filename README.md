# SmartFarm Ingest (Current)

현재 저장소는 C1/C2 재구축 기준의 **public ingest 파이프라인**만 유지합니다.

## Scope
- `pipeline/docling_parser.py`: Docling 기반 파싱(text/table/image/formula)
- `pipeline/llm_extractor.py`: OpenAI-compatible LLM 추출기
- `pipeline/vector_writer.py`: Qdrant 벡터 적재
- `pipeline/kg_writer.py`: FalkorDB KG 적재
- `pipeline/public_ingest_runner.py`: 통합 ingest 엔트리포인트
- `pipeline/reproducibility_check.py`: ingest 재현성 점검

## Run
```bash
cd smartfarm-ingest
python -m pipeline.public_ingest_runner \
  --input-dir ./data/public_docs \
  --qdrant-host localhost --qdrant-port 6333 \
  --falkor-host localhost --falkor-port 6379
```

## Env
최소 환경변수는 아래 3개입니다.
- `OPENAI_BASE_URL` (예: `https://api.featherless.ai/v1`)
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (기본: `Qwen/Qwen2.5-32B-Instruct`)

선택:
- `OPENAI_MODEL_CANDIDATES`
- `EMBED_*`
- `DOCLING_*`

## Tests
```bash
cd smartfarm-ingest
PYTHONPATH=. ../smartfarm-search/.venv/bin/python -m pytest -q -s tests
```
