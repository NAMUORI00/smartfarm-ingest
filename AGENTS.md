<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# dataset-pipeline

**LLM-as-a-Judge** dataset generation pipeline for SmartFarm QA.

## STRUCTURE

```
dataset-pipeline/
├── src/dataset_pipeline/   # Core package (src/ layout)
├── config/                 # YAML settings + secrets
├── prompts/                # Jinja2 templates
├── scripts/                # Standalone utilities
├── output/                 # Generated datasets (gitignored)
└── tests/                  # pytest tests
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI commands | `src/dataset_pipeline/cli.py` | Click-based |
| LLM API calls | `src/dataset_pipeline/llm_connector.py` | OpenAI-compatible |
| QA generation | `src/dataset_pipeline/generator.py` | Self-Instruct |
| Quality evaluation | `src/dataset_pipeline/judge.py` | LLM-as-a-Judge |
| Config loading | `src/dataset_pipeline/config.py` | YAML + env vars |
| Prompt templates | `prompts/*.jinja` | Jinja2 |

## COMMANDS

```bash
# Install
pip install -e ".[dev]"

# CLI help
python -m dataset_pipeline --help

# Full pipeline
python -m dataset_pipeline crawl-wasabi -o output/wasabi_en.jsonl
python -m dataset_pipeline translate -i output/wasabi_en.jsonl -o output/wasabi_ko.jsonl
python -m dataset_pipeline generate-qa -i output/wasabi_ko.jsonl -o output/wasabi_qa.jsonl

# Docker (profile-based)
docker-compose --profile qa run --rm qa-generator
```

## CONVENTIONS

- **Relative imports** within package: `from .llm_connector import LLMConnector`
- **src/ layout**: Package under `src/dataset_pipeline/`
- Config priority: env vars > secrets.yaml > settings.yaml

## CONFIG

```yaml
# config/settings.yaml
llm:
  generator:
    model: "gemini-2.5-flash"
    base_url: "${OPENAI_BASE_URL}"
    api_key: "${API_KEY}"
  judge:
    model: "claude-sonnet-4-5"
```

## ANTI-PATTERNS

- **NEVER** use absolute imports for internal modules
- **NEVER** commit `secrets.yaml` or API keys
- **NEVER** create docs here → use workspace root `/docs/dataset-pipeline/`

## METHODOLOGY

Self-Instruct → Evol-Instruct → RAFT → LLM-as-a-Judge (Prometheus)
