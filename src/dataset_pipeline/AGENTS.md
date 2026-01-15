# dataset_pipeline/

Core Python package for LLM-based dataset generation.

## STRUCTURE

```
dataset_pipeline/
├── __main__.py         # python -m entrypoint
├── cli.py              # Click CLI (crawl, translate, generate-qa, mqm-score)
├── config.py           # ConfigManager (YAML + env)
├── llm_connector.py    # OpenAI-compatible API wrapper
├── generator.py        # Question generation (Self-Instruct)
├── generator_enhanced.py  # + Diversity filtering
├── judge.py            # LLM-as-a-Judge evaluation
├── refiner.py          # Answer refinement (Evol-Instruct)
├── diversity.py        # ROUGE-L deduplication
├── corpus_cli.py       # Legacy corpus commands
├── corpus_io.py        # JSONL I/O utilities
├── mt_tools.py         # Translation utilities
├── validation/         # Automated validation
│   ├── validator.py
│   ├── groundedness.py
│   └── diversity_metrics.py
└── sources/            # Data source connectors
```

## WHERE TO LOOK

| Task | File |
|------|------|
| Add CLI command | `cli.py` → add `@cli.command()` |
| Modify LLM calls | `llm_connector.py` |
| Change QA prompts | `generator.py` + `../prompts/*.jinja` |
| Add evaluation metric | `judge.py` or `validation/` |
| Add data source | `sources/` or `corpus_cli.py` |

## KEY CLASSES

```python
# ConfigManager (config.py)
config = ConfigManager()
llm_cfg = config.get_llm_config("generator")

# LLMConnector (llm_connector.py)
connector = LLMConnector.from_config(config)
response = connector.generate(prompt, role="generator")

# Generator (generator.py)
generator = QuestionGenerator(connector)
qa_pairs = generator.generate_from_corpus(corpus)
```

## CONVENTIONS

- **Relative imports**: `from .config import ConfigManager`
- All LLM calls go through `LLMConnector`
- Config access via `ConfigManager.get()` with dot notation

## VALIDATION

```bash
# Run validation suite
python -m dataset_pipeline validate --input output/qa.jsonl

# Check groundedness
python -m dataset_pipeline.validation.groundedness
```

## NOTES

- Windows console fix: `sys.stdout.reconfigure(errors='replace')`
- Resume support: Most commands have `--resume` flag
- Output format: JSONL with `id`, `question`, `answer`, `context`
