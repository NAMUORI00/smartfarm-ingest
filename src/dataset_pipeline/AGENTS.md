<!-- Parent: ../../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# dataset_pipeline/

Core Python package for LLM-based dataset generation (Self-Instruct + Evol-Instruct + RAFT + LLM-as-a-Judge + Prometheus).

## STRUCTURE

```
dataset_pipeline/
├── __init__.py                 # Package exports
├── __main__.py                 # python -m entry point
├── cli.py                      # Click-based CLI (crawl, translate, generate-qa, mqm-score, validate)
├── config.py                   # ConfigManager (YAML + env vars + secrets)
├── llm_connector.py            # OpenAI-compatible LLM API wrapper (multi-backend)
├── generator.py                # Self-Instruct question generation
├── generator_enhanced.py       # Enhanced generation with diversity filtering
├── judge.py                    # LLM-as-a-Judge (Prometheus rubric-based evaluation)
├── refiner.py                  # Feedback-based answer refinement
├── diversity.py                # ROUGE-L diversity filtering
├── reproducibility.py          # Random seed + determinism management
├── corpus_cli.py               # Corpus/MT/MQM integration commands
├── corpus_io.py                # JSONL I/O + batching utilities
├── corpus_text.py              # Text processing utilities
├── chunking_utils.py           # Document chunking strategies
├── json_utils.py               # JSON parsing + validation helpers
├── langchain_compat.py         # LangChain compatibility layer
├── mt_tools.py                 # Machine translation utilities
├── rag_connector.py            # RAG retrieval (ChromaDB embeddings)
├── validation/                 # Automated quality assurance
│   ├── __init__.py
│   ├── validator.py            # Unified validation orchestrator
│   ├── groundedness.py         # Check answers grounded in documents
│   ├── diversity_metrics.py    # ROUGE-L diversity scores
│   ├── judge_consistency.py    # Inter-judge agreement
│   └── ragas_eval.py           # RAGAS metrics integration
└── sources/                    # Data source connectors
    ├── __init__.py
    ├── cgiar.py                # CGIAR agricultural dataset connectors
    └── web_crawler.py          # Web crawling (Wasabi, agriculture websites)
```

## WHERE TO LOOK

| Task | File | Details |
|------|------|---------|
| Add CLI command | `cli.py` | Add `@cli.command()` decorated function |
| Modify LLM calls | `llm_connector.py` | LLMConnector.generate() method |
| Change QA generation | `generator.py` + `../prompts/generation.jinja` | QuestionGenerator class |
| Change evaluation | `judge.py` + `../prompts/judge.jinja` | LLMJudge class |
| Add validation metric | `validation/` | Create new module following validator.py pattern |
| Add data source | `sources/` | Implement source connector |
| Update CLI help | `cli.py` | Docstring + Click decorators |

## KEY CLASSES & PATTERNS

### ConfigManager (config.py)
```python
from .config import ConfigManager, get_config

# Create instance (or get singleton)
config = ConfigManager()
config_singleton = get_config()

# Access settings
model = config.get("llm.generator.model")
chunk_size = config.get("rag.chunk_size", default=512)

# Get subsections
generator_config = config.get_llm_config("generator")  # Returns dict
```

### LLMConnector (llm_connector.py)
```python
from .llm_connector import LLMConnector

# Create from config
connector = LLMConnector.from_config(config)

# Generate with role-specific settings
response = connector.generate(
    prompt="...",
    role="generator",  # or "judge"
    temperature=0.7,   # optional override
)
```

### QuestionGenerator (generator.py)
```python
from .generator import QuestionGenerator

gen = QuestionGenerator(llm_connector, config)
questions = gen.generate(
    seed_question="What is the optimal temperature?",
    context="Document chunk...",
    num_variations=5,
)
```

### LLMJudge (judge.py)
```python
from .judge import LLMJudge

judge = LLMJudge(llm_connector, config)
score, feedback = judge.evaluate(
    question="...",
    answer="...",
    context="...",
)
```

## METHODOLOGY FLOW

```
INPUT: Seed Questions + Documents
    ↓
[Self-Instruct] (generator.py)
Generate initial Q&A pairs from documents
    ↓
[Diversity Filter] (diversity.py)
Remove redundant questions (ROUGE-L similarity)
    ↓
[Evol-Instruct] (generator_enhanced.py)
Increase complexity: basic → intermediate → advanced
    ↓
[RAFT Context] (rag_connector.py)
Retrieve top-k documents for each question
    ↓
[Answer Generation] (answer.jinja template)
Generate answers grounded in retrieved documents
    ↓
[LLM-as-a-Judge] (judge.py + judge.jinja)
Evaluate: groundedness, accuracy, completeness
    ↓
[Prometheus Scoring]
Multi-dimensional rubric: 1-5 scores
    ↓
[Refinement Loop] (refiner.py)
If score < threshold: refine answer, re-judge
    ↓
[Validation] (validation/)
- Groundedness check: answers grounded in docs
- Diversity: unique questions
- Judge consistency: agreement between evaluators
    ↓
OUTPUT: High-quality QA dataset (JSONL)
```

## CONVENTIONS

- **Relative imports**: `from .config import ConfigManager`
- **No absolute imports**: Never use `from dataset_pipeline.config import ...`
- **All LLM calls**: Route through `LLMConnector`
- **Configuration access**: Use `ConfigManager.get(dotted.key)` pattern
- **Error handling**: Raise descriptive exceptions, don't swallow silently
- **Logging**: Use `import logging; logger = logging.getLogger(__name__)`
- **Type hints**: Add return types to all public functions

## COMMANDS

```bash
cd src

# Configuration
python -m dataset_pipeline config                       # Show resolved config
python -m dataset_pipeline config --key llm.generator.model  # Query specific key

# Data crawling
python -m dataset_pipeline crawl-wasabi -o wasabi_en.jsonl

# Translation
python -m dataset_pipeline translate -i wasabi_en.jsonl -o wasabi_ko.jsonl --resume

# QA generation
python -m dataset_pipeline generate-qa -i wasabi_ko.jsonl -o wasabi_qa.jsonl --num-questions 200 --resume

# Quality evaluation
python -m dataset_pipeline mqm-score -i wasabi_ko.jsonl -o wasabi_scored.jsonl

# Validation
python -m dataset_pipeline validate -i wasabi_qa.jsonl

# CGIAR export
python -m dataset_pipeline export-cgiar -o cgiar_en.jsonl

# Help
python -m dataset_pipeline --help
python -m dataset_pipeline generate-qa --help
```

## FOR AI AGENTS

### Adding new modules
1. Create file in `dataset_pipeline/` or subdirectory
2. Add module-level docstring with methodology reference
3. Use relative imports: `from .config import ...`
4. Add type hints to public functions
5. Write tests in `tests/test_pipeline.py`
6. Document in this file

### Testing
- Unit tests: `pytest tests/ -v`
- Mock LLM calls using fixtures
- Test JSONL I/O with minimal data
- Verify output structure before committing

### Configuration changes
- Modify `config/settings.yaml` (safe for git)
- Add templates in `prompts/` if new
- Update config loading in `config.py` if new section
- Test with `python -m dataset_pipeline config`

### Error handling examples
```python
import logging
logger = logging.getLogger(__name__)

try:
    result = llm_connector.generate(prompt)
except ValueError as e:
    logger.error(f"Invalid prompt: {e}")
    raise
except Exception as e:
    logger.exception(f"LLM call failed: {e}")
    raise
```

### Windows compatibility
- Use `Path` for cross-platform paths
- Add console encoding fix if needed: `sys.stdout.reconfigure(errors='replace')`
- Test file I/O with both ASCII and UTF-8

### Resume support pattern
```python
from .corpus_io import read_jsonl, append_jsonl

processed = set()
if Path(output_file).exists():
    for item in read_jsonl(output_file):
        processed.add(item["id"])

for item in items:
    if item["id"] not in processed:
        result = process(item)
        append_jsonl(output_file, result)
```
