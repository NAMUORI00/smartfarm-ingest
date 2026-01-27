<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# config/

Configuration files (YAML settings + secrets).

## Key Files

| File | Description |
|------|-------------|
| `settings.yaml` | Default configuration (LLM models, RAG settings, pipeline parameters) |
| `secrets.yaml.example` | Template for sensitive credentials (API keys, HuggingFace token) |
| `secrets.yaml` | **Gitignored** - actual sensitive data (never commit) |

## Config Priority

Configuration loads in this order (first match wins):

1. **Environment variables** (highest priority)
   - `API_KEY`, `OPENAI_API_KEY` → LLM authentication
   - `OPENAI_BASE_URL`, `API_BASE_URL` → API endpoint
   - `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN` → HuggingFace access

2. **secrets.yaml** (if exists)
   - Stores sensitive data locally
   - **Never** commit to git

3. **settings.yaml** (lowest priority)
   - Default configuration values
   - Safe for git (no secrets here)

## Configuration Keys

### LLM Settings
- `llm.generator.*` - Model for Q&A generation (fast, low-cost)
- `llm.judge.*` - Model for quality evaluation (high-performance, accurate)

### Pipeline Settings
- `pipeline.seed_questions_per_chunk` - Questions per document chunk
- `pipeline.max_iterations` - Refinement iterations
- `pipeline.evaluation_criteria` - Quality rubric (groundedness, accuracy, completeness)

### RAG Settings
- `rag.chunk_size` - Document chunk size (bytes)
- `rag.chunk_overlap` - Overlap between chunks
- `rag.top_k` - Retrieval count

### Domain Settings
- `domain.name` - Domain identifier (e.g., "smartfarm")
- `domain.description` - Domain description
- `domain.seed_questions` - Initial seed questions

### Corpus/MT/MQM (for corpus_cli.py)
- `corpus.cgiar_datasets` - CGIAR dataset sources
- `translation.src_lang`, `translation.tgt_lang` - Translation pair (e.g., en→ko)
- `mqm.judges` - Multiple evaluators for translation quality

## For AI Agents

### When modifying settings.yaml
- Update both defaults AND documentation comments
- Use `${VAR_NAME}` syntax for env var interpolation
- Include descriptions for each parameter in YAML comments
- Add Korean comments for domain context

### When adding new config sections
1. Add to `settings.yaml` with clear defaults
2. Document the new section in this file
3. Update ConfigManager in `src/dataset_pipeline/config.py` if new logic needed
4. Add corresponding env var support in config.py

### Testing config changes
- Run: `python -m dataset_pipeline config` (shows resolved config)
- Run: `python scripts/test_config.py` (validates structure)

### Secret Handling
- **NEVER** add secrets directly to settings.yaml
- **NEVER** commit secrets.yaml
- Always use `.gitignore` exclusion
- Use env vars for CI/CD pipelines
