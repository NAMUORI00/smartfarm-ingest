<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# prompts/

Jinja2 prompt templates for LLM generation and evaluation.

## Key Files

| File | Purpose | Used By |
|------|---------|---------|
| `generation.jinja` | Self-Instruct Q&A generation | `generator.py` |
| `evolution.jinja` | Evol-Instruct complexity progression | `generator_enhanced.py` |
| `judge.jinja` | Prometheus-based quality evaluation | `judge.py` |
| `answer.jinja` | RAFT context-based answer generation | `generator.py` |
| `refine.jinja` | Feedback-based answer refinement | Refinement loop |
| `mt_translate.jinja` | Machine translation (EN→KO) | `corpus_cli.py` |
| `mt_postedit.jinja` | Post-editing translated text | `corpus_cli.py` |
| `mt_mqm_judge.jinja` | MQM (Multidimensional Quality Metrics) evaluation | `corpus_cli.py` |

## Template Variables

### generation.jinja
```jinja2
{{ domain_description }}  # Domain context (e.g., "SmartFarm tomato cultivation")
{{ seed_question }}       # Base question to evolve from
{{ document_context }}    # Retrieved document chunk
{{ language }}            # Target language (e.g., "Korean")
{{ complexity_level }}    # "basic", "intermediate", "advanced"
```

### judge.jinja
```jinja2
{{ question }}            # Question to evaluate
{{ answer }}              # Candidate answer
{{ context }}             # Source document
{{ evaluation_criteria }} # List of rubric items
{{ score_range }}         # (1, 5) tuple
{{ language }}            # Evaluation language
```

### answer.jinja
```jinja2
{{ question }}            # User question
{{ retrieved_docs }}      # Top-K documents from RAG
{{ domain_context }}      # Domain background
{{ language }}            # Response language
```

## For AI Agents

### When modifying templates
- Keep templates focused on single responsibility
- Use clear, descriptive variable names
- Add comments above complex sections
- Test with sample data before committing

### Adding new templates
1. Create `{name}.jinja` in this directory
2. Document in this AGENTS.md file
3. Import in the corresponding Python module
4. Add test in `tests/test_pipeline.py` if critical

### Template loading pattern
```python
from pathlib import Path
from jinja2 import Template

template_path = Path(__file__).parent.parent / "prompts" / "generation.jinja"
template = Template(template_path.read_text(encoding="utf-8"))
rendered = template.render(
    domain_description="...",
    seed_question="...",
    document_context="...",
)
```

### Testing templates
- Use `tests/test_pipeline.py` for template validation
- Verify output doesn't exceed `max_tokens` in config
- Check for obvious hallucinations in generated content
- Validate JSON/structured output parsing

### Language support
- All templates support configurable language via `{{ language }}`
- Default: Korean (ko) for SmartFarm domain
- Add language-specific guidance in template comments
- Maintain consistent terminology across templates

### Prompt engineering best practices
- Use Chain-of-Thought for complex reasoning
- Include example outputs for structured generation
- Set temperature/top_p in config (not hardcoded in template)
- Add explicit constraints (length, format, domain)
