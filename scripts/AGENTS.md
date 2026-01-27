<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# scripts/

Standalone utility scripts for dataset processing, testing, and administration.

## Key Files

| File | Purpose | Usage |
|------|---------|-------|
| `batch_translate.py` | Sequential EN→KO translation with resumption | `python batch_translate.py --input FILE --output FILE --resume` |
| `batch_translate_parallel.py` | 4-worker parallel translation | `python batch_translate_parallel.py --input FILE --output FILE` |
| `merge_translations.py` | Merge parallel translation results | `python merge_translations.py --inputs FILE1 FILE2 FILE3 FILE4 --output FILE` |
| `generate_wasabi_qa.py` | Standalone QA generation for Wasabi documents | `python generate_wasabi_qa.py --input FILE --output FILE` |
| `test_config.py` | Configuration validation and testing | `python test_config.py` |

## Running Scripts

All scripts must be run from the `dataset-pipeline/` root:

```bash
cd dataset-pipeline

# Test configuration
python scripts/test_config.py

# Generate QA dataset
python scripts/generate_wasabi_qa.py \
  --input output/wasabi_en_ko.jsonl \
  --output output/wasabi_qa.jsonl

# Batch translate
python scripts/batch_translate.py \
  --input output/wasabi_en.jsonl \
  --output output/wasabi_ko.jsonl \
  --resume
```

## Script Categories

### Data Processing
- `batch_translate.py` - Single-threaded translation with checkpointing
- `batch_translate_parallel.py` - Multi-worker translation
- `merge_translations.py` - Combine parallel results
- `generate_wasabi_qa.py` - QA dataset generation

### Testing & Validation
- `test_config.py` - Validate configuration loading and merging

### Administration
Scripts in this folder can be executed directly without CLI framework overhead.

## For AI Agents

### When adding new scripts
1. Add to `scripts/` directory
2. Make executable if standalone: `chmod +x script.py`
3. Document in this AGENTS.md file
4. Include docstring at top of file
5. Add `if __name__ == "__main__":` main block for direct execution
6. Support `--help` flag for usage instructions

### Script patterns
```python
#!/usr/bin/env python3
"""Script description"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--input", required=True, help="Input file")
    parser.add_argument("--output", required=True, help="Output file")
    args = parser.parse_args()

    # Implementation

if __name__ == "__main__":
    main()
```

### Resumption support
Scripts that process large files should support `--resume` flag:
- Check if output file exists
- Detect last processed line number
- Continue from that point
- Log progress to stderr

### Error handling
- Use try/except for file I/O
- Exit with non-zero code on failure
- Print clear error messages to stderr
- Include stack trace only in debug mode

### Configuration access
From scripts, load config with:
```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_pipeline.config import ConfigManager
config = ConfigManager()
model = config.get("llm.generator.model")
```

### Testing scripts
- Create test in `tests/test_pipeline.py` for critical scripts
- Use minimal fixtures for faster testing
- Mock external API calls
- Validate output format (JSONL structure, required fields)
