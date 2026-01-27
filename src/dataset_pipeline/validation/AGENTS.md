<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# validation/

Automated quality assurance modules for dataset validation.

## Key Files

| File | Purpose | Check Type |
|------|---------|-----------|
| `validator.py` | Unified validation orchestrator | Comprehensive report |
| `groundedness.py` | Answer grounding in source documents | Hallucination detection |
| `diversity_metrics.py` | Question diversity (ROUGE-L) | Redundancy detection |
| `judge_consistency.py` | Agreement between multiple judges | Inter-rater reliability |
| `ragas_eval.py` | RAGAS metrics integration | Multi-dimensional assessment |
| `__init__.py` | Module exports | - |

## Validation Workflow

```
QA Dataset (JSONL)
    ↓
[Groundedness Validator]
- Check each answer against source documents
- Calculate grounding score
- Flag hallucinations
    ↓
[Diversity Metrics]
- Compute ROUGE-L similarity between questions
- Remove near-duplicates
- Ensure diversity score > threshold
    ↓
[Judge Consistency]
- If multiple judges: check agreement (Fleiss' Kappa)
- Flag inconsistent evaluations
- Calculate inter-rater reliability
    ↓
[RAGAS Metrics] (optional)
- Context relevance
- Answer relevance
- Factual consistency
    ↓
[Unified Report]
- Overall quality score
- Per-dimension metrics
- Recommendations
- Warnings
```

## Running Validation

### From CLI
```bash
cd src

# Run full validation suite
python -m dataset_pipeline validate \
  --input ../output/wasabi_qa.jsonl \
  --output ../output/validation_report.json

# Validate specific metric
python -m dataset_pipeline.validation.groundedness \
  --input ../output/wasabi_qa.jsonl
```

### From Python
```python
from dataset_pipeline.validation.validator import DatasetValidator
from dataset_pipeline.config import ConfigManager

config = ConfigManager()
validator = DatasetValidator(config)

report = validator.validate(
    input_file="output/wasabi_qa.jsonl",
    dataset_name="wasabi_qa",
)

print(f"Overall Quality Score: {report.overall_quality_score}")
print(f"Warnings: {report.warnings}")
print(f"Recommendations: {report.recommendations}")

# Save report
import json
with open("validation_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

## Module Details

### groundedness.py
Validates that answers are grounded in source documents (prevents hallucinations).

**Metrics:**
- Grounding score per answer (0-1)
- Hallucination detection
- Citation coverage

**Usage:**
```python
from dataset_pipeline.validation.groundedness import GroundednessValidator

validator = GroundednessValidator(config)
report = validator.validate(
    qa_pairs=[
        {"question": "...", "answer": "...", "context": "..."},
    ]
)
print(f"Grounding score: {report.average_score}")
print(f"Hallucinations: {report.hallucinations}")
```

### diversity_metrics.py
Measures uniqueness of questions using ROUGE-L similarity.

**Metrics:**
- ROUGE-L similarity matrix
- Diversity score (0-1)
- Redundancy detection
- Duplicate thresholds

**Usage:**
```python
from dataset_pipeline.diversity import DiversityValidator

validator = DiversityValidator()
report = validator.validate([q1, q2, q3, ...])
print(f"Diversity score: {report.diversity_score}")
print(f"Duplicate pairs: {report.duplicate_pairs}")
```

### judge_consistency.py
Evaluates agreement between multiple judges (inter-rater reliability).

**Metrics:**
- Fleiss' Kappa agreement coefficient
- Percent agreement
- Discordant scores

**Usage:**
```python
from dataset_pipeline.validation.judge_consistency import JudgeConsistencyValidator

validator = JudgeConsistencyValidator()
report = validator.validate(
    qa_pairs=qa_list,
    judges=["claude", "gemini"],  # Multiple judge outputs
)
print(f"Kappa: {report.fleiss_kappa}")
print(f"Agreement: {report.percent_agreement}")
```

### ragas_eval.py
Integration with RAGAS (Retrieval Augmented Generation Assessment) metrics.

**Metrics:**
- Context relevance
- Answer relevance
- Faithfulness (factual consistency)

**Note:** Requires `ragas` package: `pip install ragas`

## Output Format

### Validation Report (JSON)
```json
{
  "metadata": {
    "dataset_name": "wasabi_qa",
    "validation_date": "2026-01-27T12:34:56Z",
    "total_samples": 500
  },
  "diversity": {
    "diversity_score": 0.85,
    "duplicate_pairs": 12,
    "avg_similarity": 0.18
  },
  "groundedness": {
    "average_score": 0.92,
    "hallucinations": [
      {"index": 42, "answer": "...", "reason": "..."}
    ],
    "citations_covered": 0.89
  },
  "consistency": {
    "fleiss_kappa": 0.78,
    "percent_agreement": 0.82,
    "discordant_pairs": 8
  },
  "overall_quality_score": 0.85,
  "recommendations": [
    "Review 12 duplicate questions",
    "Fix 15 hallucinated answers",
    "Investigate 8 judge disagreements"
  ],
  "warnings": [
    "Low groundedness in medical section (0.78)",
    "High similarity between questions 42-47"
  ]
}
```

## For AI Agents

### Adding new validators
1. Create new module: `validation/new_validator.py`
2. Implement validator class inheriting from abstract base
3. Return report dataclass with `to_dict()` method
4. Register in `validator.py` orchestrator
5. Document in this AGENTS.md file

### Validator pattern
```python
"""Description of validation metric"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class MyValidatorReport:
    """Report for my validator"""
    score: float = 0.0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "details": self.details,
        }

class MyValidator:
    """Custom validator implementation"""

    def validate(self, qa_pairs: List[Dict]) -> MyValidatorReport:
        # Implementation
        return MyValidatorReport(score=score, details=details)
```

### Integration with orchestrator
In `validator.py`:
```python
from .new_validator import MyValidator, MyValidatorReport

class DatasetValidator:
    def validate(self, ...):
        # ... other validators ...

        # Add new validator
        my_validator = MyValidator(self.config)
        report.my_validator = my_validator.validate(qa_pairs)

        # Update overall score
        # ...
```

### Testing validators
```python
# tests/test_pipeline.py
def test_my_validator():
    qa_pairs = [
        {"question": "Q1", "answer": "A1", "context": "C1"},
    ]
    validator = MyValidator(config)
    report = validator.validate(qa_pairs)
    assert report.score > 0.0
```

### Performance considerations
- Cache similarity matrices for large datasets
- Use vectorized operations (numpy)
- Batch LLM calls when possible
- Progress bars for long-running validators

### Configuration
Add to `config/settings.yaml`:
```yaml
validation:
  groundedness:
    threshold: 0.8
  diversity:
    min_similarity: 0.3
  judge_consistency:
    min_agreement: 0.7
```

Access in validator:
```python
groundedness_threshold = config.get("validation.groundedness.threshold", default=0.8)
```
