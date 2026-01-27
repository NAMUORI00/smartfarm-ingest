<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# tests/

pytest test suite for pipeline validation.

## Key Files

| File | Purpose | Coverage |
|------|---------|----------|
| `test_pipeline.py` | Core pipeline smoke tests | Configuration, LLM integration, JSONL I/O |
| `test_rag_grounding.py` | RAG retrieval validation | Document chunking, embedding, retrieval |

## Running Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ -v --cov=src/dataset_pipeline --cov-report=html

# Run specific test function
pytest tests/test_pipeline.py::test_config_loading -v

# Run with parallel execution
pip install pytest-xdist
pytest tests/ -n auto  # Use all CPU cores
```

## Test Categories

### Configuration Tests
- YAML loading (settings.yaml)
- Env var override (API_KEY, OPENAI_BASE_URL)
- secrets.yaml merging
- ConfigManager singleton pattern
- Key accessor (dot notation)

### LLM Integration Tests
- LLMConnector initialization
- OpenAI API compatibility
- Temperature/max_tokens parameters
- Error handling (invalid credentials, rate limits)
- Mock LLM responses for CI/CD

### JSONL I/O Tests
- Read JSONL (corpus_io.py)
- Write JSONL
- Append JSONL (resumption support)
- Field validation (id, question, answer, context)
- Batch processing

### RAG Tests (test_rag_grounding.py)
- Document chunking (chunk_size, overlap)
- Embedding generation
- ChromaDB initialization
- Retrieval accuracy
- Top-K result ranking

### Generator Tests
- Question generation from seed
- Diversity filtering
- Complexity progression (basic → advanced)
- Output format validation

### Judge Tests
- Score calculation
- Feedback generation
- Prometheus rubric application
- Multi-dimensional scoring

## Test Fixtures

### Common fixtures
```python
# conftest.py or test file
import pytest
from dataset_pipeline.config import ConfigManager
from dataset_pipeline.llm_connector import LLMConnector

@pytest.fixture
def config():
    """Load test configuration"""
    return ConfigManager()

@pytest.fixture
def mock_llm_connector(mocker):
    """Mock LLM connector for unit tests"""
    mock = mocker.MagicMock(spec=LLMConnector)
    mock.generate.return_value = "Mock LLM response"
    return mock

@pytest.fixture
def sample_qa_pairs():
    """Sample QA dataset"""
    return [
        {
            "id": "test_001",
            "question": "What is wasabi?",
            "answer": "Wasabi is a Japanese horseradish.",
            "context": "Wasabi grows in cold water streams.",
            "complexity": "basic",
        },
        {
            "id": "test_002",
            "question": "How to cultivate wasabi?",
            "answer": "Wasabi requires cold flowing water and shade.",
            "context": "Cultivation conditions: 13-17°C water, 70% shade.",
            "complexity": "intermediate",
        },
    ]
```

## Example Tests

### Configuration Test
```python
def test_config_loading(config):
    """Test YAML configuration loading"""
    # Load config
    assert config.get("domain.name") == "smartfarm"
    assert config.get("rag.chunk_size") == 512

def test_env_var_override(monkeypatch):
    """Test environment variable override"""
    monkeypatch.setenv("API_KEY", "test-key-123")
    config = ConfigManager()
    assert config.get("llm.generator.api_key") == "test-key-123"
```

### LLM Integration Test
```python
def test_llm_connector_initialization(config):
    """Test LLMConnector initialization"""
    connector = LLMConnector.from_config(config)
    assert connector is not None
    assert connector.generator_config["model"]

def test_llm_generate_mock(mock_llm_connector):
    """Test LLM generate with mock"""
    response = mock_llm_connector.generate("Test prompt", role="generator")
    assert response == "Mock LLM response"
    mock_llm_connector.generate.assert_called_once()
```

### JSONL I/O Test
```python
from dataset_pipeline.corpus_io import read_jsonl, append_jsonl
from pathlib import Path

def test_jsonl_write_read(tmp_path, sample_qa_pairs):
    """Test JSONL write and read"""
    output_file = tmp_path / "test.jsonl"

    # Write
    for item in sample_qa_pairs:
        append_jsonl(str(output_file), item)

    # Read
    items = list(read_jsonl(str(output_file)))
    assert len(items) == 2
    assert items[0]["question"] == "What is wasabi?"
```

### Generator Test
```python
def test_question_generator(mock_llm_connector, config, sample_qa_pairs):
    """Test question generation"""
    from dataset_pipeline.generator import QuestionGenerator

    generator = QuestionGenerator(mock_llm_connector, config)
    questions = generator.generate(
        seed_question="Tell about wasabi",
        context="Wasabi is...",
        num_variations=3,
    )

    assert len(questions) > 0
    mock_llm_connector.generate.assert_called()
```

## For AI Agents

### Running tests before commit
1. Run all tests: `pytest tests/ -v`
2. Check coverage: `pytest tests/ --cov=src/dataset_pipeline`
3. Minimum coverage: 70%
4. Zero test failures

### Adding new tests
1. Add to appropriate test file (or create new file if new feature)
2. Use clear, descriptive test names: `test_<feature>_<scenario>`
3. Follow AAA pattern: Arrange, Act, Assert
4. Mock external dependencies (LLM, APIs)
5. Use fixtures for common setup
6. Add docstring explaining what's tested

### Test structure
```python
def test_feature_behavior(fixture1, fixture2):
    """
    Test that feature X does Y under condition Z.

    Given: Initial setup
    When: Action taken
    Then: Expected result
    """
    # ARRANGE
    input_data = {"key": "value"}
    expected_output = {"result": "expected"}

    # ACT
    actual_output = function_under_test(input_data)

    # ASSERT
    assert actual_output == expected_output
```

### Mocking patterns
```python
import pytest
from unittest.mock import MagicMock, patch

# Mock LLM responses
@pytest.fixture
def mock_llm(mocker):
    mock = mocker.MagicMock()
    mock.generate.return_value = "Generated text"
    return mock

# Mock file I/O
@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "output.jsonl"

# Patch external calls
@patch('dataset_pipeline.llm_connector.OpenAI')
def test_with_patch(mock_openai):
    mock_openai.return_value.generate.return_value = "Response"
    # Test code
```

### Skip tests conditionally
```python
import pytest

@pytest.mark.skip(reason="Requires real LLM API key")
def test_real_llm_call():
    # Real API test
    pass

@pytest.mark.skipif(
    not os.getenv("RUN_SLOW_TESTS"),
    reason="Slow test, set RUN_SLOW_TESTS=1 to run"
)
def test_slow_operation():
    # Slow test
    pass
```

### Parameterized tests
```python
@pytest.mark.parametrize("input,expected", [
    ("basic", 1),
    ("intermediate", 2),
    ("advanced", 3),
])
def test_complexity_levels(input, expected):
    assert get_complexity_level(input) == expected
```

### Error testing
```python
def test_invalid_config_raises():
    """Test that invalid config raises error"""
    with pytest.raises(ValueError, match="Invalid model"):
        ConfigManager(invalid_model="xyz")
```

### Performance benchmarks (optional)
```python
@pytest.mark.benchmark
def test_generator_performance(benchmark):
    """Benchmark question generation speed"""
    def run():
        return generator.generate(seed_question, context)

    result = benchmark(run)
    # Benchmark results logged automatically
```

### Integration tests (slower, fewer)
Mark with `@pytest.mark.integration` for optional running:
```python
@pytest.mark.integration
def test_full_pipeline(config):
    """Full pipeline test (requires real API)"""
    # This test touches real LLM API
    pass
```

Run only integration tests:
```bash
pytest tests/ -m integration -v
```

### Continuous Integration
Minimal test setup for GitHub Actions:
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=src/dataset_pipeline
```
