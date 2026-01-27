<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# sources/

Data source connectors for corpus acquisition.

## Key Files

| File | Purpose | Data Source |
|------|---------|-------------|
| `cgiar.py` | CGIAR agricultural datasets (HuggingFace Hub) | CGIAR gardian-ai-ready-docs, CIRAD, IFPRI |
| `web_crawler.py` | Web crawling for domain documents | Wasabi, agriculture websites |
| `__init__.py` | Module exports | - |

## Data Sources

### CGIAR (cgiar.py)
Official agricultural research datasets from CGIAR institutions.

**Datasets:**
- `CGIAR/gardian-ai-ready-docs` - Global agricultural innovation data
- `CGIAR/cirad-ai-documents` - CIRAD agricultural research
- `CGIAR/ifpri-ai-documents` - IFPRI food policy research

**Format:** HuggingFace Datasets (JSON)

**Usage:**
```bash
# Export CGIAR datasets (via CLI)
python -m dataset_pipeline export-cgiar \
  --output output/cgiar_en.jsonl \
  --limit-per-dataset 200
```

**Python API:**
```python
from dataset_pipeline.sources.cgiar import CgiarLoader

loader = CgiarLoader(hf_token="your-token")
docs = loader.load_dataset("CGIAR/gardian-ai-ready-docs")

for doc in docs:
    print(doc.id, doc.text, doc.metadata)
```

**Output Format:**
```json
{
  "id": "cgiar_gardian_001",
  "text": "Document content...",
  "metadata": {
    "source": "CGIAR/gardian-ai-ready-docs",
    "title": "...",
    "authors": "...",
    "url": "..."
  }
}
```

### Web Crawler (web_crawler.py)
Scrapes domain-specific websites for SmartFarm content.

**Targets:**
- Wasabi cultivation resources
- Agricultural extension sites
- Crop management databases

**Usage:**
```bash
# Crawl Wasabi documents
python -m dataset_pipeline crawl-wasabi \
  --output output/wasabi_web_en.jsonl \
  --limit 100
```

**Python API:**
```python
from dataset_pipeline.sources.web_crawler import WebCrawler

crawler = WebCrawler()
docs = crawler.crawl(
    seed_urls=["https://example.com/wasabi"],
    max_depth=3,
    max_docs=100,
)

for doc in docs:
    print(doc.id, doc.text, doc.metadata)
```

**Output Format:**
```json
{
  "id": "web_wiki_wasabi_001",
  "text": "Wasabi (Japanese horseradish)...",
  "metadata": {
    "source": "wikipedia",
    "url": "https://en.wikipedia.org/wiki/Wasabi",
    "crawl_date": "2026-01-27",
    "language": "en"
  }
}
```

## Source Pipeline

```
Data Sources (CGIAR, Web)
    ↓
[Source Loader]
Extract text + metadata
    ↓
[Text Cleaning]
Remove boilerplate, normalize encoding
    ↓
[Chunking] (in rag_connector.py)
Split into chunks for context retrieval
    ↓
[Embedding] (ChromaDB)
Vectorize for semantic search
    ↓
[JSONL Export]
Corpus ready for QA generation
```

## For AI Agents

### Adding new data sources
1. Create `new_source.py` in this directory
2. Implement loader class following existing pattern
3. Return documents with `id`, `text`, `metadata` fields
4. Register in `__init__.py`
5. Add CLI command in `../cli.py` if needed
6. Document in this AGENTS.md file

### Data source template
```python
"""
Description of data source

References:
- Paper/documentation link
"""

from dataclasses import dataclass
from typing import Iterator, Dict, Any, List

@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]

class NewSourceLoader:
    """Load documents from [source]"""

    def __init__(self, api_key: str = None, **kwargs):
        self.api_key = api_key

    def load(self, limit: int = None) -> Iterator[Document]:
        """Load documents from source"""
        count = 0
        for item in self._fetch_items():
            if limit and count >= limit:
                break

            doc = Document(
                id=self._generate_id(item),
                text=self._extract_text(item),
                metadata=self._extract_metadata(item),
            )
            yield doc
            count += 1

    def _fetch_items(self) -> Iterator[Any]:
        """Fetch items from source API/database"""
        pass

    def _generate_id(self, item: Any) -> str:
        """Generate unique document ID"""
        pass

    def _extract_text(self, item: Any) -> str:
        """Extract text content"""
        pass

    def _extract_metadata(self, item: Any) -> Dict[str, Any]:
        """Extract metadata"""
        pass
```

### Handling API keys
- Use config or environment variables
- **NEVER** hardcode credentials
- Support both API key and token-based auth
- Add error handling for auth failures

### Text extraction best practices
- Remove HTML/boilerplate if web source
- Normalize encoding (UTF-8)
- Handle special characters gracefully
- Preserve section structure when possible
- Log extraction issues

### Metadata requirements
Every document should include:
```python
metadata = {
    "source": "source_name",           # Required
    "url": "https://...",              # If applicable
    "title": "Document Title",         # If available
    "authors": ["Author 1", "..."],   # If available
    "date": "2026-01-27",             # Publication/crawl date
    "language": "en",                  # Language code
    "domain": "agriculture",           # Domain tag
}
```

### Testing sources
```python
# tests/test_pipeline.py
def test_new_source_loader():
    loader = NewSourceLoader()
    docs = list(loader.load(limit=5))

    assert len(docs) > 0
    assert all(doc.id for doc in docs)
    assert all(doc.text for doc in docs)
    assert all(doc.metadata for doc in docs)
    assert all("source" in doc.metadata for doc in docs)
```

### Performance considerations
- Implement pagination/streaming for large sources
- Cache results when possible
- Add progress reporting for CLI
- Set reasonable timeouts for API calls
- Handle rate limiting gracefully

### Configuration support
Add to `config/settings.yaml`:
```yaml
sources:
  cgiar:
    enabled: true
    datasets:
      - "CGIAR/gardian-ai-ready-docs"
  web_crawler:
    enabled: true
    seed_urls: ["https://example.com"]
    max_depth: 3
    request_timeout: 30
```

Access in source:
```python
cgiar_datasets = config.get("sources.cgiar.datasets", default=[])
max_depth = config.get("sources.web_crawler.max_depth", default=3)
```
