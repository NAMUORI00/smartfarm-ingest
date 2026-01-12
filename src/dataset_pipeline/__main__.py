#!/usr/bin/env python3
"""
Dataset Pipeline CLI 엔트리포인트.

사용법:
    python -m dataset_pipeline --help
    python -m dataset_pipeline config
    python -m dataset_pipeline crawl-wasabi --output output/wasabi.jsonl
"""

from .cli import main

if __name__ == "__main__":
    main()
