#!/usr/bin/env python3
"""Merge translated shards from parallel workers."""

import json
import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("DATA_ROOT", "/app/output"))
FINAL_OUTPUT = OUTPUT_DIR / "wasabi_en_ko_parallel.jsonl"


def main():
    # Find all worker output files
    worker_files = sorted(OUTPUT_DIR.glob("wasabi_translated_worker*.jsonl"))
    
    if not worker_files:
        print("No worker output files found!")
        return
    
    print(f"Found {len(worker_files)} worker files")
    
    # Merge all results
    all_results = {}
    for wf in worker_files:
        print(f"Reading {wf.name}...")
        with open(wf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    all_results[row["id"]] = row
                except:
                    pass
    
    print(f"Total unique translations: {len(all_results)}")
    
    # Write merged output
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        for rid in sorted(all_results.keys()):
            f.write(json.dumps(all_results[rid], ensure_ascii=False) + "\n")
    
    print(f"Merged output written to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
