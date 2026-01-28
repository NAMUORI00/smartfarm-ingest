#!/usr/bin/env python3
"""Parallel batch translation script for wasabi corpus EN->KO.

Usage:
    python batch_translate_parallel.py --worker-id 0 --num-workers 4
    
Each worker processes a shard of the input file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset_pipeline.constants import Defaults, EnvVars


def _get_data_root() -> Path:
    """Get data root from environment, defaults to /app/output for Docker."""
    root = os.environ.get(EnvVars.DATA_ROOT, "/app/output")
    return Path(root)


DATA_ROOT = _get_data_root()
INPUT_FILE = DATA_ROOT / os.environ.get(EnvVars.INPUT_FILE, "wasabi_web_en.jsonl")
OUTPUT_DIR = DATA_ROOT
BATCH_SIZE = int(os.environ.get(EnvVars.BATCH_SIZE, str(Defaults.BATCH_SIZE)))
SLEEP_BETWEEN = float(os.environ.get(EnvVars.SLEEP_BETWEEN, str(Defaults.SLEEP_BETWEEN_REQUESTS)))
MAX_RETRIES = int(os.environ.get(EnvVars.MAX_RETRIES, str(Defaults.MAX_RETRIES)))


def load_existing_ids(path: Path) -> set:
    """Load already translated IDs."""
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                ids.add(row.get("id"))
            except:
                pass
    return ids


def translate_text(client: OpenAI, text: str, model: str = "gemini-2.5-flash") -> str:
    """Translate English text to Korean with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator specializing in agricultural science. "
                            "Translate the following English text to Korean. "
                            "Maintain technical terminology accuracy. "
                            "For plant scientific names, keep the Latin name in parentheses. "
                            "Output only the translation, no explanations."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


def process_chunk(client: OpenAI, row: dict) -> dict:
    """Process a single chunk."""
    text_ko = translate_text(client, row["text"])
    return {
        "id": row["id"],
        "text_en": row["text"],
        "text_ko": text_ko,
        "metadata": row.get("metadata", {}),
        "translation": {
            "model": "gemini-2.5-flash",
            "src_lang": "en",
            "tgt_lang": "ko",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True, help="Worker ID (0-indexed)")
    parser.add_argument("--num-workers", type=int, required=True, help="Total number of workers")
    parser.add_argument("--threads", type=int, default=3, help="Threads per worker")
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("API_KEY"),
    )

    # Load input
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f]

    # Shard data for this worker
    my_rows = [r for i, r in enumerate(all_rows) if i % args.num_workers == args.worker_id]
    
    output_file = OUTPUT_DIR / f"wasabi_translated_worker{args.worker_id}.jsonl"
    
    print(f"Worker {args.worker_id}/{args.num_workers}: {len(my_rows)} chunks to translate")
    print(f"Output: {output_file}")

    # Load existing translations (resume support)
    existing_ids = load_existing_ids(output_file)
    to_translate = [r for r in my_rows if r["id"] not in existing_ids]
    print(f"Already translated: {len(existing_ids)}, Remaining: {len(to_translate)}")

    if not to_translate:
        print("All translations complete for this worker!")
        return

    # Process with thread pool
    translated = 0
    errors = 0
    batch = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_chunk, client, row): row for row in to_translate}
        
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                batch.append(result)
                translated += 1

                if translated % 10 == 0:
                    print(f"Worker {args.worker_id}: {translated}/{len(to_translate)} ({100*translated/len(to_translate):.1f}%)")

            except Exception as e:
                print(f"Worker {args.worker_id} error on {row['id']}: {e}")
                errors += 1

            # Flush batch periodically
            if len(batch) >= BATCH_SIZE:
                with open(output_file, "a", encoding="utf-8") as f:
                    for r in batch:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                batch.clear()

            time.sleep(SLEEP_BETWEEN)

    # Final flush
    if batch:
        with open(output_file, "a", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWorker {args.worker_id} complete! Translated: {translated}, Errors: {errors}")


if __name__ == "__main__":
    main()
