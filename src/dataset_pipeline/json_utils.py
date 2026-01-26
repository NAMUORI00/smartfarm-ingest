from __future__ import annotations

import json
from typing import Any, Optional


def _strip_code_fences(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_balanced_json_substring(text: str) -> Optional[str]:
    """Best-effort extraction of the first JSON object/array substring."""
    start_idx = None
    opening = None
    closing = None

    for i, ch in enumerate(text):
        if ch == "{":
            start_idx = i
            opening, closing = "{", "}"
            break
        if ch == "[":
            start_idx = i
            opening, closing = "[", "]"
            break

    if start_idx is None or opening is None or closing is None:
        return None

    depth = 0
    in_string = False
    escape = False
    for j in range(start_idx, len(text)):
        ch = text[j]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start_idx : j + 1].strip()

    return None


def try_parse_json(text: str) -> Optional[Any]:
    """Parse JSON from LLM output that may include code fences or extra text."""
    if not text:
        return None

    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidate = _extract_balanced_json_substring(cleaned)
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

