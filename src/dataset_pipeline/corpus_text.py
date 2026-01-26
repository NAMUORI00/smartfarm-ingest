from __future__ import annotations

import re
from typing import Iterable, List


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RE.sub(" ", s)
    s = _NL_RE.sub("\n\n", s)
    return s.strip()


def join_nonempty(parts: Iterable[str], sep: str = "\n\n") -> str:
    out: List[str] = []
    for p in parts:
        p = (p or "").strip()
        if p:
            out.append(p)
    return sep.join(out)


def extract_numbers_and_units(text: str) -> List[str]:
    """Very lightweight extractor for numeric tokens + optional units.

    Used as a validator signal (not a full parser).
    """
    t = (text or "")
    pat = re.compile(
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:℃|°c|c|%|ppm|ppb|ds/m|ms/cm|ph|ec|kg|g|mg|l|ml|cm|mm|m)\b"
    )
    return [m.group(0).strip() for m in pat.finditer(t)]

