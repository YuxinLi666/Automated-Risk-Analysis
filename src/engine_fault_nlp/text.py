from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, List, Sequence


_SPACE_RE = re.compile(r"\s+")
_DTC_RE = re.compile(r"(?<!\d)0*([1-9]\d{1,4})(?!\d)")
_PUNCT_TRANSLATION = str.maketrans({"，": ",", "；": ";", "：": ":", "（": "(", "）": ")"})


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).translate(_PUNCT_TRANSLATION)
    text = _SPACE_RE.sub(" ", text).strip()
    return "" if text == "#" else text


def deduplicate_fragments(values: Iterable[object]) -> str:
    seen = set()
    output: List[str] = []
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return "\n".join(output)


def parse_dtc_codes(value: object) -> List[str]:
    text = normalize_text(value)
    codes = []
    for match in _DTC_RE.finditer(text):
        raw = match.group(1)
        normalized = str(int(raw))
        if normalized not in codes:
            codes.append(normalized)
    return codes


def stable_id(prefix: str, value: str, length: int = 12) -> str:
    digest = hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()[:length].upper()
    return "{}_{}".format(prefix, digest)


def character_ngrams(text: str, sizes: Sequence[int] = (2, 3, 4)) -> List[str]:
    compact = re.sub(r"\s+", "", normalize_text(text).lower())
    grams = []
    for size in sizes:
        if len(compact) < size:
            continue
        grams.extend(compact[i : i + size] for i in range(len(compact) - size + 1))
    return grams


def is_negated(text: str, start: int, window: int = 10) -> bool:
    prefix = text[max(0, start - window) : start]
    return any(token in prefix for token in ("未见", "无", "没有", "排除", "未发现", "正常"))

