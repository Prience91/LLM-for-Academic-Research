from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Evidence, ResponseMode

_CLOSED_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "should ",
    "has ",
    "have ",
    "will ",
)

_LONG_FORM_SIGNALS = {
    "analyze",
    "assess",
    "compare",
    "comprehensive",
    "discuss",
    "essay",
    "explore",
    "literature review",
    "review",
    "summarize",
    "synthesise",
    "synthesize",
}


def select_response_mode(query: str, requested: ResponseMode = ResponseMode.AUTO) -> ResponseMode:
    if requested is not ResponseMode.AUTO:
        return requested
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    if normalized.startswith(_CLOSED_PREFIXES) and len(normalized.split()) <= 25:
        return ResponseMode.STANDARD
    if any(signal in normalized for signal in _LONG_FORM_SIGNALS):
        return ResponseMode.LONG_FORM
    if normalized.startswith(("how ", "why ")) and len(normalized.split()) >= 8:
        return ResponseMode.LONG_FORM
    return ResponseMode.STANDARD


def evidence_is_sufficient(
    evidence: Sequence[Evidence],
    mode: ResponseMode,
    minimum_distinct_sources_for_long_form: int,
) -> bool:
    if not evidence:
        return False
    if mode in {ResponseMode.LONG_FORM, ResponseMode.SOURCE_SUMMARY}:
        distinct = {item.document_id for item in evidence}
        if mode is ResponseMode.LONG_FORM and len(distinct) < minimum_distinct_sources_for_long_form:
            return False
    return True

