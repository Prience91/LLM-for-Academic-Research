from __future__ import annotations

import html
import re
from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256
from urllib.parse import quote

from .models import Evidence

_MARKER = re.compile(r"\[E(\d+)\]")
_MARKER_RUN = re.compile(r"\[E\d+\](?:\s*(?:[,;]\s*)?\[E\d+\])*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_NUMERIC_CITATION = re.compile(r"\[(?:\d{1,3})(?:\s*[-,;]\s*\d{1,3})*\]")
_NARRATIVE_AUTHOR_YEAR = re.compile(
    r"\b[A-Z][\w'’.-]+(?:\s+et\s+al\.)?\s*\((?:19|20)\d{2}[a-z]?\)",
    flags=re.UNICODE,
)
_PARENTHETICAL_AUTHOR_YEAR = re.compile(
    r"\([A-Z][\w'’.-]+(?:\s+(?:&|and)\s+[A-Z][\w'’.-]+)?"
    r"(?:\s+et\s+al\.)?\s*,\s*(?:19|20)\d{2}[a-z]?"
    r"(?:\s*;\s*[A-Z][\w'’.-]+(?:\s+et\s+al\.)?\s*,\s*"
    r"(?:19|20)\d{2}[a-z]?)*\)",
    flags=re.UNICODE,
)
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", flags=re.IGNORECASE)
_DISPLAY_GLYPHS = str.maketrans({"Ƶ": "Z", "ƶ": "z"})


def _display_text(value: str) -> str:
    return value.translate(_DISPLAY_GLYPHS)


def _popover_id(document_id: str) -> str:
    digest = sha256(document_id.encode("utf-8")).hexdigest()[:16]
    return f"giis-source-{digest}"


def _markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def short_author(authors: Sequence[str]) -> str:
    if not authors:
        return "Unverified author"
    families = [author.strip().split()[-1] for author in authors if author.strip()]
    if not families:
        return "Unverified author"
    if len(families) == 1:
        return families[0]
    if len(families) == 2:
        return f"{families[0]} & {families[1]}"
    return f"{families[0]} et al."


def _short_title(title: str, max_words: int = 6) -> str:
    words = title.split()
    if not words:
        return '"Untitled corpus source"'
    shortened = " ".join(words[:max_words])
    if len(words) > max_words:
        shortened += "…"
    return f'"{shortened}"'


def normalized_doi(value: str | None) -> str | None:
    """Return a canonical DOI identifier only when a stored value is valid."""

    raw = (value or "").strip()
    prefixes = ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:")
    lowered = raw.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break
    return raw if _DOI.fullmatch(raw) else None


def doi_url(value: str | None) -> str | None:
    identifier = normalized_doi(value)
    if identifier is None:
        return None
    return "https://doi.org/" + quote(identifier, safe="/:._-;")


def inline_citation(evidence: Evidence) -> str:
    source = short_author(evidence.authors) if evidence.authors else _short_title(evidence.title)
    year = str(evidence.year) if evidence.year else "n.d."
    # Evidence IDs remain in the raw auditable draft but are not exposed in the
    # reader-facing answer. Page provenance remains available in the metadata modal.
    return _display_text(f"({source}, {year})")


def contains_non_corpus_citation(text: str) -> bool:
    """Detect bibliography-style citations that are not GIIS evidence markers."""

    without_evidence_markers = _MARKER.sub("", text)
    return bool(
        _NUMERIC_CITATION.search(without_evidence_markers)
        or _NARRATIVE_AUTHOR_YEAR.search(without_evidence_markers)
        or _PARENTHETICAL_AUTHOR_YEAR.search(without_evidence_markers)
    )


def validate_markers(text: str, evidence: Sequence[Evidence]) -> tuple[bool, set[str]]:
    allowed = {item.evidence_id for item in evidence}
    used = {f"E{number}" for number in _MARKER.findall(text)}
    return bool(used) and used.issubset(allowed), used


def citation_coverage(text: str) -> float:
    candidates: list[str] = []
    prose_lines: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            prose_lines.append(line)
            continue
        if _TABLE_SEPARATOR.fullmatch(stripped):
            continue

        # A Markdown header is immediately followed by the table separator and is
        # a label, not an evidence-bearing claim. Every other substantive row is
        # one citation unit so tabular answers are assessed like prose answers.
        next_nonempty = next(
            (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()),
            "",
        )
        if _TABLE_SEPARATOR.fullmatch(next_nonempty):
            continue
        without_markers = _MARKER.sub("", stripped)
        cell_text = re.sub(r"[|`*_]", " ", without_markers)
        if re.search(r"\w", cell_text, flags=re.UNICODE):
            candidates.append(stripped)

    for sentence in _SENTENCE.split("\n".join(prose_lines)):
        stripped = sentence.strip()
        if stripped.startswith("#"):
            continue
        without_markers = _MARKER.sub("", stripped)
        if not re.search(r"\w", without_markers, flags=re.UNICODE):
            continue
        # Concise scientific facts (including values such as "4.2% [E1].") are real
        # citation-bearing units. Retain the historical eight-word threshold only for
        # uncited short transitions so a marker-only fragment never inflates coverage.
        if not _MARKER.search(stripped) and len(without_markers.split()) < 8:
            continue
        candidates.append(stripped)
    if not candidates:
        return 0.0
    cited = sum(bool(_MARKER.search(sentence)) for sentence in candidates)
    return cited / len(candidates)


def render_inline_citations(text: str, evidence: Sequence[Evidence]) -> str:
    mapping = {item.evidence_id: item for item in evidence}

    def replace_run(match: re.Match[str]) -> str:
        evidence_ids = [f"E{number}" for number in _MARKER.findall(match.group(0))]
        items = [mapping.get(evidence_id) for evidence_id in evidence_ids]
        if any(item is None for item in items):
            return match.group(0)

        labels: list[str] = []
        seen_documents: set[str] = set()
        for item in items:
            assert item is not None
            if item.document_id in seen_documents:
                continue
            seen_documents.add(item.document_id)
            label = inline_citation(item)[1:-1]
            labels.append(_markdown_label(label))
        return f"({'; '.join(labels)})"

    return _display_text(_MARKER_RUN.sub(replace_run, text))


def render_reference_panel(evidence: Sequence[Evidence]) -> str:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        grouped[item.document_id].append(item)
    blocks = ["## Reference Panel"]
    for document_id, items in grouped.items():
        first = items[0]
        page_ranges = sorted({(item.page_start, item.page_end) for item in items})
        pages = ", ".join(
            str(start) if start == end else f"{start}-{end}" for start, end in page_ranges
        )
        authors = (
            ", ".join(first.authors)
            if first.authors
            else f"{first.title} (title used; author metadata unavailable)"
        )
        year = str(first.year) if first.year else "Not available in corpus metadata"
        similarities = [
            float(item.dense_similarity)
            for item in items
            if item.dense_similarity is not None
        ]
        if similarities:
            similarity = max(0.0, min(1.0, max(similarities)))
            similarity_percent = similarity * 100.0
            if similarity >= 0.75:
                relevance_badge = "🟢 High relevance"
            elif similarity >= 0.65:
                relevance_badge = "🟡 Moderate relevance"
            else:
                relevance_badge = "🟠 Marginal relevance"
            similarity_line = (
                f"\n- **Similarity score:** `{similarity_percent:.1f}%` "
                f"({relevance_badge})"
            )
        else:
            similarity_line = "\n- **Similarity score:** Not available for this retrieval"
        identifier = normalized_doi(first.doi)
        url = doi_url(first.doi)
        doi_line = (
            f"\n- **DOI:** [{identifier}]({url})" if identifier and url else ""
        )
        blocks.append(
            f"### {first.title}\n"
            f"- **Authors:** {authors}\n"
            f"- **Year:** {year}\n"
            f"- **Journal / publication:** "
            f"{first.journal or 'Not available in corpus metadata'}\n"
            f"- **Supporting pages:** {pages}"
            f"{similarity_line}"
            f"{doi_line}"
        )
    return _display_text("\n\n".join(blocks))


def render_citation_popovers(evidence: Sequence[Evidence]) -> str:
    """Render in-page metadata modals targeted by reader-facing inline citations."""

    if not evidence:
        return '<div class="giis-citation-modals" aria-hidden="true"></div>'

    blocks = [
        """
<style>
.giis-citation-modal { display: none; position: fixed; inset: 0; z-index: 100000;
  padding: 1rem; background: rgba(0,0,0,.72); align-items: center; justify-content: center; }
.giis-citation-modal:target { display: flex; }
.giis-citation-card { width: min(42rem, 92vw); max-height: 84vh; overflow: auto;
  border: 1px solid #555; border-radius: .75rem; padding: 1rem 1.2rem;
  background: #050505 !important; color: #fff !important;
  box-shadow: 0 .9rem 3rem rgba(0,0,0,.55); }
.giis-citation-card, .giis-citation-card * { color: #fff !important; }
.giis-citation-card a { color: #8dccff !important; }
.giis-citation-card dl { display: grid; grid-template-columns: max-content 1fr;
  gap: .35rem .8rem; }
.giis-citation-card dt { font-weight: 700; }
.giis-citation-close { float: right; border: 1px solid #aaa; border-radius: .4rem;
  padding: .25rem .55rem; text-decoration: none; }
</style>
<div class="giis-citation-modals">
""".strip()
    ]
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        grouped[item.document_id].append(item)
    for document_id, items in grouped.items():
        first = items[0]
        popover_id = _popover_id(document_id)
        authors = (
            ", ".join(first.authors)
            if first.authors
            else f"{first.title} (title used; author metadata unavailable)"
        )
        year = str(first.year) if first.year else "Not available in corpus metadata"
        journal = first.journal or "Not available in corpus metadata"
        identifier = normalized_doi(first.doi)
        url = doi_url(first.doi)
        doi_html = (
            f'<dt>DOI</dt><dd><a href="{html.escape(url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">'
            f"{html.escape(identifier)}</a></dd>"
            if identifier and url
            else ""
        )
        page_ranges = sorted({(item.page_start, item.page_end) for item in items})
        pages = ", ".join(
            str(start) if start == end else f"{start}-{end}"
            for start, end in page_ranges
        )
        blocks.append(
            f'<div class="giis-citation-modal" id="{html.escape(popover_id, quote=True)}" '
            f'role="dialog" aria-modal="true" aria-label="Source metadata">'
            '<section class="giis-citation-card">'
            '<a class="giis-citation-close" href="#" aria-label="Close source metadata">Close</a>'
            f"<h3>{html.escape(first.title)}</h3>"
            "<dl>"
            f"<dt>Authors</dt><dd>{html.escape(authors)}</dd>"
            f"<dt>Year</dt><dd>{html.escape(year)}</dd>"
            f"<dt>Journal / publication</dt><dd>{html.escape(journal)}</dd>"
            f"<dt>Supporting pages</dt><dd>{html.escape(pages)}</dd>"
            f"{doi_html}</dl></section></div>"
        )
    blocks.append("</div>")
    return _display_text("".join(blocks))
