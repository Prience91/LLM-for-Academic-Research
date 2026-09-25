from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class MetadataStatus(StrEnum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


class ResponseMode(StrEnum):
    AUTO = "auto"
    STANDARD = "standard"
    LONG_FORM = "long_form"
    SOURCE_SUMMARY = "source_summary"
    NO_EVIDENCE = "no_evidence"


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    source_file: str
    sha256: str
    title: str
    authors: list[str]
    year: int | None
    journal: str | None
    doi: str | None
    theme: str | None
    page_count: int
    metadata_status: MetadataStatus
    metadata_confidence: float
    parser: str
    document_type: str = "unknown"
    publisher: str | None = None
    volume: str | None = None
    issue: str | None = None
    metadata_match_source: str = "unresolved"
    metadata_provenance: dict[str, str] = field(default_factory=dict)
    metadata_verified_at: str | None = None
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata_status"] = self.metadata_status.value
        return data


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    section: str | None
    page_start: int
    page_end: int
    ordinal: int
    token_count: int
    title: str
    authors: list[str]
    year: int | None
    journal: str | None
    doi: str | None
    theme: str | None
    document_type: str = "unknown"
    publisher: str | None = None
    volume: str | None = None
    issue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    document_id: str
    text: str
    title: str
    authors: list[str]
    year: int | None
    journal: str | None
    page_start: int
    page_end: int
    dense_similarity: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    reranker_score: float | None = None
    doi: str | None = None


@dataclass(slots=True)
class RetrievalResult:
    query: str
    rewritten_query: str
    evidence: list[Evidence]
    abstained: bool
    reason: str | None = None
    obligations: list[str] = field(default_factory=list)
    obligation_coverage: list[dict[str, Any]] = field(default_factory=list)
    coverage_complete: bool | None = None
    retrieval_passes: int = 1
    atomic_planning_succeeded: bool | None = None
