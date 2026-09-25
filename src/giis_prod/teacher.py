from __future__ import annotations

import json
import math
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any

from .candidate_validation import ANSWER_STYLES, validate_answer_style
from .citations import citation_coverage, contains_non_corpus_citation
from .config import Settings
from .dataset_release import begin_dataset_release, finalize_dataset_release
from .io import (
    content_version,
    read_jsonl,
    sha256_file,
    stable_digest,
    write_json_atomic,
    write_jsonl_atomic,
)
from .production_readiness import assert_dataset_generation_ready
from .semantic_audit import plan_semantic_audit

_EVIDENCE_MARKER = re.compile(r"\[E\d+\]")
_EVIDENCE_LIKE_BRACKET = re.compile(r"\[[^\]\n]*E\d+[^\]\n]*\]")
_LOW_INFORMATION_SECTION = re.compile(
    r"\b(references?|literature cited|bibliography|acknowledg(?:e)?ments?|"
    r"author biographies?|author contributions?|authorship contribution statement|funding|"
    r"conflicts? of interest|competing interests?|supplementary material)\b",
    re.IGNORECASE,
)
_QUESTION_WORD = re.compile(r"[a-z0-9]+")
_CLAIM_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+(?=\S)")
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "their",
    "to",
    "what",
    "which",
    "why",
    "with",
}
_CHECKPOINT_VERSION = 1
_GENERATION_POLICY_VERSION = 15
_SEMANTIC_VERIFIER_PROTOCOL_VERSION = 3


class TeacherInfrastructureError(RuntimeError):
    """A teacher request failed before candidate quality could be assessed."""


class DatasetAuditJournal:
    """Append-only JSONL diagnostics that survive interruption and failed releases."""

    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, **fields: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


@dataclass(slots=True)
class EvidenceBundle:
    bundle_id: str
    document_ids: tuple[str, ...]
    chunks: list[dict[str, Any]]

    def context(self) -> str:
        blocks: list[str] = []
        for index, chunk in enumerate(self.chunks, start=1):
            pages = (
                str(chunk["page_start"])
                if chunk["page_start"] == chunk["page_end"]
                else f"{chunk['page_start']}-{chunk['page_end']}"
            )
            blocks.append(
                f"[E{index}]\n"
                f"Document ID: {chunk['document_id']}\n"
                f"Title: {chunk['title']}\n"
                f"Pages: {pages}\n"
                f"Section: {chunk.get('section') or 'Unspecified'}\n"
                f"Text: {chunk['text']}"
            )
        return "\n\n".join(blocks)


def _stratified_chunks(chunks: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(chunks, key=lambda row: (row["page_start"], row["ordinal"]))
    if len(ordered) <= count:
        return ordered
    positions = {round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)}
    return [ordered[position] for position in sorted(positions)]


def _is_substantive_chunk(chunk: dict[str, Any]) -> bool:
    section = re.sub(r"\s+", " ", str(chunk.get("section") or "")).strip()
    compact_section = re.sub(r"[^a-z]", "", section.casefold())
    if _LOW_INFORMATION_SECTION.search(section) or compact_section.startswith(
        (
            "articleinfo",
            "keywords",
            "declarationofcompetinginterest",
            "creditauthorship",
        )
    ):
        return False
    if re.fullmatch(r"[A-Z]\.\s+[\wÀ-ž'’-]+", section):
        return False
    title = re.sub(r"\s+", " ", str(chunk.get("title") or "")).strip()
    text = str(chunk.get("text") or "").strip()
    if section.casefold() == title.casefold() and "abstract" not in text.casefold():
        return False
    return bool(text)


def _question_similarity(first: str, second: str) -> float:
    left = re.sub(r"\s+", " ", first.casefold()).strip()
    right = re.sub(r"\s+", " ", second.casefold()).strip()
    character_score = SequenceMatcher(None, left, right).ratio()
    left_terms = {
        token for token in _QUESTION_WORD.findall(left) if token not in _QUESTION_STOPWORDS
    }
    right_terms = {
        token for token in _QUESTION_WORD.findall(right) if token not in _QUESTION_STOPWORDS
    }
    union = left_terms | right_terms
    token_score = len(left_terms & right_terms) / len(union) if union else 1.0
    return max(character_score, token_score)


def _claim_units(answer: str) -> list[dict[str, Any]]:
    """Identify verifier units without changing or re-rendering the original answer."""
    claims: list[dict[str, Any]] = []
    for raw in _CLAIM_BOUNDARY.split(str(answer)):
        text = raw.strip()
        without_markers = _EVIDENCE_MARKER.sub("", text).strip()
        if not text or not re.search(r"\w", without_markers, flags=re.UNICODE):
            continue
        claims.append(
            {
                "claim_id": f"claim_{len(claims) + 1}",
                "text": text,
                "cited_evidence_ids": [
                    int(marker[2:-1]) for marker in _EVIDENCE_MARKER.findall(text)
                ],
            }
        )
    return claims


def _cited_document_ids(bundle: EvidenceBundle, answer: str) -> set[str]:
    return {
        str(bundle.chunks[int(marker[2:-1]) - 1]["document_id"])
        for marker in _EVIDENCE_MARKER.findall(answer)
        if 1 <= int(marker[2:-1]) <= len(bundle.chunks)
    }


def split_document_ids(
    documents: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        grouped[str(document.get("theme") or "__unclassified__")].append(document["document_id"])
    rng = random.Random(seed)  # noqa: S311 - deterministic dataset split, not security
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    for identifiers in grouped.values():
        rng.shuffle(identifiers)
        validation_count = (
            max(1, round(len(identifiers) * validation_fraction))
            if len(identifiers) > 1
            else 0
        )
        validation_ids.update(identifiers[:validation_count])
        train_ids.update(identifiers[validation_count:])
    if not train_ids or not validation_ids:
        raise ValueError("Document-level split requires at least two usable documents")
    return train_ids, validation_ids


def _benchmark_document_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(document_id)
        for row in read_jsonl(path)
        for document_id in row.get("expected_document_ids", [])
    }


def _heldout_plan(settings: Settings) -> tuple[set[str], set[str], str | None]:
    try:
        path = settings.path("paths.heldout_manifest")
    except KeyError:
        return set(), set(), None
    if not path.exists():
        raise FileNotFoundError(
            f"Held-out split manifest is required before dataset generation: {path}. "
            "Run `python3 -u giis_pipeline.py plan-heldout` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    heldout = {str(value) for value in payload.get("heldout_document_ids", [])}
    sft_pool = {str(value) for value in payload.get("sft_document_ids", [])}
    if not heldout or not sft_pool or heldout & sft_pool:
        raise ValueError("Held-out split manifest is empty, overlapping, or malformed")
    return heldout, sft_pool, str(payload.get("selection_digest") or "")


def select_training_documents(
    settings: Settings, expected_documents: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match the hand-picked training PDFs to the already parsed corpus by SHA-256.

    The training directory is a selection boundary, not a second corpus. This function never
    parses, embeds, or indexes its PDFs; it only verifies that every selected byte-identical file
    already exists in the complete corpus registry.
    """
    training_dir = settings.path("paths.training_pdf_dir")
    pdf_files = sorted(
        (
            path for path in training_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in training selection directory: {training_dir}"
        )

    target = (
        int(expected_documents)
        if expected_documents is not None
        else int(settings.get("teacher.expected_training_documents"))
    )
    if target < 2:
        raise ValueError("The training selection requires at least two documents for splitting")
    if len(pdf_files) != target:
        raise ValueError(
            f"Training selection contains {len(pdf_files)} PDFs; expected exactly {target}. "
            "Use --expected-documents only for an intentional smoke test."
        )

    registry_path = settings.path("paths.document_registry")
    documents = list(read_jsonl(registry_path))
    documents_by_hash: dict[str, dict[str, Any]] = {}
    duplicate_registry_hashes: set[str] = set()
    for document in documents:
        file_hash = str(document["sha256"]).lower()
        if file_hash in documents_by_hash:
            duplicate_registry_hashes.add(file_hash)
        documents_by_hash[file_hash] = document
    if duplicate_registry_hashes:
        raise RuntimeError(
            "Corpus registry contains duplicate SHA-256 identities: "
            f"{sorted(duplicate_registry_hashes)[:10]}"
        )

    selected: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_hashes: dict[str, str] = {}
    missing: list[str] = []
    duplicates: list[tuple[str, str]] = []
    rejected: list[str] = []
    for path in pdf_files:
        file_hash = sha256_file(path).lower()
        if file_hash in selected_hashes:
            duplicates.append((selected_hashes[file_hash], path.name))
            continue
        selected_hashes[file_hash] = path.name
        document = documents_by_hash.get(file_hash)
        if document is None:
            missing.append(path.name)
            continue
        if document.get("metadata_status") == "rejected":
            rejected.append(path.name)
            continue
        selected.append(document)
        selection_rows.append(
            {
                "document_id": document["document_id"],
                "sha256": file_hash,
                "training_source_file": path.name,
                "corpus_source_file": document["source_file"],
                "title": document["title"],
                "theme": document.get("theme"),
                "metadata_status": document.get("metadata_status"),
            }
        )

    errors: list[str] = []
    if missing:
        errors.append(f"not present byte-for-byte in Papers_RAG/: {missing[:20]}")
    if duplicates:
        errors.append(f"duplicate training file content: {duplicates[:20]}")
    if rejected:
        errors.append(f"rejected corpus documents selected for training: {rejected[:20]}")
    if errors:
        raise ValueError("Invalid training selection; " + "; ".join(errors))

    benchmark_ids = _benchmark_document_ids(settings.path("paths.benchmark_file"))
    heldout_ids, planned_sft_ids, heldout_digest = _heldout_plan(settings)
    selected_ids = {str(document["document_id"]) for document in selected}
    overlap = sorted(selected_ids & (benchmark_ids | heldout_ids))
    if overlap:
        raise ValueError(
            "Training selection contains benchmark-held-out documents: " f"{overlap[:20]}"
        )
    if planned_sft_ids and selected_ids != planned_sft_ids:
        missing_planned = sorted(planned_sft_ids - selected_ids)
        unexpected = sorted(selected_ids - planned_sft_ids)
        raise ValueError(
            "Training selection does not match the frozen SFT pool; "
            f"missing={missing_planned[:20]}, unexpected={unexpected[:20]}"
        )
    if len(selected) != target:
        raise RuntimeError(
            f"Validated {len(selected)} unique training documents after expecting {target}"
        )

    selection_rows.sort(key=lambda row: (str(row["document_id"]), str(row["training_source_file"])))
    manifest = {
        "selection_method": "operator_curated_sha256_match",
        "training_pdf_dir": str(training_dir),
        "corpus_document_registry": str(registry_path),
        "expected_document_count": target,
        "selected_document_count": len(selected),
        "selection_digest": content_version(selection_rows),
        "excluded_benchmark_document_ids": sorted(benchmark_ids),
        "excluded_heldout_document_ids": sorted(heldout_ids),
        "heldout_selection_digest": heldout_digest,
        "documents": selection_rows,
    }
    write_json_atomic(settings.path("paths.training_selection_manifest"), manifest)
    return selected, manifest


def build_bundles(
    chunks: list[dict[str, Any]],
    allowed_document_ids: set[str],
    multi_source_fraction: float,
    seed: int,
    chunks_per_document: int = 8,
    minimum_substantive_chunks: int = 4,
) -> list[EvidenceBundle]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if chunk["document_id"] in allowed_document_ids and _is_substantive_chunk(chunk):
            by_document[chunk["document_id"]].append(chunk)

    rng = random.Random(seed)  # noqa: S311 - deterministic bundle sampling, not security
    document_ids = sorted(
        document_id
        for document_id, document_chunks in by_document.items()
        if len(document_chunks) >= minimum_substantive_chunks
    )
    bundles: list[EvidenceBundle] = []
    for document_id in document_ids:
        selected = _stratified_chunks(
            by_document[document_id], count=chunks_per_document
        )
        bundles.append(
            EvidenceBundle(
                bundle_id="bundle_"
                + stable_digest(
                    document_id,
                    "single",
                    *(str(chunk["chunk_id"]) for chunk in selected),
                ),
                document_ids=(document_id,),
                chunks=selected,
            )
        )

    desired_multi = math.ceil(len(bundles) * multi_source_fraction)
    candidates = document_ids[:]
    rng.shuffle(candidates)
    for index in range(desired_multi):
        first = candidates[index % len(candidates)]
        second = candidates[(index + 1) % len(candidates)]
        if first == second:
            continue
        per_source = max(2, chunks_per_document // 2)
        combined = _stratified_chunks(by_document[first], per_source) + _stratified_chunks(
            by_document[second], per_source
        )
        rng.shuffle(combined)
        bundles.append(
            EvidenceBundle(
                bundle_id="bundle_"
                + stable_digest(
                    first,
                    second,
                    "multi",
                    *(str(chunk["chunk_id"]) for chunk in combined),
                ),
                document_ids=tuple(sorted((first, second))),
                chunks=combined,
            )
        )
    rng.shuffle(bundles)
    return bundles


def _teacher_schema(exact_items: int = 2) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "giis_sft_examples",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["examples"],
                "properties": {
                    "examples": {
                        "type": "array",
                        "minItems": exact_items,
                        "maxItems": exact_items,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "question",
                                "answer",
                                "response_mode",
                                "answer_style",
                            ],
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                                "response_mode": {
                                    "type": "string",
                                    "enum": ["standard", "long_form"],
                                },
                                "answer_style": {
                                    "type": "string",
                                    "enum": sorted(ANSWER_STYLES),
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def _grounding_schema(
    exact_items: int = 2, maximum_evidence_index: int = 1
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "giis_semantic_verdicts",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["verdicts"],
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "minItems": exact_items,
                        "maxItems": exact_items,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "candidate_id",
                                "fully_answers_question",
                                "speculative",
                                "relies_on_secondary_source",
                                "claim_verdicts",
                            ],
                            "properties": {
                                "candidate_id": {"type": "string", "minLength": 1},
                                "fully_answers_question": {"type": "boolean"},
                                "speculative": {"type": "boolean"},
                                "relies_on_secondary_source": {"type": "boolean"},
                                "claim_verdicts": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["claim_id", "entailed", "reason", "supports"],
                                        "properties": {
                                            "claim_id": {"type": "string", "minLength": 1},
                                            "entailed": {"type": "boolean"},
                                            "reason": {
                                                "type": "string",
                                                "enum": [
                                                    "entailed",
                                                    "unsupported",
                                                    "contradicted",
                                                    "insufficient_context",
                                                ],
                                            },
                                            "supports": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "additionalProperties": False,
                                                    "required": [
                                                        "evidence_id",
                                                        "supporting_quote",
                                                    ],
                                                    "properties": {
                                                        "evidence_id": {
                                                            "type": "integer",
                                                            "minimum": 1,
                                                            "maximum": maximum_evidence_index,
                                                        },
                                                        "supporting_quote": {
                                                            "type": "string",
                                                            "minLength": 1,
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def _normalize_supporting_text(value: Any) -> str:
    """Normalize PDF/JSON whitespace without weakening exact substring verification."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _render_grounded_segments(
    segments: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> str:
    """Validate provenance and append markers without reparsing scientific prose."""
    if not chunks:
        raise ValueError("Grounding verifier cannot render without evidence chunks")
    normalized_chunks = [_normalize_supporting_text(chunk.get("text") or "") for chunk in chunks]
    rendered: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Grounding verifier returned an invalid evidence segment")
        text = str(segment.get("text") or "").strip()
        separator_before = segment.get("separator_before")
        supports = segment.get("supports")
        if (
            not text
            or separator_before not in {"space", "newline", "blank_line"}
            or not isinstance(supports, list)
            or not supports
            or _EVIDENCE_LIKE_BRACKET.search(text)
        ):
            raise ValueError("Grounding verifier returned an invalid evidence segment")
        evidence_ids: list[int] = []
        for support in supports:
            if not isinstance(support, dict):
                raise ValueError("Grounding verifier returned an invalid evidence segment")
            evidence_id = support.get("evidence_id")
            quote = support.get("supporting_quote")
            normalized_quote = (
                _normalize_supporting_text(quote) if isinstance(quote, str) else ""
            )
            if (
                isinstance(evidence_id, bool)
                or not isinstance(evidence_id, int)
                or not 1 <= evidence_id <= len(chunks)
                or not normalized_quote
                or normalized_quote not in normalized_chunks[evidence_id - 1]
            ):
                raise ValueError("Grounding verifier returned an invalid evidence segment")
            evidence_ids.append(evidence_id)
        markers = "".join(f"[E{index}]" for index in dict.fromkeys(evidence_ids))
        punctuation = re.search(r"([.!?])([\"'\)\]\}]*)$", text)
        if punctuation:
            marker_start = punctuation.start(1)
            rendered_text = f"{text[:marker_start].rstrip()} {markers}{text[marker_start:]}"
        else:
            rendered_text = f"{text.rstrip()} {markers}"
        if rendered:
            separator = {"space": " ", "newline": "\n", "blank_line": "\n\n"}[
                separator_before
            ]
            rendered_text = separator + rendered_text
        rendered.append(rendered_text)
    if not rendered:
        raise ValueError("Grounding verifier returned no supported evidence segments")
    return "".join(rendered)


class TeacherDatasetBuilder:
    def __init__(self, settings: Settings):
        from openai import OpenAI
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        self.settings = settings
        self.teacher = settings.section("teacher")
        temperature = float(self.teacher["temperature"])
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Teacher temperature must be between 0.0 and 1.0")
        minimum_core = int(self.teacher["minimum_core_examples_per_document"])
        maximum_core = int(self.teacher["maximum_core_examples_per_document"])
        if not 1 <= minimum_core <= maximum_core:
            raise ValueError("Core example allocation must satisfy 1 <= minimum <= maximum")
        if int(self.teacher.get("maximum_core_requests_per_document", 100)) < 1:
            raise ValueError("maximum_core_requests_per_document must be positive")
        if int(self.teacher.get("maximum_zero_yield_requests_before_minimum", 2)) < 1:
            raise ValueError("maximum_zero_yield_requests_before_minimum must be positive")
        if int(self.teacher.get("maximum_transient_retries_per_request", 2)) < 0:
            raise ValueError("maximum_transient_retries_per_request cannot be negative")
        generation_concurrency = int(self.teacher.get("generation_concurrency", 1))
        if not 1 <= generation_concurrency <= 8:
            raise ValueError("generation_concurrency must be between 1 and 8")
        if float(self.teacher.get("transient_retry_backoff_seconds", 1.0)) < 0:
            raise ValueError("transient_retry_backoff_seconds cannot be negative")
        if int(self.teacher.get("verification_batch_size", 2)) < 1:
            raise ValueError("verification_batch_size must be positive")
        if int(self.teacher.get("verification_max_output_tokens", 3000)) < 1:
            raise ValueError("verification_max_output_tokens must be positive")
        verification_reasoning_effort = str(
            self.teacher.get("verification_reasoning_effort", "low")
        ).strip().lower()
        if verification_reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "verification_reasoning_effort must be low, medium, or high"
            )
        if int(self.teacher.get("maximum_verification_attempts", 2)) < 1:
            raise ValueError("maximum_verification_attempts must be positive")
        if int(self.teacher.get("maximum_consecutive_verification_quarantines", 3)) < 1:
            raise ValueError(
                "maximum_consecutive_verification_quarantines must be positive"
            )
        api_key = os.getenv("TEACHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TEACHER_API_KEY must be set; use a local-only placeholder if applicable"
            )
        self.client = OpenAI(
            base_url=self.teacher["base_url"],
            api_key=api_key,
            timeout=float(self.teacher["request_timeout_seconds"]),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(settings.get("models.base"))
        self.question_encoder = SentenceTransformer(
            settings.get("models.embedding"), device="cpu"
        )
        self.timing = {"generation_seconds": 0.0, "verification_seconds": 0.0}
        self.audit_journal: DatasetAuditJournal | None = None
        self._consecutive_verification_quarantines = 0
        self._audit_lock = Lock()
        self._timing_lock = Lock()
        # A/B infrastructure may overlap HTTP generation calls. Keep the CPU embedding model's
        # semantic-duplicate pass serialized so concurrency changes only teacher serving/dispatch.
        self._prevalidation_lock = Lock()

    def _audit(self, event: str, **fields: Any) -> None:
        journal = getattr(self, "audit_journal", None)
        if journal is not None:
            audit_lock = getattr(self, "_audit_lock", None)
            if audit_lock is None:
                journal.append(event, **fields)
            else:
                with audit_lock:
                    journal.append(event, **fields)

    @staticmethod
    def _token_count(encoded: Any) -> int:
        """Return sequence length for tokenizer lists, tensors, or BatchEncoding objects."""
        input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
        if input_ids is None:
            raise ValueError("Tokenizer output does not contain input_ids")
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape):
            return int(shape[-1])
        if len(input_ids) and hasattr(input_ids[0], "__len__"):
            return len(input_ids[0])
        return len(input_ids)

    def _chat_token_count(self, messages: list[dict[str, str]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        return self._token_count(encoded)

    @staticmethod
    def _assign_candidate_ids(
        bundle: EvidenceBundle, examples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Assign each locally generated candidate an immutable request-scoped identity."""
        seen: set[str] = set()
        for position, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError("Teacher returned a non-object candidate")
            candidate_id = example.get("candidate_id")
            if candidate_id is None:
                candidate_id = "candidate_" + stable_digest(
                    bundle.bundle_id,
                    str(position),
                    str(example.get("question") or ""),
                    str(example.get("answer") or ""),
                    str(example.get("response_mode") or ""),
                )
                example["candidate_id"] = candidate_id
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError("Candidate identity must be a non-empty string")
            if candidate_id in seen:
                raise ValueError("Candidate identities must be unique within a request")
            seen.add(candidate_id)
        return examples

    def _request_examples_with_retries(
        self,
        bundle: EvidenceBundle,
        accepted_questions: list[str] | None = None,
        *,
        allow_long_form: bool = True,
        requested_examples: int = 2,
        partition: str | None = None,
        request_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retry request/protocol failures without spending the quality zero-yield budget."""
        retries = int(self.teacher.get("maximum_transient_retries_per_request", 2))
        backoff = float(self.teacher.get("transient_retry_backoff_seconds", 1.0))
        for attempt in range(retries + 1):
            try:
                return self._request_examples(
                    bundle,
                    accepted_questions=accepted_questions,
                    allow_long_form=allow_long_form,
                    requested_examples=requested_examples,
                    partition=partition,
                    request_number=request_number,
                )
            except TeacherInfrastructureError:
                # The semantic verifier already exhausted its own bounded retries while
                # preserving the generated candidates. Regenerating here would waste work and
                # change candidate identities, so propagate the failure unchanged.
                raise
            except Exception as exc:
                self._audit(
                    "infrastructure_error",
                    partition=partition,
                    request_number=request_number,
                    bundle_id=bundle.bundle_id,
                    document_ids=list(bundle.document_ids),
                    attempt=attempt + 1,
                    maximum_attempts=retries + 1,
                    error_type=type(exc).__name__,
                    error_message=re.sub(r"\s+", " ", str(exc)).strip()[:500],
                )
                if attempt >= retries:
                    raise TeacherInfrastructureError(
                        "Teacher request or structured-output protocol failed after "
                        f"{attempt + 1} attempt(s): {type(exc).__name__}: {exc}"
                    ) from exc
                print(
                    "[dataset] retrying teacher infrastructure failure "
                    f"bundle={bundle.bundle_id} attempt={attempt + 1}/{retries + 1} "
                    f"error={type(exc).__name__}",
                    flush=True,
                )
                if backoff:
                    time.sleep(backoff * (2**attempt))
        raise AssertionError("unreachable")

    def _multi_source_fraction_for_run(
        self, expected_documents: int | None
    ) -> tuple[float, bool]:
        timed_smoke = (
            bool(self.teacher.get("timed_smoke_disable_multi_source", True))
            and expected_documents is not None
            and expected_documents <= int(self.teacher.get("timed_smoke_max_documents", 5))
        )
        fraction = 0.0 if timed_smoke else float(self.teacher["multi_source_fraction"])
        return fraction, timed_smoke

    def _prevalidate_for_verifier(
        self,
        bundle: EvidenceBundle,
        examples: list[dict[str, Any]],
        accepted_questions: list[str],
    ) -> list[dict[str, Any]]:
        """Discard cheap failures and semantic duplicates before an expensive teacher audit."""
        viable: list[dict[str, Any]] = []
        for example in examples:
            existing_reason = example.get("_prevalidation_rejection_reason")
            valid, reason = (
                (False, str(existing_reason))
                if existing_reason
                else self._validate_before_verifier(bundle, example)
            )
            if valid:
                viable.append(example)
            else:
                example["_prevalidation_rejection_reason"] = reason

        if not viable:
            return []

        prior = [str(question).strip() for question in accepted_questions if str(question).strip()]
        questions = [str(example.get("question") or "").strip() for example in viable]
        all_questions = prior + questions
        question_encoder = getattr(self, "question_encoder", None)
        embeddings = (
            question_encoder.encode(
                all_questions,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if question_encoder is not None
            else None
        )
        lexical_threshold = float(self.teacher.get("maximum_question_similarity", 0.72))
        embedding_threshold = float(
            self.teacher.get("maximum_question_embedding_similarity", 0.88)
        )
        kept: list[dict[str, Any]] = []
        kept_indices: list[int] = []
        prior_indices = list(range(len(prior)))
        for offset, example in enumerate(viable):
            index = len(prior) + offset
            comparison_indices = prior_indices + kept_indices
            lexical_score = max(
                (
                    _question_similarity(all_questions[index], all_questions[other])
                    for other in comparison_indices
                ),
                default=0.0,
            )
            embedding_score = max(
                (
                    float(embeddings[index] @ embeddings[other])
                    for other in comparison_indices
                ),
                default=0.0,
            ) if embeddings is not None else 0.0
            example["_maximum_lexical_similarity"] = round(lexical_score, 6)
            example["_maximum_embedding_similarity"] = round(embedding_score, 6)
            if lexical_score >= lexical_threshold:
                example["_prevalidation_rejection_reason"] = "lexical_duplicate_question"
            elif embedding_score >= embedding_threshold:
                example["_prevalidation_rejection_reason"] = "semantic_duplicate_question"
            else:
                kept.append(example)
                kept_indices.append(index)
        return kept

    @staticmethod
    def _training_messages(
        bundle: EvidenceBundle, question: str, answer: str
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Answer only from supplied GIIS corpus evidence. If evidence is absent, "
                    "respond exactly: No corpus evidence. Cite claims using [E#] markers."
                ),
            },
            {
                "role": "user",
                "content": f"Evidence:\n{bundle.context()}\n\nQuestion: {question}",
            },
            {"role": "assistant", "content": answer},
        ]

    def _generation_request_options(
        self,
        bundle: EvidenceBundle,
        accepted_questions: list[str] | None = None,
        *,
        allow_long_form: bool = True,
        requested_examples: int = 2,
    ) -> dict[str, Any]:
        context = bundle.context()
        novelty_instruction = ""
        if accepted_questions:
            prior_questions = "\n".join(f"- {question}" for question in accepted_questions)
            novelty_instruction = f"""

The following questions have already passed validation for this paper. Produce materially distinct
questions that do not repeat or closely paraphrase them:
{prior_questions}
"""
        minimum_words = int(self.settings.get("response.minimum_long_form_words"))
        target_words = int(self.settings.get("response.target_long_form_words"))
        maximum_words = int(self.settings.get("response.maximum_long_form_words"))
        allowed_markers = ", ".join(
            f"[E{index}]" for index in range(1, len(bundle.chunks) + 1)
        )
        source_instruction = (
            "- This is a multi-source bundle. Each question must require synthesis across both "
            "documents, and each answer must cite evidence from both document IDs."
            if len(bundle.document_ids) > 1
            else (
                "- This is a single-source bundle. Keep every question answerable from this "
                "paper alone."
            )
        )
        mode_instruction = (
            f"Produce {requested_examples} focused standard answers by default. You may make at "
            "most one example "
            f"long_form only when the supplied evidence can sustain at least {minimum_words} "
            "non-repetitive grounded words; otherwise keep both examples standard."
            if allow_long_form
            else (
                f"Produce exactly {requested_examples} focused standard answers; do not produce "
                "a long_form example."
            )
        )
        prompt = f"""Create exactly {requested_examples} high-quality training examples for a
corpus-grounded scientific research assistant.

Rules:
- Use only the supplied evidence; never add foundational or outside knowledge.
- Cite every factual sentence with one or more exact evidence markers placed in that sentence.
- The only allowed evidence markers are: {allowed_markers}. Use at least one allowed marker in every
  answer, and never invent a marker outside this list.
- Use separate markers such as [E1][E3], never combined forms such as [E1, E3], and never insert
  invisible characters inside a marker.
- Do not add mechanisms, numerical values, methods, regional claims, recommendations, implications,
  or policy actions unless they are explicitly supported by the supplied evidence.
- If the evidence cannot support a claim, omit the claim rather than making a plausible inference.
- Never cite or directly attribute a claim to a study, author, report, or numbered reference merely
  mentioned inside a corpus paper. Such a second-hand source is not evidence unless it appears in
  this bundle under its own Document ID. Omit second-hand claims instead of reproducing the nested
  citation; every citation must resolve to an ingested corpus document.
{source_instruction}
- {mode_instruction}
- The long-form answer must contain at least {minimum_words} words.
- Aim for {target_words}-{maximum_words} words when the evidence supports useful additional depth;
  never pad, repeat claims, or introduce unsupported information merely to increase length.
- If evidence is insufficient for a proposed question, choose a different question.
- Every question must be fully answerable from the supplied excerpts. Never ask for an exhaustive
  list, complete set, comparison, or explanation when any requested part is absent from evidence.
- Multi-source questions may compare directly reported facts or methods only. Never ask for or
  propose cross-domain adaptation, recommendations, interventions, policy transfer, or hypothetical
  application unless every element is explicitly stated in the evidence.
- Synthesize by theme instead of listing extracts.
- Write natural, human-friendly prose while preserving scientific accuracy.
- Set answer_style to prose, sectioned, table, comparison, causal_chain, or concept_diagram.
- Use table, comparison, or causal_chain only when every row or step can carry an evidence marker.
- Use concept_diagram only as one safe Mermaid flowchart with 2-20 explicitly defined nodes, at
  least one edge, and an evidence marker inside every node label. Use only node definitions and
  unlabelled arrows; never use URLs, click directives, HTML, scripts, comments, or styling commands.
- Vary sentence rhythm and organization according to the question; do not force every answer into
  rigid journal-style etiquette.
- Use headings, bullets, or a compact table only when they materially improve comprehension.
- Do not expose hidden reasoning or chain-of-thought.
{novelty_instruction}

EVIDENCE:
{context}
"""
        request_options: dict[str, Any] = {}
        if "enable_thinking" in self.teacher:
            request_options["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": bool(self.teacher["enable_thinking"])
                }
            }
        return {
            "model": self.settings.get("models.teacher"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate strictly grounded synthetic training data in the "
                        "requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": _teacher_schema(exact_items=requested_examples),
            "temperature": float(self.teacher["temperature"]),
            "max_tokens": int(self.teacher["max_output_tokens"]),
            **request_options,
        }

    def _request_examples(
        self,
        bundle: EvidenceBundle,
        accepted_questions: list[str] | None = None,
        *,
        allow_long_form: bool = True,
        requested_examples: int = 2,
        partition: str | None = None,
        request_number: int | None = None,
    ) -> list[dict[str, Any]]:
        request = self._generation_request_options(
            bundle,
            accepted_questions=accepted_questions,
            allow_long_form=allow_long_form,
            requested_examples=requested_examples,
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        generation_elapsed = time.perf_counter() - started
        timing = getattr(self, "timing", None)
        if timing is not None:
            timing_lock = getattr(self, "_timing_lock", None)
            if timing_lock is None:
                timing["generation_seconds"] += generation_elapsed
            else:
                with timing_lock:
                    timing["generation_seconds"] += generation_elapsed
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        self._audit(
            "generation_response_received",
            partition=partition,
            request_number=request_number,
            bundle_id=bundle.bundle_id,
            document_ids=list(bundle.document_ids),
            finish_reason=str(
                getattr(response.choices[0], "finish_reason", None) or "unknown"
            ),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            response_characters=len(content or ""),
            elapsed_seconds=round(generation_elapsed, 3),
        )
        if not content:
            raise ValueError("Teacher returned empty structured output")
        payload = json.loads(content)
        examples = list(payload["examples"])
        if len(examples) != requested_examples:
            raise ValueError("Teacher changed the requested candidate count")
        self._assign_candidate_ids(bundle, examples)
        prevalidation_lock = getattr(self, "_prevalidation_lock", None)
        if prevalidation_lock is None:
            viable = self._prevalidate_for_verifier(
                bundle, examples, accepted_questions or []
            )
        else:
            with prevalidation_lock:
                viable = self._prevalidate_for_verifier(
                    bundle, examples, accepted_questions or []
                )
        self._audit(
            "generation_completed",
            partition=partition,
            request_number=request_number,
            bundle_id=bundle.bundle_id,
            document_ids=list(bundle.document_ids),
            requested_candidates=requested_examples,
            generated_candidates=len(examples),
            structurally_viable_candidates=len(viable),
            elapsed_seconds=round(generation_elapsed, 3),
        )
        for example in examples:
            reason = example.get("_prevalidation_rejection_reason")
            self._audit(
                "candidate_prevalidation",
                partition=partition,
                request_number=request_number,
                bundle_id=bundle.bundle_id,
                document_ids=list(bundle.document_ids),
                candidate_id=str(example.get("candidate_id") or ""),
                question=str(example.get("question") or "").strip(),
                response_mode=str(example.get("response_mode") or ""),
                answer_style=str(example.get("answer_style") or "prose"),
                decision="rejected" if reason else "passed",
                reason=str(reason or "ok"),
                maximum_lexical_similarity=float(
                    example.get("_maximum_lexical_similarity") or 0.0
                ),
                maximum_embedding_similarity=float(
                    example.get("_maximum_embedding_similarity") or 0.0
                ),
            )
        semantic_verification_enabled = bool(
            self.teacher.get("semantic_verification_enabled", True)
        )
        reviewed = (
            self._verify_grounding(
                bundle,
                viable,
                partition=partition,
                request_number=request_number,
            )
            if viable and semantic_verification_enabled
            else viable
        )
        if viable and not semantic_verification_enabled:
            self._audit(
                "semantic_verification_deferred",
                partition=partition,
                request_number=request_number,
                bundle_id=bundle.bundle_id,
                document_ids=list(bundle.document_ids),
                candidate_ids=[str(row.get("candidate_id") or "") for row in viable],
                review_status="pending_manual",
            )
        reviewed_by_id = {str(row.get("candidate_id") or ""): row for row in reviewed}
        return [
            reviewed_by_id.get(str(row.get("candidate_id") or ""), row)
            for row in examples
        ]

    def _verify_grounding(
        self,
        bundle: EvidenceBundle,
        examples: list[dict[str, Any]],
        *,
        partition: str | None = None,
        request_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run compact semantic adjudication in bounded batches; never rewrite answers."""
        if not examples:
            return []
        self._assign_candidate_ids(bundle, examples)
        batch_size = int(self.teacher.get("verification_batch_size", 2))
        maximum_attempts = int(
            self.teacher.get(
                "maximum_verification_attempts",
                int(self.teacher.get("maximum_transient_retries_per_request", 2)) + 1,
            )
        )
        backoff = float(self.teacher.get("transient_retry_backoff_seconds", 1.0))
        maximum_quarantines = int(
            self.teacher.get("maximum_consecutive_verification_quarantines", 3)
        )

        def quarantine_batch(
            batch: list[dict[str, Any]], failure: Exception, attempts: int
        ) -> list[dict[str, Any]]:
            consecutive = int(
                getattr(self, "_consecutive_verification_quarantines", 0)
            ) + 1
            self._consecutive_verification_quarantines = consecutive
            candidate_ids = [str(row.get("candidate_id") or "") for row in batch]
            self._audit(
                "verification_batch_quarantined",
                partition=partition,
                request_number=request_number,
                bundle_id=bundle.bundle_id,
                candidate_ids=candidate_ids,
                attempts=attempts,
                error_type=type(failure).__name__,
                error_message=re.sub(r"\s+", " ", str(failure)).strip()[:500],
                consecutive_quarantined_batches=consecutive,
                maximum_consecutive_quarantined_batches=maximum_quarantines,
            )
            if consecutive > maximum_quarantines:
                raise TeacherInfrastructureError(
                    "Semantic verifier quarantine circuit breaker opened after "
                    f"{consecutive} consecutive failed batches"
                ) from failure

            quarantined: list[dict[str, Any]] = []
            for original in batch:
                candidate = dict(original)
                total_claims = len(_claim_units(str(original.get("answer") or "")))
                candidate["_grounding_supported"] = False
                candidate["_grounding_rejection_reason"] = (
                    "semantic_verification_unavailable"
                )
                candidate["_grounding_reason_codes"] = [
                    "semantic_verification_unavailable"
                ]
                candidate["_verified_claim_count"] = 0
                candidate["_total_claim_count"] = total_claims
                quarantined.append(candidate)
                self._audit(
                    "candidate_semantic_verdict",
                    partition=partition,
                    request_number=request_number,
                    bundle_id=bundle.bundle_id,
                    document_ids=list(bundle.document_ids),
                    candidate_id=str(candidate.get("candidate_id") or ""),
                    decision="rejected",
                    reason_codes=["semantic_verification_unavailable"],
                    supported_claims=0,
                    total_claims=total_claims,
                    verification_batch_size=len(batch),
                    verification_quarantined=True,
                )
            return quarantined

        def quarantinable_response_failure(failure: Exception) -> bool:
            return isinstance(failure, json.JSONDecodeError) or (
                isinstance(failure, ValueError)
                and str(failure) == "Teacher returned empty grounding-verification output"
            )

        def verify_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for attempt in range(maximum_attempts):
                try:
                    reviewed = self._verify_grounding_batch(
                        bundle,
                        batch,
                        partition=partition,
                        request_number=request_number,
                    )
                    self._consecutive_verification_quarantines = 0
                    return reviewed
                except json.JSONDecodeError as exc:
                    self._audit(
                        "verification_schema_error",
                        partition=partition,
                        request_number=request_number,
                        bundle_id=bundle.bundle_id,
                        candidate_ids=[str(row.get("candidate_id") or "") for row in batch],
                        attempt=attempt + 1,
                        maximum_attempts=maximum_attempts,
                        error_type=type(exc).__name__,
                    )
                    if len(batch) > 1:
                        midpoint = len(batch) // 2
                        return verify_batch(batch[:midpoint]) + verify_batch(batch[midpoint:])
                    failure: Exception = exc
                except Exception as exc:
                    failure = exc
                    self._audit(
                        "verification_infrastructure_error",
                        partition=partition,
                        request_number=request_number,
                        bundle_id=bundle.bundle_id,
                        candidate_ids=[str(row.get("candidate_id") or "") for row in batch],
                        attempt=attempt + 1,
                        maximum_attempts=maximum_attempts,
                        error_type=type(exc).__name__,
                        error_message=re.sub(r"\s+", " ", str(exc)).strip()[:500],
                    )
                    if quarantinable_response_failure(exc) and len(batch) > 1:
                        midpoint = len(batch) // 2
                        return verify_batch(batch[:midpoint]) + verify_batch(
                            batch[midpoint:]
                        )
                if attempt + 1 >= maximum_attempts:
                    if quarantinable_response_failure(failure):
                        return quarantine_batch(batch, failure, attempt + 1)
                    raise TeacherInfrastructureError(
                        "Semantic verifier failed after "
                        f"{attempt + 1} attempt(s) for {len(batch)} preserved candidate(s): "
                        f"{type(failure).__name__}: {failure}"
                    ) from failure
                if backoff:
                    time.sleep(backoff * (2**attempt))
            raise AssertionError("unreachable")

        reviewed: list[dict[str, Any]] = []
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset : offset + batch_size]
            reviewed.extend(verify_batch(batch))
        return reviewed

    def _verify_grounding_batch(
        self,
        bundle: EvidenceBundle,
        examples: list[dict[str, Any]],
        *,
        partition: str | None = None,
        request_number: int | None = None,
    ) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        claims_by_candidate: dict[str, list[dict[str, Any]]] = {}
        for example in examples:
            candidate_id = str(example.get("candidate_id") or "")
            claims = _claim_units(str(example.get("answer") or ""))
            if not candidate_id or not claims:
                raise ValueError("Grounding request contains an invalid candidate packet")
            claims_by_candidate[candidate_id] = claims
            packets.append(
                {
                    "candidate_id": candidate_id,
                    "question": str(example.get("question") or "").strip(),
                    "response_mode": str(example.get("response_mode") or ""),
                    "claims": claims,
                }
            )

        prompt = f"""Judge only semantic support and completeness. Do not rewrite any question,
answer, or claim, and do not explain the audit.

EVIDENCE:
{bundle.context()}

For every candidate packet below:
- Return every candidate_id and claim_id exactly once.
- Mark a claim entailed only when the supplied evidence directly supports its full meaning.
- For an entailed claim, return at least one evidence_id and a short supporting_quote copied exactly
  from that evidence chunk. When the claim already cites evidence IDs, use only those cited IDs.
- Mark unsupported, contradicted, or insufficient-context claims accordingly; do not repair them.
- fully_answers_question is true only when the answer addresses every requested part without relying
  on missing information, padding, or outside knowledge.
- speculative is true for unsupported mechanisms, implications, recommendations, interventions,
  cross-domain adaptation, policy transfer, or hypothetical applications.
- relies_on_secondary_source is true when the answer relies on a work merely mentioned or cited
  inside a corpus paper rather than an ingested Document ID in this evidence bundle.
- For multi-source packets, fully_answers_question is true only when both documents are used
  substantively where the question requires synthesis.

CANDIDATE PACKETS:
{json.dumps(packets, ensure_ascii=False)}
"""
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.settings.get("models.teacher"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict scientific claim-entailment judge. Return only the "
                        "requested compact schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format=_grounding_schema(
                exact_items=len(examples),
                maximum_evidence_index=len(bundle.chunks),
            ),
            reasoning_effort=str(
                self.teacher.get("verification_reasoning_effort", "low")
            ).strip().lower(),
            temperature=0.0,
            max_tokens=int(self.teacher.get("verification_max_output_tokens", 3000)),
        )
        verification_elapsed = time.perf_counter() - started
        timing = getattr(self, "timing", None)
        if timing is not None:
            timing["verification_seconds"] += verification_elapsed
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Teacher returned empty grounding-verification output")
        verdicts = list(json.loads(content)["verdicts"])
        if len(verdicts) != len(examples):
            raise ValueError("Grounding verifier changed the candidate count")

        expected_ids = [str(example.get("candidate_id") or "") for example in examples]
        if any(not candidate_id for candidate_id in expected_ids) or len(set(expected_ids)) != len(
            expected_ids
        ):
            raise ValueError("Grounding request contains invalid candidate identities")
        returned_ids = [
            str(verdict.get("candidate_id") or "")
            for verdict in verdicts
            if isinstance(verdict, dict)
        ]
        if len(returned_ids) != len(verdicts) or any(not value for value in returned_ids):
            raise ValueError("Grounding verifier omitted a candidate identity")
        if len(set(returned_ids)) != len(returned_ids):
            raise ValueError("Grounding verifier duplicated a candidate identity")
        missing_ids = set(expected_ids) - set(returned_ids)
        unknown_ids = set(returned_ids) - set(expected_ids)
        if missing_ids or unknown_ids:
            raise ValueError(
                "Grounding verifier changed candidate identities: "
                f"missing={sorted(missing_ids)} unknown={sorted(unknown_ids)}"
            )

        verdicts_by_id = {str(row["candidate_id"]): row for row in verdicts}
        normalized_chunks = [
            _normalize_supporting_text(chunk.get("text") or "") for chunk in bundle.chunks
        ]
        ordered: list[dict[str, Any]] = []
        for original in examples:
            candidate_id = str(original["candidate_id"])
            verdict = verdicts_by_id[candidate_id]
            expected_claims = claims_by_candidate[candidate_id]
            expected_claim_ids = [str(claim["claim_id"]) for claim in expected_claims]
            claim_rows = list(verdict.get("claim_verdicts") or [])
            returned_claim_ids = [
                str(row.get("claim_id") or "") for row in claim_rows if isinstance(row, dict)
            ]
            if (
                len(returned_claim_ids) != len(claim_rows)
                or len(set(returned_claim_ids)) != len(returned_claim_ids)
                or set(returned_claim_ids) != set(expected_claim_ids)
            ):
                raise ValueError(
                    f"Grounding verifier changed claim identities for {candidate_id}"
                )

            claim_rows_by_id = {str(row["claim_id"]): row for row in claim_rows}
            rejection_reasons: list[str] = []
            supported_claims = 0
            for claim in expected_claims:
                claim_id = str(claim["claim_id"])
                claim_verdict = claim_rows_by_id[claim_id]
                entailed = claim_verdict.get("entailed") is True
                reason = str(claim_verdict.get("reason") or "")
                if entailed != (reason == "entailed"):
                    raise ValueError(
                        f"Grounding verifier returned an inconsistent claim verdict for {claim_id}"
                    )
                provenance_valid = entailed
                supports = list(claim_verdict.get("supports") or [])
                if entailed and not supports:
                    provenance_valid = False
                cited_ids = set(claim.get("cited_evidence_ids") or [])
                for support in supports:
                    evidence_id = support.get("evidence_id") if isinstance(support, dict) else None
                    quote = support.get("supporting_quote") if isinstance(support, dict) else None
                    normalized_quote = (
                        _normalize_supporting_text(quote) if isinstance(quote, str) else ""
                    )
                    if (
                        isinstance(evidence_id, bool)
                        or not isinstance(evidence_id, int)
                        or not 1 <= evidence_id <= len(bundle.chunks)
                        or not normalized_quote
                        or normalized_quote not in normalized_chunks[evidence_id - 1]
                        or (cited_ids and evidence_id not in cited_ids)
                    ):
                        provenance_valid = False
                if entailed and provenance_valid:
                    supported_claims += 1
                else:
                    rejection_reasons.append(
                        "semantic_unsupported_claim"
                        if not entailed
                        else "invalid_verifier_provenance"
                    )

            if verdict.get("fully_answers_question") is not True:
                rejection_reasons.append("semantic_incomplete_answer")
            if verdict.get("speculative") is True:
                rejection_reasons.append("semantic_speculation")
            if verdict.get("relies_on_secondary_source") is True:
                rejection_reasons.append("semantic_secondary_source_reliance")
            rejection_reasons = list(dict.fromkeys(rejection_reasons))

            candidate = dict(original)
            candidate["_grounding_supported"] = not rejection_reasons
            candidate["_grounding_rejection_reason"] = (
                rejection_reasons[0] if rejection_reasons else "ok"
            )
            candidate["_grounding_reason_codes"] = rejection_reasons
            candidate["_verified_claim_count"] = supported_claims
            candidate["_total_claim_count"] = len(expected_claims)
            ordered.append(candidate)
            self._audit(
                "candidate_semantic_verdict",
                partition=partition,
                request_number=request_number,
                bundle_id=bundle.bundle_id,
                document_ids=list(bundle.document_ids),
                candidate_id=candidate_id,
                decision="accepted" if not rejection_reasons else "rejected",
                reason_codes=rejection_reasons or ["ok"],
                supported_claims=supported_claims,
                total_claims=len(expected_claims),
                verification_batch_size=len(examples),
                batch_elapsed_seconds=round(verification_elapsed, 3),
            )
        self._audit(
            "verification_batch_completed",
            partition=partition,
            request_number=request_number,
            bundle_id=bundle.bundle_id,
            candidate_count=len(examples),
            elapsed_seconds=round(verification_elapsed, 3),
        )
        return ordered

    def _checkpoint_policy_fingerprint(
        self,
        *,
        generation_policy_version: int = _GENERATION_POLICY_VERSION,
        minimum_long_form_words: int | None = None,
    ) -> str:
        policy = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "generation_policy_version": generation_policy_version,
            "secondary_citation_policy_version": 1,
            "semantic_verifier_protocol_version": _SEMANTIC_VERIFIER_PROTOCOL_VERSION,
            "teacher_model": str(self.settings.get("models.teacher")),
            "base_model": str(self.settings.get("models.base")),
            "temperature": float(self.teacher["temperature"]),
            "max_output_tokens": int(self.teacher["max_output_tokens"]),
            "initial_candidate_batch_size": int(
                self.teacher.get("initial_candidate_batch_size", 8)
            ),
            "refill_candidate_batch_size": int(
                self.teacher.get("refill_candidate_batch_size", 4)
            ),
            "minimum_core_examples_per_document": int(
                self.teacher["minimum_core_examples_per_document"]
            ),
            "maximum_core_examples_per_document": int(
                self.teacher["maximum_core_examples_per_document"]
            ),
            "maximum_zero_yield_requests_before_minimum": int(
                self.teacher.get("maximum_zero_yield_requests_before_minimum", 2)
            ),
            "maximum_transient_retries_per_request": int(
                self.teacher.get("maximum_transient_retries_per_request", 2)
            ),
            "transient_retry_backoff_seconds": float(
                self.teacher.get("transient_retry_backoff_seconds", 1.0)
            ),
            "verification_batch_size": int(
                self.teacher.get("verification_batch_size", 2)
            ),
            "verification_max_output_tokens": int(
                self.teacher.get("verification_max_output_tokens", 3000)
            ),
            "verification_reasoning_effort": str(
                self.teacher.get("verification_reasoning_effort", "low")
            ).strip().lower(),
            "maximum_verification_attempts": int(
                self.teacher.get("maximum_verification_attempts", 2)
            ),
            "maximum_consecutive_verification_quarantines": int(
                self.teacher.get("maximum_consecutive_verification_quarantines", 3)
            ),
            "evidence_chunks_per_document": int(
                self.teacher.get("evidence_chunks_per_document", 8)
            ),
            "minimum_substantive_chunks_per_document": int(
                self.teacher.get("minimum_substantive_chunks_per_document", 4)
            ),
            "minimum_training_citation_coverage": float(
                self.teacher.get("minimum_training_citation_coverage", 0.8)
            ),
            "maximum_question_similarity": float(
                self.teacher.get("maximum_question_similarity", 0.6)
            ),
            "maximum_question_embedding_similarity": float(
                self.teacher.get("maximum_question_embedding_similarity", 0.88)
            ),
            "multi_source_fraction": float(self.teacher.get("multi_source_fraction", 0.0)),
            "timed_smoke_max_documents": int(
                self.teacher.get("timed_smoke_max_documents", 5)
            ),
            "timed_smoke_disable_multi_source": bool(
                self.teacher.get("timed_smoke_disable_multi_source", True)
            ),
            "minimum_long_form_examples_per_document": int(
                self.teacher.get("minimum_long_form_examples_per_document", 0)
            ),
            "maximum_long_form_examples_per_document": int(
                self.teacher.get(
                    "maximum_long_form_examples_per_document",
                    self.teacher["maximum_core_examples_per_document"],
                )
            ),
            "minimum_long_form_words": (
                int(self.settings.get("response.minimum_long_form_words"))
                if minimum_long_form_words is None
                else minimum_long_form_words
            ),
            "training_max_length": int(self.settings.get("training.max_length")),
        }
        return content_version([policy])

    def _core_checkpoint_path(
        self,
        selection_digest: str,
        partition: str,
        document_id: str,
        *,
        policy_fingerprint: str | None = None,
    ) -> Path:
        return (
            self.settings.path("paths.sft_dir")
            / "checkpoints"
            / selection_digest
            / (policy_fingerprint or self._checkpoint_policy_fingerprint())
            / partition
            / "core"
            / f"{stable_digest(document_id)}.json"
        )

    def _load_core_checkpoint(
        self,
        selection_digest: str,
        partition: str,
        document_id: str,
        bundle: EvidenceBundle,
        minimum: int,
        maximum: int,
    ) -> dict[str, Any]:
        current_fingerprint = self._checkpoint_policy_fingerprint()
        path = self._core_checkpoint_path(selection_digest, partition, document_id)
        checkpoint_fingerprint = current_fingerprint
        if not path.exists():
            # Every policy fingerprint starts a clean generation namespace. Earlier accepted
            # records remain on disk as audit evidence but are not imported.
            return {
                "accepted": [],
                "accepted_questions": [],
                "rejected": [],
                "requests_completed": 0,
                "consecutive_empty_requests": 0,
                "adaptive_complete": False,
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read dataset checkpoint {path}: {exc}") from exc
        expected = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "selection_digest": selection_digest,
            "policy_fingerprint": checkpoint_fingerprint,
            "partition": partition,
            "document_id": document_id,
            "bundle_id": bundle.bundle_id,
            "minimum": minimum,
            "target": maximum,
        }
        mismatched = {
            key: {"expected": value, "found": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"Dataset checkpoint metadata mismatch at {path}: {mismatched}")
        accepted = list(payload.get("accepted") or [])
        accepted_questions = list(payload.get("accepted_questions") or [])
        rejected = list(payload.get("rejected") or [])
        if len(accepted) != len(accepted_questions) or len(accepted) > maximum:
            raise RuntimeError(f"Dataset checkpoint has inconsistent accepted records: {path}")
        for record, question in zip(accepted, accepted_questions, strict=True):
            messages = record.get("messages") or []
            if (
                record.get("split") != partition
                or record.get("dataset_kind") != "core_single_source"
                or record.get("document_ids") != [document_id]
                or record.get("bundle_id") != bundle.bundle_id
                or not messages
                or messages[-1].get("role") != "assistant"
            ):
                raise RuntimeError(f"Dataset checkpoint contains an invalid record: {path}")
            answer = str(messages[-1].get("content") or "").strip()
            example = {
                "question": str(question).strip(),
                "answer": answer,
                "response_mode": record.get("response_mode"),
            }
            valid, reason = self._validate_example(bundle, example)
            expected_id = "sft_" + stable_digest(
                bundle.bundle_id, example["question"], answer
            )
            if not valid or record.get("example_id") != expected_id:
                raise RuntimeError(
                    f"Dataset checkpoint record no longer validates at {path}: {reason}"
                )
        state = {
            "accepted": accepted,
            "accepted_questions": accepted_questions,
            "rejected": rejected,
            "requests_completed": int(payload.get("requests_completed") or 0),
            "consecutive_empty_requests": int(
                payload.get("consecutive_empty_requests") or 0
            ),
            "adaptive_complete": bool(payload.get("adaptive_complete"))
            or len(accepted) >= maximum,
        }
        zero_yield_limit = int(
            self.teacher.get("maximum_zero_yield_requests_before_minimum", 2)
        )
        release_floor = min(
            minimum,
            int(self.teacher.get("underfilled_release_floor", minimum)),
        )
        if release_floor < 1:
            raise ValueError("teacher.underfilled_release_floor must be positive")
        if len(accepted) < minimum and state["consecutive_empty_requests"] >= zero_yield_limit:
            if len(accepted) >= release_floor:
                # Exhausted documents may contribute fewer examples than the preferred
                # allocation, but only when every retained candidate still passes the
                # unchanged validation policy. Do not spend the same zero-yield budget
                # again merely because the process was restarted.
                state["adaptive_complete"] = True
                print(
                    "[dataset] resuming released underfilled checkpoint "
                    f"split={partition} document={document_id} "
                    f"accepted={len(accepted)}/{minimum} floor={release_floor}",
                    flush=True,
                )
            else:
                # Below the release floor, a deliberate new invocation receives another
                # bounded rescue budget without discarding its accepted candidates.
                state["consecutive_empty_requests"] = 0
                state["adaptive_complete"] = False
                print(
                    "[dataset] retrying underfilled checkpoint "
                    f"split={partition} document={document_id} "
                    f"accepted={len(accepted)}/{minimum} floor={release_floor}",
                    flush=True,
                )
        return state

    def _write_core_checkpoint(
        self,
        selection_digest: str,
        partition: str,
        document_id: str,
        bundle: EvidenceBundle,
        minimum: int,
        maximum: int,
        state: dict[str, Any],
    ) -> None:
        path = self._core_checkpoint_path(selection_digest, partition, document_id)
        write_json_atomic(
            path,
            {
                "checkpoint_version": _CHECKPOINT_VERSION,
                "selection_digest": selection_digest,
                "policy_fingerprint": self._checkpoint_policy_fingerprint(),
                "partition": partition,
                "document_id": document_id,
                "bundle_id": bundle.bundle_id,
                "minimum": minimum,
                "target": maximum,
                "accepted": state["accepted"],
                "accepted_questions": state["accepted_questions"],
                "rejected": state["rejected"],
                "requests_completed": state["requests_completed"],
                "consecutive_empty_requests": state["consecutive_empty_requests"],
                "adaptive_complete": state["adaptive_complete"],
            },
        )

    def _validate_before_verifier(
        self, bundle: EvidenceBundle, example: dict[str, Any]
    ) -> tuple[bool, str]:
        """Run every deterministic gate before spending semantic-verifier inference."""
        existing_reason = example.get("_prevalidation_rejection_reason")
        if existing_reason:
            return False, str(existing_reason)
        return self._validate_deterministic(bundle, example)

    def _validate_deterministic(
        self, bundle: EvidenceBundle, example: dict[str, Any]
    ) -> tuple[bool, str]:
        question = str(example.get("question") or "").strip()
        answer = str(example.get("answer") or "").strip()
        mode = str(example.get("response_mode") or "")
        if not question or not answer:
            return False, "empty_question_or_answer"
        if mode not in {"standard", "long_form"}:
            return False, "invalid_response_mode"
        style_valid, style_reason = validate_answer_style(example)
        if not style_valid:
            return False, style_reason
        evidence_like = set(_EVIDENCE_LIKE_BRACKET.findall(answer))
        malformed = {marker for marker in evidence_like if not _EVIDENCE_MARKER.fullmatch(marker)}
        if malformed:
            return False, "malformed_evidence_markers"
        allowed = {f"[E{index}]" for index in range(1, len(bundle.chunks) + 1)}
        used = set(_EVIDENCE_MARKER.findall(answer))
        if not used:
            return False, "missing_evidence_markers"
        if not used.issubset(allowed):
            return False, "out_of_range_evidence_markers"
        if contains_non_corpus_citation(answer):
            return False, "non_corpus_secondary_citation"
        if citation_coverage(answer) < float(
            self.teacher.get("minimum_training_citation_coverage", 0.8)
        ):
            return False, "insufficient_citation_coverage"
        word_count = len(answer.split())
        if mode == "long_form" and word_count < int(
            self.settings.get("response.minimum_long_form_words")
        ):
            return False, "long_form_too_short"
        if mode == "standard" and len(answer.split()) > 500:
            return False, "standard_answer_too_long"
        token_count = self._chat_token_count(self._training_messages(bundle, question, answer))
        if token_count > int(self.settings.get("training.max_length")):
            return False, "sample_exceeds_training_context"
        return True, "ok"

    def _validate_example(
        self, bundle: EvidenceBundle, example: dict[str, Any]
    ) -> tuple[bool, str]:
        prevalidation_reason = example.get("_prevalidation_rejection_reason")
        if prevalidation_reason:
            return False, str(prevalidation_reason)
        if example.get("_grounding_supported") is False:
            return False, str(
                example.get("_grounding_rejection_reason") or "grounding_verifier_rejected"
            )
        return self._validate_deterministic(bundle, example)

    def _records_for_partition(
        self,
        name: str,
        document_ids: set[str],
        chunks: list[dict[str, Any]],
        checkpoint_digest: str | None = None,
        multi_source_fraction: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        seed = int(self.teacher["seed"]) + (1 if name == "validation" else 0)
        bundles = build_bundles(
            chunks,
            allowed_document_ids=document_ids,
            multi_source_fraction=(
                float(self.teacher["multi_source_fraction"])
                if multi_source_fraction is None
                else multi_source_fraction
            ),
            seed=seed,
            chunks_per_document=int(
                self.teacher.get("evidence_chunks_per_document", 8)
            ),
            minimum_substantive_chunks=int(
                self.teacher.get("minimum_substantive_chunks_per_document", 4)
            ),
        )
        single_bundles = {
            bundle.document_ids[0]: bundle
            for bundle in bundles
            if len(bundle.document_ids) == 1
        }
        multi_bundles = [bundle for bundle in bundles if len(bundle.document_ids) > 1]
        if set(single_bundles) != document_ids:
            missing = sorted(document_ids - set(single_bundles))[:20]
            raise RuntimeError(f"Documents have no usable single-source evidence bundle: {missing}")

        minimum_per_document = int(
            self.teacher["minimum_core_examples_per_document"]
        )
        underfilled_release_floor = min(
            minimum_per_document,
            int(
                self.teacher.get(
                    "underfilled_release_floor", minimum_per_document
                )
            ),
        )
        if underfilled_release_floor < 1:
            raise ValueError("teacher.underfilled_release_floor must be positive")
        maximum_per_document = int(
            self.teacher["maximum_core_examples_per_document"]
        )
        minimum_long_form = int(
            self.teacher.get("minimum_long_form_examples_per_document", 0)
        )
        maximum_long_form = int(
            self.teacher.get(
                "maximum_long_form_examples_per_document", maximum_per_document
            )
        )
        if not 0 <= minimum_long_form <= maximum_long_form <= maximum_per_document:
            raise ValueError("Invalid per-document long-form allocation")
        counts: dict[str, int] = defaultdict(int)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        accepted_questions_by_scope: dict[tuple[str, ...], list[str]] = defaultdict(list)
        core_states: dict[str, dict[str, Any]] = {}

        def rejection_record(
            bundle: EvidenceBundle,
            reason: str,
            example: dict[str, Any] | None = None,
            detail: str | None = None,
        ) -> dict[str, Any]:
            record: dict[str, Any] = {"bundle_id": bundle.bundle_id, "reason": reason}
            if example is not None:
                record["question"] = str(example.get("question") or "").strip()
                record["response_mode"] = str(example.get("response_mode") or "")
                record["answer_style"] = str(example.get("answer_style") or "prose")
            if detail:
                record["detail"] = re.sub(r"\s+", " ", detail).strip()[:500]
            return record

        def accept_example(
            bundle: EvidenceBundle,
            example: dict[str, Any],
            dataset_kind: str,
            request_number: int,
        ) -> bool:
            question = str(example.get("question") or "").strip()
            scope = tuple(sorted(bundle.document_ids))

            def reject(reason: str) -> bool:
                rejected.append(rejection_record(bundle, reason, example))
                self._audit(
                    "candidate_final_decision",
                    partition=name,
                    request_number=request_number,
                    bundle_id=bundle.bundle_id,
                    document_ids=list(bundle.document_ids),
                    candidate_id=str(example.get("candidate_id") or ""),
                    question=question,
                    dataset_kind=dataset_kind,
                    decision="rejected",
                    reason=reason,
                )
                return False

            maximum_similarity = float(
                self.teacher.get("maximum_question_similarity", 0.6)
            )
            if question and any(
                _question_similarity(question, prior) >= maximum_similarity
                for prior in accepted_questions_by_scope[scope]
            ):
                return reject("lexical_duplicate_question")
            valid, reason = self._validate_example(bundle, example)
            if not valid:
                return reject(reason)
            question = example["question"].strip()
            mode = str(example["response_mode"])
            if dataset_kind == "core_single_source":
                existing_modes = [
                    str(record["response_mode"])
                    for record in accepted
                    if record["dataset_kind"] == "core_single_source"
                    and record["document_ids"] == [bundle.document_ids[0]]
                ]
                if mode == "long_form" and existing_modes.count("long_form") >= maximum_long_form:
                    return reject("long_form_quota_filled")
                maximum_standard = maximum_per_document - minimum_long_form
                if mode == "standard" and existing_modes.count("standard") >= maximum_standard:
                    return reject("standard_quota_filled")
            if dataset_kind == "multi_source_extra":
                cited_documents = _cited_document_ids(bundle, str(example["answer"]))
                if len(cited_documents) < 2:
                    return reject("insufficient_distinct_cited_documents")
            normalized_question = re.sub(r"\s+", " ", question.casefold())
            if normalized_question in seen_questions:
                return reject("duplicate_question")
            answer = example["answer"].strip()
            messages = self._training_messages(bundle, question, answer)
            accepted.append(
                {
                    "example_id": "sft_"
                    + stable_digest(bundle.bundle_id, question, answer),
                    "split": name,
                    "dataset_kind": dataset_kind,
                    "document_ids": list(bundle.document_ids),
                    "bundle_id": bundle.bundle_id,
                    "response_mode": example["response_mode"],
                    "answer_style": str(example.get("answer_style") or "prose"),
                    "messages": messages,
                    "token_count": self._chat_token_count(messages),
                }
            )
            seen_questions.add(normalized_question)
            accepted_questions_by_scope[scope].append(question)
            self._audit(
                "candidate_final_decision",
                partition=name,
                request_number=request_number,
                bundle_id=bundle.bundle_id,
                document_ids=list(bundle.document_ids),
                candidate_id=str(example.get("candidate_id") or ""),
                question=question,
                dataset_kind=dataset_kind,
                decision="accepted",
                reason="ok",
            )
            return True

        # Core examples are generated only from a single paper. Validated examples are checkpointed
        # per document so a retry requests only the missing allocation. Multi-source examples never
        # reduce the core allocation and are added in a separate phase below.
        for document_id in sorted(document_ids):
            bundle = single_bundles[document_id]
            state = (
                self._load_core_checkpoint(
                    checkpoint_digest,
                    name,
                    document_id,
                    bundle,
                    minimum_per_document,
                    maximum_per_document,
                )
                if checkpoint_digest
                else {
                    "accepted": [],
                    "accepted_questions": [],
                    "rejected": [],
                    "requests_completed": 0,
                    "consecutive_empty_requests": 0,
                    "adaptive_complete": False,
                }
            )
            core_states[document_id] = state
            for question in state["accepted_questions"]:
                normalized = re.sub(r"\s+", " ", str(question).strip().casefold())
                if not normalized or normalized in seen_questions:
                    raise RuntimeError(
                        "Dataset checkpoint contains an invalid or duplicate question: "
                        f"{document_id}"
                    )
                seen_questions.add(normalized)
                accepted_questions_by_scope[(document_id,)].append(str(question).strip())
            accepted.extend(state["accepted"])
            rejected.extend(state["rejected"])
            counts[document_id] = len(state["accepted"])

        maximum_requests = int(
            self.teacher.get("maximum_core_requests_per_document", 100)
        )
        maximum_zero_yield_before_minimum = int(
            self.teacher.get("maximum_zero_yield_requests_before_minimum", 2)
        )
        initial_batch_size = int(self.teacher.get("initial_candidate_batch_size", 8))
        refill_batch_size = int(self.teacher.get("refill_candidate_batch_size", 4))
        if initial_batch_size < 1 or refill_batch_size < 1:
            raise ValueError("Candidate batch sizes must be positive")
        attempts_this_run: dict[str, int] = defaultdict(int)

        def needs_generation(document_id: str) -> bool:
            state = core_states[document_id]
            if state["adaptive_complete"]:
                return False
            if counts[document_id] < minimum_per_document:
                return (
                    state["consecutive_empty_requests"]
                    < maximum_zero_yield_before_minimum
                )
            return False

        generation_concurrency = int(self.teacher.get("generation_concurrency", 1))

        def request_core_examples(document_id: str) -> list[dict[str, Any]]:
            bundle = single_bundles[document_id]
            state = core_states[document_id]
            long_form_count = sum(
                1
                for record in state["accepted"]
                if record["response_mode"] == "long_form"
            )
            remaining = maximum_per_document - counts[document_id]
            requested_examples = min(
                remaining,
                initial_batch_size
                if state["requests_completed"] == 0
                else refill_batch_size,
            )
            return self._request_examples_with_retries(
                bundle,
                accepted_questions=list(state["accepted_questions"]),
                allow_long_form=long_form_count < maximum_long_form,
                requested_examples=requested_examples,
                partition=name,
                request_number=state["requests_completed"] + 1,
            )

        def apply_core_examples(
            document_id: str, examples: list[dict[str, Any]]
        ) -> None:
            bundle = single_bundles[document_id]
            state = core_states[document_id]
            rejected_before = len(rejected)
            accepted_count_before = counts[document_id]
            for example in examples:
                if counts[document_id] >= maximum_per_document:
                    break
                accepted_before = len(accepted)
                if accept_example(
                    bundle,
                    example,
                    "core_single_source",
                    state["requests_completed"] + 1,
                ):
                    counts[document_id] += 1
                    state["accepted"].append(accepted[accepted_before])
                    state["accepted_questions"].append(
                        str(example["question"]).strip()
                    )
            accepted_this_request = counts[document_id] - accepted_count_before
            state["consecutive_empty_requests"] = (
                0
                if accepted_this_request
                else state["consecutive_empty_requests"] + 1
            )
            state["adaptive_complete"] = counts[document_id] >= minimum_per_document
            state["rejected"].extend(rejected[rejected_before:])
            state["requests_completed"] += 1
            request_rejection_counts = Counter(
                str(row.get("reason") or "unknown")
                for row in rejected[rejected_before:]
            )
            generation_seconds = getattr(self, "timing", {}).get(
                "generation_seconds", 0.0
            )
            verification_seconds = getattr(self, "timing", {}).get(
                "verification_seconds", 0.0
            )
            print(
                "[dataset] "
                f"split={name} document={document_id} "
                f"accepted={counts[document_id]}/{maximum_per_document} "
                f"request={state['requests_completed']} "
                f"generation_s={generation_seconds:.1f} "
                f"verification_s={verification_seconds:.1f}",
                flush=True,
            )
            self._audit(
                "document_request_completed",
                partition=name,
                request_number=state["requests_completed"],
                bundle_id=bundle.bundle_id,
                document_ids=list(bundle.document_ids),
                accepted_this_request=accepted_this_request,
                accepted_total=counts[document_id],
                minimum_required=minimum_per_document,
                maximum_allowed=maximum_per_document,
                rejection_counts=dict(sorted(request_rejection_counts.items())),
                cumulative_generation_seconds=round(generation_seconds, 3),
                cumulative_verification_seconds=round(verification_seconds, 3),
            )
            if checkpoint_digest:
                self._write_core_checkpoint(
                    checkpoint_digest,
                    name,
                    document_id,
                    bundle,
                    minimum_per_document,
                    maximum_per_document,
                    state,
                )

        while any(needs_generation(document_id) for document_id in document_ids):
            scheduled = [
                document_id
                for document_id in sorted(document_ids)
                if needs_generation(document_id)
                and attempts_this_run[document_id] < maximum_requests
            ]
            if not scheduled:
                break
            for document_id in scheduled:
                attempts_this_run[document_id] += 1

            generated: dict[str, list[dict[str, Any]]] = {}
            if generation_concurrency == 1:
                for document_id in scheduled:
                    generated[document_id] = request_core_examples(document_id)
            else:
                with ThreadPoolExecutor(max_workers=generation_concurrency) as executor:
                    futures = {
                        executor.submit(request_core_examples, document_id): document_id
                        for document_id in scheduled
                    }
                    for future in as_completed(futures):
                        generated[futures[future]] = future.result()

            # Applying candidates in stable document order preserves deterministic duplicate,
            # quota, checkpoint, and release ordering. Only teacher HTTP/decode overlaps.
            for document_id in scheduled:
                apply_core_examples(document_id, generated[document_id])

        underfilled = {
            document_id: counts[document_id]
            for document_id in document_ids
            if counts[document_id] < minimum_per_document
        }
        for document_id, accepted_count in sorted(underfilled.items()):
            self._audit(
                "underfilled_document_released",
                partition=name,
                document_ids=[document_id],
                bundle_id=single_bundles[document_id].bundle_id,
                accepted_total=accepted_count,
                preferred_minimum=minimum_per_document,
                release_floor=underfilled_release_floor,
            )
        below_release_floor = {
            document_id: count
            for document_id, count in underfilled.items()
            if count < underfilled_release_floor
        }
        if below_release_floor:
            preview = dict(list(sorted(below_release_floor.items()))[:20])
            raise RuntimeError(
                "Teacher generation did not produce the minimum "
                f"{underfilled_release_floor} core examples per document "
                f"(underfilled release floor): {preview}"
            )

        generated_multi: dict[str, list[dict[str, Any]]] = {}
        if generation_concurrency == 1:
            for bundle in multi_bundles:
                generated_multi[bundle.bundle_id] = self._request_examples_with_retries(
                    bundle,
                    partition=name,
                    request_number=1,
                )
        else:
            with ThreadPoolExecutor(max_workers=generation_concurrency) as executor:
                futures = {
                    executor.submit(
                        self._request_examples_with_retries,
                        bundle,
                        partition=name,
                        request_number=1,
                    ): bundle
                    for bundle in multi_bundles
                }
                for future in as_completed(futures):
                    bundle = futures[future]
                    generated_multi[bundle.bundle_id] = future.result()
        for bundle in multi_bundles:
            for example in generated_multi[bundle.bundle_id]:
                accept_example(bundle, example, "multi_source_extra", 1)
        return accepted, rejected

    def build(self, expected_documents: int | None = None) -> dict[str, Any]:
        # This must remain the first executable statement. Production-size work fails
        # before selection, checkpoints, teacher requests, or SFT artifact mutation.
        assert_dataset_generation_ready(self.settings, expected_documents)
        build_started = time.perf_counter()
        policy_fingerprint = self._checkpoint_policy_fingerprint()
        started_utc = datetime.now(UTC)
        run_id = (
            f"v15rc_{started_utc.strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}_"
            f"{policy_fingerprint[:10]}"
        )
        audit_path = (
            self.settings.path("paths.sft_dir") / "audit" / f"dataset_audit_{run_id}.jsonl"
        )
        self.audit_journal = DatasetAuditJournal(audit_path, run_id)
        self._audit(
            "run_started",
            generation_policy_version=_GENERATION_POLICY_VERSION,
            semantic_verifier_protocol_version=_SEMANTIC_VERIFIER_PROTOCOL_VERSION,
            policy_fingerprint=policy_fingerprint,
            expected_documents=expected_documents,
        )
        try:
            result = self._build_release(
                expected_documents,
                build_started=build_started,
                policy_fingerprint=policy_fingerprint,
            )
        except BaseException as exc:
            self._audit(
                "run_failed",
                error_type=type(exc).__name__,
                error_message=re.sub(r"\s+", " ", str(exc)).strip()[:1000],
                dataset_build_seconds=round(time.perf_counter() - build_started, 3),
            )
            raise
        self._audit(
            "run_completed",
            dataset_build_seconds=result["dataset_build_seconds"],
            train_examples=result["train_examples"],
            validation_examples=result["validation_examples"],
            rejected_examples=result["rejected_examples"],
        )
        return result

    def _build_release(
        self,
        expected_documents: int | None,
        *,
        build_started: float,
        policy_fingerprint: str,
    ) -> dict[str, Any]:
        # Invalidate any older complete release before selecting a new document set.
        # If this build is interrupted, training sees status=building and fails closed.
        begin_dataset_release(
            self.settings,
            generation_policy_version=_GENERATION_POLICY_VERSION,
            policy_fingerprint=policy_fingerprint,
            expected_documents=expected_documents,
        )
        documents, selection_manifest = select_training_documents(
            self.settings, expected_documents=expected_documents
        )
        selected_ids = {str(document["document_id"]) for document in documents}
        chunks = [
            row
            for row in read_jsonl(self.settings.path("paths.chunk_file"))
            if str(row["document_id"]) in selected_ids
        ]
        train_ids, validation_ids = split_document_ids(
            documents,
            validation_fraction=float(self.teacher["validation_fraction"]),
            seed=int(self.teacher["seed"]),
        )
        output_dir = self.settings.path("paths.sft_dir")
        checkpoint_digest = str(selection_manifest["selection_digest"])
        effective_multi_source_fraction, timed_smoke = (
            self._multi_source_fraction_for_run(expected_documents)
        )
        train, train_rejected = self._records_for_partition(
            "train",
            train_ids,
            chunks,
            checkpoint_digest=checkpoint_digest,
            multi_source_fraction=effective_multi_source_fraction,
        )
        validation, validation_rejected = self._records_for_partition(
            "validation",
            validation_ids,
            chunks,
            checkpoint_digest=checkpoint_digest,
            multi_source_fraction=effective_multi_source_fraction,
        )
        if {doc for row in train for doc in row["document_ids"]} & {
            doc for row in validation for doc in row["document_ids"]
        }:
            raise RuntimeError("Document leakage detected between train and validation")
        write_jsonl_atomic(output_dir / "train.jsonl", train)
        write_jsonl_atomic(output_dir / "validation.jsonl", validation)
        write_jsonl_atomic(output_dir / "rejected.jsonl", train_rejected + validation_rejected)
        all_examples = train + validation
        all_rejected = train_rejected + validation_rejected
        rejection_counts = Counter(str(row.get("reason") or "unknown") for row in all_rejected)
        result: dict[str, Any] = {
            "train_examples": len(train),
            "validation_examples": len(validation),
            "core_single_source_examples": sum(
                row["dataset_kind"] == "core_single_source" for row in all_examples
            ),
            "multi_source_extra_examples": sum(
                row["dataset_kind"] == "multi_source_extra" for row in all_examples
            ),
            "timed_smoke_multi_source_disabled": timed_smoke,
            "generation_seconds": round(self.timing["generation_seconds"], 3),
            "verification_seconds": round(self.timing["verification_seconds"], 3),
            "semantic_verification_enabled": bool(
                self.teacher.get("semantic_verification_enabled", True)
            ),
            "semantic_review_status": (
                "teacher_verified"
                if bool(self.teacher.get("semantic_verification_enabled", True))
                else "pending_manual"
            ),
            "dataset_build_seconds": round(time.perf_counter() - build_started, 3),
            "minimum_core_examples_per_document": int(
                self.teacher["minimum_core_examples_per_document"]
            ),
            "maximum_core_examples_per_document": int(
                self.teacher["maximum_core_examples_per_document"]
            ),
            "selected_documents": len(selected_ids),
            "training_selection_manifest": str(
                self.settings.path("paths.training_selection_manifest")
            ),
            "training_selection_digest": selection_manifest["selection_digest"],
            "rejected_examples": len(all_rejected),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "train_documents": len(train_ids),
            "validation_documents": len(validation_ids),
            "audit_journal": str(self.audit_journal.path) if self.audit_journal else None,
        }
        # Publish the manifest last. It binds the selection and every JSONL by hash,
        # byte size, and row count; a failure here deliberately leaves the release unusable.
        release = finalize_dataset_release(
            self.settings,
            generation_policy_version=_GENERATION_POLICY_VERSION,
            policy_fingerprint=policy_fingerprint,
            build_metadata=result,
        )
        result["dataset_release_manifest"] = str(
            self.settings.path("paths.sft_dir") / "dataset_manifest.json"
        )
        result["dataset_release_status"] = release["status"]
        result["semantic_audit"] = plan_semantic_audit(self.settings)
        return result
