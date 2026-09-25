from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .io import read_jsonl, sha256_file, stable_digest, write_json_atomic, write_jsonl_atomic
from .metadata import EnrichmentResult, MetadataResolver, title_relation
from .models import ChunkRecord, DocumentRecord, MetadataStatus

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_Z_STROKE_TRANSLATION = str.maketrans({"Ƶ": "Z", "ƶ": "z"})
_Z_STROKE_WARNING_PREFIX = "unicode_z_stroke_normalized="
_CORPUS_FILENAME_PREFIX = re.compile(r"^[0-9a-fA-F]{12}__")


def _count_z_stroke(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_z_stroke(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_count_z_stroke(item) for item in value)
    if value is None:
        return 0
    text = str(value)
    return text.count("Ƶ") + text.count("ƶ")


def _add_z_stroke_warning(warnings: list[str], count: int) -> None:
    if count <= 0:
        return
    previous = 0
    retained: list[str] = []
    for warning in warnings:
        if warning.startswith(_Z_STROKE_WARNING_PREFIX):
            previous += int(warning.removeprefix(_Z_STROKE_WARNING_PREFIX))
        else:
            retained.append(warning)
    retained.append(f"{_Z_STROKE_WARNING_PREFIX}{previous + count}")
    warnings[:] = retained


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(_Z_STROKE_TRANSLATION)
    cleaned = "".join(
        character
        for character in normalized
        if character in "\n\t" or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    )
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def _split_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = normalize_text(raw)
    candidates = re.split(r"\s*(?:;|\band\b|\|)\s*", text)
    return [candidate.strip(" ,") for candidate in candidates if candidate.strip(" ,")]


def _select_title(
    filename_stem: str, embedded_title: str | None, filename_is_curated: bool
) -> tuple[str, list[str]]:
    clean_stem, prefix_count = _CORPUS_FILENAME_PREFIX.subn("", filename_stem, count=1)
    filename_title = normalize_text(clean_stem)
    pdf_title = normalize_text(str(embedded_title or ""))
    prefix_warnings = ["corpus_filename_hash_prefix_removed"] if prefix_count else []
    if filename_is_curated:
        warnings = ["title_from_curated_filename", *prefix_warnings]
        if pdf_title and pdf_title.casefold() != filename_title.casefold():
            warnings.append("embedded_title_disagrees_with_curated_filename")
            if title_relation(filename_title, pdf_title) == "verified_truncation":
                warnings.append("verified_embedded_title_expands_truncated_filename")
                return pdf_title, warnings
        return filename_title, warnings
    if pdf_title:
        return pdf_title, []
    return filename_title, ["title_from_filename", *prefix_warnings]


def _native_text_is_adequate(
    texts: list[str], minimum_characters: int, minimum_chunks: int
) -> bool:
    substantive = []
    for text in texts:
        normalized = normalize_text(text)
        if normalized:
            substantive.append(normalized)
    return (
        len(substantive) >= minimum_chunks
        and sum(len(text) for text in substantive) >= minimum_characters
    )


def _bounded_token_windows(
    text: str, tokenizer: Any, maximum_tokens: int, overlap_tokens: int
) -> list[tuple[str, int]]:
    if maximum_tokens < 1:
        raise ValueError("maximum_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= maximum_tokens:
        raise ValueError("overlap_tokens must be between zero and maximum_tokens - 1")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    windows: list[tuple[str, int]] = []
    start = 0
    while start < len(token_ids):
        end = min(start + maximum_tokens, len(token_ids))
        segment = normalize_text(tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
        segment_count = len(tokenizer.encode(segment, add_special_tokens=False))
        while segment_count > maximum_tokens and end > start + 1:
            end -= 1
            segment = normalize_text(
                tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
            )
            segment_count = len(tokenizer.encode(segment, add_special_tokens=False))
        if segment:
            windows.append((segment, segment_count))
        if end >= len(token_ids):
            break
        next_start = end - overlap_tokens
        if next_start <= start:
            raise RuntimeError("Token-window splitter failed to make forward progress")
        start = next_start
    return windows


def _raw_pdf_year(metadata: Any) -> int | None:
    """Read malformed PDF date strings without invoking pypdf date conversion."""
    for key in ("/CreationDate", "/ModDate"):
        try:
            raw_value = metadata.get(key)
        except (AttributeError, KeyError, TypeError, ValueError):
            raw_value = None
        match = _YEAR.search(str(raw_value or ""))
        if match:
            return int(match.group())
    return None


def _metadata_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get("sha256") or row.get("source_file") or "").strip().lower()
        if not key:
            raise ValueError(f"Metadata override has no sha256 or source_file: {row}")
        overrides[key] = row
    return overrides


@dataclass(slots=True)
class ParsedDocument:
    record: DocumentRecord
    chunks: list[ChunkRecord]


def _enforce_ingestion_gate(
    accepted_documents: int,
    quarantined_documents: int,
    expected_documents: int,
    maximum_quarantine_fraction: float,
) -> float:
    if expected_documents < 1:
        raise ValueError("Expected corpus document count must be positive")
    attempted = accepted_documents + quarantined_documents
    quarantine_fraction = quarantined_documents / attempted if attempted else 1.0
    if quarantine_fraction > maximum_quarantine_fraction:
        raise RuntimeError(
            f"Corpus quarantine fraction {quarantine_fraction:.2%} exceeds "
            f"{maximum_quarantine_fraction:.2%}; review ingestion_report.json before indexing."
        )
    if accepted_documents < expected_documents:
        raise RuntimeError(
            f"Corpus has {accepted_documents} accepted unique documents; expected at least "
            f"{expected_documents}. Add verified replacement PDFs before indexing."
        )
    return quarantine_fraction


class CorpusBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ingestion = settings.section("ingestion")
        self.overrides = _metadata_overrides(settings.path("paths.metadata_overrides"))
        self.metadata_resolver = MetadataResolver(settings)
        self._native_converter = None
        self._ocr_converter = None
        self._chunker = None
        self._tokenizer = None

    def _initialize_parser(self) -> None:
        if self._native_converter is not None:
            return
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        embedding_model = self.settings.get("models.embedding")
        max_tokens = int(self.ingestion["max_chunk_tokens"])
        self._tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        docling_tokenizer = HuggingFaceTokenizer(
            tokenizer=self._tokenizer,
            max_tokens=max_tokens,
        )
        self._chunker = HybridChunker(tokenizer=docling_tokenizer, merge_peers=True)
        native_options = PdfPipelineOptions(do_ocr=False)
        self._native_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=native_options)
            }
        )
        if bool(self.ingestion.get("ocr_fallback_enabled", True)):
            ocr_options = PdfPipelineOptions(do_ocr=True)
            self._ocr_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=ocr_options)
                }
            )

    def _read_pdf_metadata(
        self, path: Path
    ) -> tuple[dict[str, Any], int, list[str], str]:
        import pypdf

        warnings: list[str] = []
        reader = pypdf.PdfReader(path)
        metadata = reader.metadata or {}
        raw_title = getattr(metadata, "title", None)
        raw_author = getattr(metadata, "author", None)
        metadata_year = _raw_pdf_year(metadata)
        title, title_warnings = _select_title(
            path.stem,
            raw_title,
            bool(self.settings.get("metadata.filename_is_curated_title")),
        )
        authors = _split_authors(raw_author)
        _add_z_stroke_warning(warnings, _count_z_stroke([raw_title, raw_author]))
        warnings.extend(title_warnings)
        if not authors:
            warnings.append("authors_missing")
        if metadata_year is None:
            warnings.append("year_missing")
        title_page_count = int(self.settings.get("metadata.title_page_count"))
        first_pages: list[str] = []
        for page in reader.pages[:title_page_count]:
            try:
                first_pages.append(page.extract_text() or "")
            except Exception as exc:
                warnings.append(f"title_page_text_error:{type(exc).__name__}")
        return {
            "title": title,
            "authors": authors,
            "year": metadata_year,
            "journal": None,
            "doi": None,
            "theme": None,
            "document_type": "unknown",
            "publisher": None,
            "volume": None,
            "issue": None,
        }, len(reader.pages), warnings, normalize_text("\n".join(first_pages))

    def _resolve_metadata(
        self,
        path: Path,
        file_hash: str,
        base: dict[str, Any],
        page_count: int,
        warnings: list[str],
        first_pages_text: str,
    ) -> DocumentRecord:
        override = self.overrides.get(file_hash) or self.overrides.get(path.name.lower()) or {}
        _add_z_stroke_warning(warnings, _count_z_stroke(override))
        try:
            enrichment = self.metadata_resolver.enrich(base, first_pages_text, override)
        except RuntimeError as exc:
            enrichment = EnrichmentResult(
                values=dict(base),
                verified=False,
                confidence=0.0,
                match_source="lookup_failure",
                provenance={key: "pdf_native_or_title_page" for key, value in base.items() if value},
                warnings=[f"metadata_lookup_failure:{type(exc).__name__}"],
            )
        warnings.extend(enrichment.warnings)
        merged = enrichment.values
        authors = merged.get("authors", [])
        if isinstance(authors, str):
            authors = _split_authors(authors)
        doi = str(merged.get("doi") or "").strip() or None
        if doi:
            match = _DOI.search(doi)
            doi = match.group().rstrip(".,") if match else doi

        verified = enrichment.verified
        complete = bool(merged.get("title") and authors and merged.get("year"))
        if verified and not complete:
            raise ValueError(f"Verified metadata is incomplete for {path.name}")

        status = MetadataStatus.VERIFIED if verified else MetadataStatus.REVIEW_REQUIRED
        confidence = enrichment.confidence
        if bool(self.ingestion["require_verified_metadata"]) and not verified:
            warnings.append("verified_metadata_required")

        return DocumentRecord(
            document_id=f"doc_{file_hash[:20]}",
            source_file=path.name,
            sha256=file_hash,
            title=normalize_text(str(merged.get("title") or path.stem)),
            authors=[normalize_text(str(author)) for author in authors],
            year=int(merged["year"]) if merged.get("year") else None,
            journal=normalize_text(str(merged["journal"])) if merged.get("journal") else None,
            doi=doi,
            theme=normalize_text(str(merged["theme"])) if merged.get("theme") else None,
            page_count=page_count,
            metadata_status=status,
            metadata_confidence=confidence,
            parser="docling-hybrid",
            document_type=normalize_text(str(merged.get("document_type") or "unknown")),
            publisher=(
                normalize_text(str(merged["publisher"])) if merged.get("publisher") else None
            ),
            volume=normalize_text(str(merged["volume"])) if merged.get("volume") else None,
            issue=normalize_text(str(merged["issue"])) if merged.get("issue") else None,
            metadata_match_source=enrichment.match_source,
            metadata_provenance=enrichment.provenance,
            metadata_verified_at=datetime.now(UTC).isoformat() if verified else None,
            parse_warnings=warnings,
        )

    def parse(self, path: Path) -> ParsedDocument:
        self._initialize_parser()
        file_hash = sha256_file(path)
        base, page_count, warnings, first_pages_text = self._read_pdf_metadata(path)
        record = self._resolve_metadata(
            path, file_hash, base, page_count, warnings, first_pages_text
        )
        if record.metadata_status in {MetadataStatus.REJECTED, MetadataStatus.REVIEW_REQUIRED}:
            return ParsedDocument(record=record, chunks=[])

        native_error: Exception | None = None
        try:
            conversion = self._native_converter.convert(path)
            raw_chunks = list(self._chunker.chunk(dl_doc=conversion.document))
        except Exception as exc:
            native_error = exc
            raw_chunks = []

        native_texts = [str(raw_chunk.text) for raw_chunk in raw_chunks]
        native_is_adequate = _native_text_is_adequate(
            native_texts,
            int(self.ingestion.get("native_text_minimum_characters", 1000)),
            int(self.ingestion.get("native_text_minimum_chunks", 2)),
        )
        if native_error is not None:
            record.parse_warnings.append(
                f"native_text_extraction_error:{type(native_error).__name__}"
            )
        if not native_is_adequate and self._ocr_converter is not None:
            record.parse_warnings.append("ocr_fallback_used")
            conversion = self._ocr_converter.convert(path)
            raw_chunks = list(self._chunker.chunk(dl_doc=conversion.document))
            record.parser = "docling-hybrid-ocr-fallback"
        elif native_error is not None:
            raise native_error
        else:
            record.parser = "docling-hybrid-native-text"

        chunks: list[ChunkRecord] = []
        chunk_z_stroke_count = 0
        oversized_chunks_split = 0
        output_ordinal = 0
        for raw_ordinal, raw_chunk in enumerate(raw_chunks):
            raw_text = str(raw_chunk.text)
            chunk_z_stroke_count += _count_z_stroke(raw_text)
            text = normalize_text(raw_text)
            if len(text) < int(self.ingestion["min_chunk_characters"]):
                continue
            pages = sorted(
                {
                    int(provenance.page_no)
                    for item in getattr(raw_chunk.meta, "doc_items", [])
                    for provenance in getattr(item, "prov", [])
                    if getattr(provenance, "page_no", None) is not None
                }
            )
            headings = getattr(raw_chunk.meta, "headings", None) or []
            chunk_z_stroke_count += _count_z_stroke(headings)
            section = normalize_text(" > ".join(map(str, headings))) if headings else None
            page_start = pages[0] if pages else 1
            page_end = pages[-1] if pages else page_start
            windows = _bounded_token_windows(
                text,
                self._tokenizer,
                int(self.ingestion["max_chunk_tokens"]),
                int(self.ingestion.get("chunk_overlap_tokens", 64)),
            )
            if len(windows) > 1:
                oversized_chunks_split += 1
            for window_index, (window_text, token_count) in enumerate(windows):
                chunk_id = "chk_" + stable_digest(
                    record.document_id,
                    str(page_start),
                    str(page_end),
                    section or "",
                    str(raw_ordinal),
                    str(window_index),
                    window_text,
                )
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        document_id=record.document_id,
                        text=window_text,
                        section=section,
                        page_start=page_start,
                        page_end=page_end,
                        ordinal=output_ordinal,
                        token_count=token_count,
                        title=record.title,
                        authors=record.authors,
                        year=record.year,
                        journal=record.journal,
                        doi=record.doi,
                        theme=record.theme,
                        document_type=record.document_type,
                        publisher=record.publisher,
                        volume=record.volume,
                        issue=record.issue,
                    )
                )
                output_ordinal += 1
        if not chunks:
            record.parse_warnings.append("no_usable_chunks")
        if oversized_chunks_split:
            record.parse_warnings.append(
                f"oversized_chunks_split={oversized_chunks_split}"
            )
        _add_z_stroke_warning(record.parse_warnings, chunk_z_stroke_count)
        return ParsedDocument(record=record, chunks=chunks)

    def build(self, expected_documents: int | None = None) -> dict[str, Any]:
        pdf_dir = self.settings.path("paths.pdf_dir")
        pdf_files = (
            sorted(
                (
                    path
                    for path in pdf_dir.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".pdf"
                ),
                key=lambda path: path.name.casefold(),
            )
            if pdf_dir.exists()
            else []
        )
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {pdf_dir}. Place PDFs directly in the top-level "
                "`Papers_RAG/` directory; nested directories are not scanned."
            )

        documents: list[DocumentRecord] = []
        chunks: list[ChunkRecord] = []
        failures: list[dict[str, str]] = []
        seen_hashes: dict[str, str] = {}
        duplicate_files: list[dict[str, str]] = []
        for path in pdf_files:
            try:
                file_hash = sha256_file(path)
                if file_hash in seen_hashes:
                    duplicate_files.append(
                        {
                            "source_file": path.name,
                            "duplicate_of": seen_hashes[file_hash],
                            "sha256": file_hash,
                        }
                    )
                    continue
                seen_hashes[file_hash] = path.name
                parsed = self.parse(path)
                documents.append(parsed.record)
                chunks.extend(parsed.chunks)
            except Exception as exc:
                failures.append({"source_file": path.name, "error": f"{type(exc).__name__}: {exc}"})

        rejected = sum(document.metadata_status is MetadataStatus.REJECTED for document in documents)
        review_required = sum(
            document.metadata_status is MetadataStatus.REVIEW_REQUIRED for document in documents
        )
        chunked_document_ids = {chunk.document_id for chunk in chunks}
        accepted_documents = [
            document
            for document in documents
            if document.metadata_status is MetadataStatus.VERIFIED
            and document.document_id in chunked_document_ids
        ]
        quarantined_records = [
            {
                "document_id": document.document_id,
                "source_file": document.source_file,
                "reason": (
                    document.metadata_status.value
                    if document.metadata_status is not MetadataStatus.VERIFIED
                    else "no_usable_chunks"
                ),
            }
            for document in documents
            if document not in accepted_documents
        ]
        quarantined_count = len(quarantined_records) + len(failures)
        unique_pdf_count = len(documents) + len(failures)
        quarantine_fraction = (
            quarantined_count / unique_pdf_count if unique_pdf_count else 1.0
        )
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "pdf_count": len(pdf_files),
            "unique_pdf_count": unique_pdf_count,
            "duplicate_file_count": len(duplicate_files),
            "duplicate_files": duplicate_files,
            "document_count": len(documents),
            "accepted_document_count": len(accepted_documents),
            "quarantined_document_count": quarantined_count,
            "quarantined_documents": quarantined_records,
            "chunk_count": len(chunks),
            "verified_document_count": sum(
                document.metadata_status is MetadataStatus.VERIFIED for document in documents
            ),
            "provisional_document_count": sum(
                document.metadata_status is MetadataStatus.PROVISIONAL for document in documents
            ),
            "rejected_document_count": rejected,
            "review_required_document_count": review_required,
            "failure_count": len(failures),
            "quarantine_fraction": quarantine_fraction,
            "rejected_fraction": quarantine_fraction,
            "z_stroke_normalized_documents": [
                {
                    "document_id": document.document_id,
                    "source_file": document.source_file,
                    "replacement_count": sum(
                        int(warning.removeprefix(_Z_STROKE_WARNING_PREFIX))
                        for warning in document.parse_warnings
                        if warning.startswith(_Z_STROKE_WARNING_PREFIX)
                    ),
                }
                for document in documents
                if any(
                    warning.startswith(_Z_STROKE_WARNING_PREFIX)
                    for warning in document.parse_warnings
                )
            ],
            "failures": failures,
        }

        write_jsonl_atomic(
            self.settings.path("paths.document_registry"),
            (document.to_dict() for document in documents),
        )
        write_jsonl_atomic(
            self.settings.path("paths.chunk_file"),
            (chunk.to_dict() for chunk in chunks),
        )
        write_jsonl_atomic(
            self.settings.path("paths.metadata_review_queue"),
            (
                document.to_dict()
                for document in documents
                if document.metadata_status is MetadataStatus.REVIEW_REQUIRED
            ),
        )
        write_json_atomic(self.settings.path("paths.ingestion_report"), report)

        target = (
            int(expected_documents)
            if expected_documents is not None
            else int(self.settings.get("smoke_run_ii.expected_corpus_documents"))
        )
        _enforce_ingestion_gate(
            len(accepted_documents),
            quarantined_count,
            target,
            float(self.ingestion["maximum_rejected_document_fraction"]),
        )
        return report
