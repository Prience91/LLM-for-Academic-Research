from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .io import content_version, read_jsonl, write_json_atomic
from .metadata import normalize_doi, title_relation, title_similarity


_CORPUS_FILENAME_PREFIX = __import__("re").compile(r"^[0-9a-fA-F]{12}__")


class CorpusAuditError(RuntimeError):
    """Raised after a deterministic corpus audit report records blocking defects."""


def _source_title_variants(source_file: str) -> list[str]:
    stem = Path(source_file).stem.strip()
    stem = _CORPUS_FILENAME_PREFIX.sub("", stem, count=1)
    parts = [part.strip() for part in stem.split(";") if part.strip()]
    variants = [stem]
    variants.extend(" ".join(parts[index:]) for index in range(1, len(parts)))
    return list(dict.fromkeys(variants))


def _source_title_similarity(source_file: str, resolved_title: str) -> float:
    return max(
        (title_similarity(candidate, resolved_title) for candidate in _source_title_variants(source_file)),
        default=0.0,
    )


def _issue(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **context}


class CorpusAuditor:
    """Bounded, deterministic audit of corpus metadata, chunks, splits, and indexes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.minimum_title_similarity = float(
            settings.get("corpus_audit.minimum_source_title_similarity")
        )
        self.minimum_fragment_tokens = int(
            settings.get("metadata.minimum_title_fragment_tokens")
        )
        self.minimum_fragment_characters = int(
            settings.get("metadata.minimum_title_fragment_characters")
        )
        self.maximum_chunk_tokens = int(settings.get("ingestion.max_chunk_tokens"))

    def _audit_documents_and_chunks(
        self, documents: list[dict[str, Any]], chunks: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        document_ids = [str(row.get("document_id") or "") for row in documents]
        hashes = [str(row.get("sha256") or "") for row in documents]
        sources = [str(row.get("source_file") or "") for row in documents]
        chunk_ids = [str(row.get("chunk_id") or "") for row in chunks]

        for label, values in (
            ("document_id", document_ids),
            ("sha256", hashes),
            ("source_file", sources),
        ):
            duplicates = sorted(value for value, count in Counter(values).items() if value and count > 1)
            if duplicates:
                errors.append(
                    _issue(
                        f"duplicate_{label}",
                        f"Duplicate {label} values found",
                        samples=duplicates[:20],
                    )
                )
        duplicate_chunks = sorted(
            value for value, count in Counter(chunk_ids).items() if value and count > 1
        )
        if duplicate_chunks:
            errors.append(
                _issue(
                    "duplicate_chunk_id",
                    "Duplicate chunk IDs found",
                    samples=duplicate_chunks[:20],
                )
            )

        documents_by_id = {str(row["document_id"]): row for row in documents}
        chunk_counts = Counter(str(row.get("document_id") or "") for row in chunks)
        required_provenance = {"title", "authors", "year"}
        for document in documents:
            document_id = str(document.get("document_id") or "")
            source_file = str(document.get("source_file") or "")
            title = str(document.get("title") or "")
            missing = [
                key for key in ("title", "authors", "year") if not document.get(key)
            ]
            if str(document.get("metadata_status")) != "verified":
                errors.append(
                    _issue(
                        "metadata_not_verified",
                        "Only verified documents may enter an active corpus",
                        document_id=document_id,
                        source_file=source_file,
                    )
                )
            if missing:
                errors.append(
                    _issue(
                        "required_metadata_missing",
                        "Required citation metadata is missing",
                        document_id=document_id,
                        source_file=source_file,
                        fields=missing,
                    )
                )
            provenance = document.get("metadata_provenance") or {}
            missing_provenance = sorted(required_provenance - set(provenance))
            if missing_provenance:
                errors.append(
                    _issue(
                        "metadata_provenance_missing",
                        "Required metadata lacks provenance",
                        document_id=document_id,
                        source_file=source_file,
                        fields=missing_provenance,
                    )
                )
            variants = _source_title_variants(source_file)
            score = _source_title_similarity(source_file, title)
            relations = [
                title_relation(
                    variant,
                    title,
                    self.minimum_title_similarity,
                    self.minimum_fragment_tokens,
                    self.minimum_fragment_characters,
                )
                for variant in variants
            ]
            if not any(relation != "conflict" for relation in relations):
                warnings.append(
                    _issue(
                        "filename_differs_from_canonical_title",
                        "Tentative PDF filename differs from the verified canonical title; canonical metadata remains authoritative",
                        document_id=document_id,
                        source_file=source_file,
                        resolved_title=title,
                        similarity=round(score, 4),
                    )
                )
            elif "verified_truncation" in relations:
                warnings.append(
                    _issue(
                        "verified_filename_title_truncation",
                        "Filename is a verified distinctive fragment of the canonical title",
                        document_id=document_id,
                        source_file=source_file,
                        resolved_title=title,
                    )
                )
            doi = document.get("doi")
            if doi and normalize_doi(doi) != str(doi).casefold():
                errors.append(
                    _issue(
                        "malformed_doi",
                        "Stored DOI is not a normalized DOI value",
                        document_id=document_id,
                        source_file=source_file,
                        doi=doi,
                    )
                )
            if chunk_counts[document_id] == 0:
                errors.append(
                    _issue(
                        "document_without_chunks",
                        "Accepted document has no usable chunks",
                        document_id=document_id,
                        source_file=source_file,
                    )
                )

        comparable_fields = (
            "title",
            "authors",
            "year",
            "journal",
            "doi",
            "document_type",
            "publisher",
            "volume",
            "issue",
        )
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            document_id = str(chunk.get("document_id") or "")
            document = documents_by_id.get(document_id)
            if document is None:
                errors.append(
                    _issue(
                        "orphan_chunk",
                        "Chunk references an unknown document",
                        chunk_id=chunk_id,
                        document_id=document_id,
                    )
                )
                continue
            mismatched = [
                field for field in comparable_fields if chunk.get(field) != document.get(field)
            ]
            if mismatched:
                errors.append(
                    _issue(
                        "chunk_metadata_mismatch",
                        "Chunk citation metadata differs from its document record",
                        chunk_id=chunk_id,
                        document_id=document_id,
                        fields=mismatched,
                    )
                )
            token_count = int(chunk.get("token_count") or 0)
            if token_count < 1 or token_count > self.maximum_chunk_tokens:
                errors.append(
                    _issue(
                        "chunk_token_count_invalid",
                        "Chunk token count is outside the configured boundary",
                        chunk_id=chunk_id,
                        token_count=token_count,
                        maximum=self.maximum_chunk_tokens,
                    )
                )
            page_start = int(chunk.get("page_start") or 0)
            page_end = int(chunk.get("page_end") or 0)
            if page_start < 1 or page_end < page_start or page_end > int(document["page_count"]):
                errors.append(
                    _issue(
                        "chunk_page_range_invalid",
                        "Chunk page range is outside its source document",
                        chunk_id=chunk_id,
                        page_start=page_start,
                        page_end=page_end,
                        page_count=document["page_count"],
                    )
                )

        heldout_path = self.settings.path("paths.heldout_manifest")
        if heldout_path.exists():
            manifest = json.loads(heldout_path.read_text(encoding="utf-8"))
            stale_ids: list[str] = []
            for row in manifest.get("documents") or []:
                document = documents_by_id.get(str(row.get("document_id") or ""))
                if document is None or any(
                    row.get(field) != document.get(field)
                    for field in ("sha256", "source_file", "title")
                ):
                    stale_ids.append(str(row.get("document_id") or ""))
            if len(manifest.get("documents") or []) != len(documents) or stale_ids:
                warnings.append(
                    _issue(
                        "heldout_manifest_stale",
                        "Held-out split must be regenerated after corpus changes",
                        samples=stale_ids[:20],
                    )
                )
        return errors, warnings

    def _audit_indexes(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        active_path = self.settings.path("paths.active_index")
        if not active_path.exists():
            return [_issue("active_index_missing", "Active index manifest does not exist")]
        manifest = json.loads(active_path.read_text(encoding="utf-8"))
        version = content_version(chunks)
        if manifest.get("corpus_version") != version:
            errors.append(
                _issue(
                    "active_index_version_mismatch",
                    "Active index was built from different chunk content",
                    expected=version,
                    actual=manifest.get("corpus_version"),
                )
            )
        if int(manifest.get("chunk_count") or -1) != len(chunks):
            errors.append(
                _issue(
                    "active_index_manifest_count_mismatch",
                    "Active index manifest chunk count is incorrect",
                    expected=len(chunks),
                    actual=manifest.get("chunk_count"),
                )
            )

        lexical_path = Path(str(manifest.get("lexical_db") or ""))
        if not lexical_path.is_file():
            errors.append(_issue("lexical_index_missing", "Lexical index file does not exist"))
        else:
            try:
                connection = sqlite3.connect(lexical_path)
                try:
                    lexical_count = int(
                        connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                    )
                    fts_count = int(
                        connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                    )
                finally:
                    connection.close()
            except sqlite3.Error as exc:
                errors.append(
                    _issue(
                        "lexical_index_unreadable",
                        "Lexical index could not be verified",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                if lexical_count != len(chunks) or fts_count != len(chunks):
                    errors.append(
                        _issue(
                            "lexical_index_count_mismatch",
                            "Lexical and FTS row counts must equal the chunk registry",
                            expected=len(chunks),
                            chunks=lexical_count,
                            fts=fts_count,
                        )
                    )

        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(manifest.get("chroma_dir") or ""))
            collection = client.get_collection(str(manifest.get("collection_name") or ""))
            chroma_count = int(collection.count())
            if chroma_count != len(chunks):
                errors.append(
                    _issue(
                        "chroma_count_mismatch",
                        "Chroma vector count must equal the chunk registry",
                        expected=len(chunks),
                        actual=chroma_count,
                    )
                )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            errors.append(
                _issue(
                    "chroma_index_unreadable",
                    "Active Chroma collection could not be verified",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        return errors

    def run(self, check_indexes: bool = True) -> dict[str, Any]:
        documents = list(read_jsonl(self.settings.path("paths.document_registry")))
        chunks = list(read_jsonl(self.settings.path("paths.chunk_file")))
        errors, warnings = self._audit_documents_and_chunks(documents, chunks)
        if check_indexes:
            errors.extend(self._audit_indexes(chunks))
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "check_indexes": check_indexes,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
        }
        write_json_atomic(self.settings.path("paths.corpus_audit_report"), report)
        if errors:
            codes = sorted({str(issue["code"]) for issue in errors})
            raise CorpusAuditError(
                f"Corpus audit failed with {len(errors)} errors: {', '.join(codes)}; "
                f"review {self.settings.path('paths.corpus_audit_report')}"
            )
        return report
