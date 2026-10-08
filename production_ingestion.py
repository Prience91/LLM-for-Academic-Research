from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Layout:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.corpus = self.root / "Full_Corpus"
        self.parsed = self.root / "Parsed_Corpus"
        self.index = self.root / "Vector_Index"
        self.manifests = self.root / "manifests"
        self.inventory = self.manifests / "pdf_inventory.jsonl"
        self.duplicates = self.manifests / "duplicate_files.jsonl"
        self.journal = self.parsed / "state" / "document_journal.jsonl"
        self.document_shards = self.parsed / "shards" / "documents"
        self.chunk_shards = self.parsed / "shards" / "chunks"
        self.documents = self.parsed / "documents.jsonl"
        self.chunks = self.parsed / "chunks.jsonl"
        self.review_queue = self.parsed / "metadata" / "review_required.jsonl"
        self.deferred_backlog = self.parsed / "metadata" / "openalex_deferred_backlog.jsonl"
        self.report = self.parsed / "reports" / "ingestion_report.json"
        self.chroma = self.index / "chroma"
        self.lexical = self.index / "lexical.sqlite3"
        self.active_index = self.index / "active_index.json"
        self.binding = self.manifests / "full_rag_service_binding.json"

    def create(self) -> None:
        for path in (
            self.parsed,
            self.index,
            self.manifests,
            self.document_shards,
            self.chunk_shards,
            self.review_queue.parent,
            self.report.parent,
            self.chroma,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_binding(self, status: str, **extra: Any) -> None:
        current = {}
        if self.binding.exists():
            current = json.loads(self.binding.read_text(encoding="utf-8"))
        current.update(
            {
                "schema_version": "giis.full-rag.binding.v1",
                "updated_at": utc_now(),
                "status": status,
                "root": str(self.root),
                "source_corpus": str(self.corpus),
                "parsed_corpus": str(self.parsed),
                "document_registry": str(self.documents),
                "chunk_file": str(self.chunks),
                "metadata_review_queue": str(self.review_queue),
                "chroma_dir": str(self.chroma),
                "lexical_db": str(self.lexical),
                "active_index": str(self.active_index),
                "pipeline_config": str(self.root / "Pipeline" / "config" / "settings_full_rag.toml"),
            }
        )
        current.update(extra)
        dump_json(self.binding, current)


def inventory(layout: Layout) -> dict[str, Any]:
    previous = {row["relative_path"]: row for row in iter_jsonl(layout.inventory)}
    files = sorted(
        (path for path in layout.corpus.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: path.relative_to(layout.corpus).as_posix().casefold(),
    )
    rows: list[dict[str, Any]] = []
    hash_owner: dict[str, str] = {}
    duplicates: list[dict[str, Any]] = []
    for ordinal, path in enumerate(files, 1):
        relative = path.relative_to(layout.corpus).as_posix()
        stat = path.stat()
        old = previous.get(relative)
        if old and old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
            digest = old["sha256"]
        else:
            digest = sha256_file(path)
        duplicate_of = hash_owner.get(digest)
        if duplicate_of is None:
            hash_owner[digest] = relative
        else:
            duplicates.append({"relative_path": relative, "duplicate_of": duplicate_of, "sha256": digest})
        rows.append(
            {
                "ordinal": ordinal,
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
                "duplicate_of": duplicate_of,
            }
        )
        if ordinal % 250 == 0:
            dump_json(layout.manifests / "inventory_progress.json", {"updated_at": utc_now(), "scanned": ordinal, "total": len(files)})
    dump_jsonl(layout.inventory, rows)
    dump_jsonl(layout.duplicates, duplicates)
    result = {"pdf_count": len(rows), "unique_pdf_count": len(rows) - len(duplicates), "duplicate_file_count": len(duplicates)}
    dump_json(layout.manifests / "inventory_report.json", {"created_at": utc_now(), **result})
    layout.write_binding("inventory_complete", inventory=result)
    return result


def latest_journal(path: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        state[str(row["sha256"])] = row
    return state


def parse(layout: Layout, settings_path: Path, retry_quarantine: bool = False, limit: int | None = None, selection_manifest: Path | None = None) -> dict[str, Any]:
    from giis_prod.config import Settings
    from giis_prod.ingestion import CorpusBuilder

    settings = Settings.load(settings_path)
    builder = CorpusBuilder(settings)
    gold_rejection_path = settings.path("paths.metadata_overrides").parent / "gold_rejected_manual_human_review_only.jsonl"
    gold_rejected = {str(row["sha256"]) for row in iter_jsonl(gold_rejection_path)}
    state = latest_journal(layout.journal)
    selected = [row for row in iter_jsonl(layout.inventory) if not row.get("duplicate_of")]
    if selection_manifest is not None:
        wanted = {str(row["sha256"]) for row in iter_jsonl(selection_manifest)}
        if not wanted:
            raise ValueError(f"Selection manifest is empty: {selection_manifest}")
        available = {str(row["sha256"]) for row in selected}
        missing = wanted - available
        if missing:
            raise ValueError(f"Selection manifest contains {len(missing)} unknown document hashes")
        selected = [row for row in selected if row["sha256"] in wanted]
    quota_path = layout.parsed / "metadata" / "cache" / "openalex" / "rate_limit_state.json"
    quota_state = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path.exists() else {}
    blocked_until = (
        float(quota_state.get("blocked_until_epoch") or 0)
        if quota_state.get("reason") in {"daily_budget_floor", "provider_429"}
        else 0.0
    )
    # Existing deferred work receives each renewed allowance before fresh papers.
    deferred = sorted(
        (row for row in selected if state.get(row["sha256"], {}).get("status") == "metadata_deferred"),
        key=lambda row: state[row["sha256"]]["at"],
    )
    fresh = [row for row in selected if state.get(row["sha256"], {}).get("status") != "metadata_deferred"]
    pending = deque([*deferred, *fresh])
    stashed: list[dict[str, Any]] = []
    processed = 0
    while pending:
        if blocked_until and time.time() >= blocked_until and stashed:
            pending.extendleft(reversed(stashed))
            stashed.clear()
            blocked_until = 0.0
        row = pending.popleft()
        digest = row["sha256"]
        prior = state.get(digest)
        if digest in gold_rejected:
            if not prior or prior.get("status") != "gold_rejected":
                rejected_event = {"at": utc_now(), "sha256": digest, "source_file": row["relative_path"], "status": "gold_rejected", "chunk_count": 0, "recovery_policy": "manual_human_review_only"}
                append_jsonl(layout.journal, rejected_event)
                state[digest] = rejected_event
            continue
        if prior and prior.get("status") == "accepted":
            continue
        if prior and prior.get("status") == "metadata_deferred" and blocked_until > time.time():
            stashed.append(row)
            continue
        verified_override = bool((builder.overrides.get(digest) or {}).get("verified"))
        if prior and not retry_quarantine and not verified_override and prior.get("status") in {"metadata_review_required", "no_usable_chunks", "failed"}:
            continue
        source = Path(row["absolute_path"])
        started = time.monotonic()
        event = {"at": utc_now(), "sha256": digest, "source_file": row["relative_path"]}
        try:
            parsed = builder.parse(source)
            parsed.record.source_file = row["relative_path"]
            document = parsed.record.to_dict()
            chunks = [chunk.to_dict() for chunk in parsed.chunks]
            if document["metadata_status"] != "verified":
                status = "metadata_review_required"
            elif not chunks:
                status = "no_usable_chunks"
            else:
                status = "accepted"
            dump_json(layout.document_shards / f"{digest}.json", document)
            if chunks:
                dump_jsonl(layout.chunk_shards / f"{digest}.jsonl", chunks)
            else:
                (layout.chunk_shards / f"{digest}.jsonl").unlink(missing_ok=True)
            event.update(status=status, chunk_count=len(chunks), metadata_match_source=document.get("metadata_match_source"))
        except Exception as exc:
            if type(exc).__name__ == "MetadataQuotaDeferred":
                # Preserve the paper for the next allowance; scan Crossref-verifiable work.
                event.update(status="metadata_deferred", chunk_count=0, error=str(exc))
                quota_state = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path.exists() else {}
                blocked_until = float(quota_state.get("blocked_until_epoch") or 0)
                stashed.append(row)
            else:
                event.update(status="failed", chunk_count=0, error=f"{type(exc).__name__}: {exc}")
        event["elapsed_seconds"] = round(time.monotonic() - started, 3)
        append_jsonl(layout.journal, event)
        state[digest] = event
        processed += 1
        if processed % 10 == 0:
            counts: dict[str, int] = {}
            for value in state.values():
                counts[value["status"]] = counts.get(value["status"], 0) + 1
            dump_json(layout.parsed / "state" / "progress.json", {"updated_at": utc_now(), "total_unique": len(selected), "processed_this_run": processed, "status_counts": counts})
            layout.write_binding("parsing", parse_status=counts)
        if limit is not None and processed >= limit:
            break
    return materialize(layout, final=False)


def materialize(layout: Layout, final: bool = True) -> dict[str, Any]:
    state = latest_journal(layout.journal)
    retroactive_deferred = {row["sha256"] for row in iter_jsonl(layout.parsed / "metadata" / "retroactive_metadata_deferred_184.jsonl")}
    accepted = {digest for digest, row in state.items() if row.get("status") == "accepted" and digest not in retroactive_deferred}
    document_paths = sorted(
        layout.document_shards / f"{digest}.json"
        for digest, row in state.items()
        if row.get("status") != "metadata_deferred"
        and (layout.document_shards / f"{digest}.json").exists()
    )

    def visible_document(digest: str) -> dict[str, Any]:
        document = json.loads((layout.document_shards / f"{digest}.json").read_text(encoding="utf-8"))
        if digest in retroactive_deferred:
            document["metadata_status"] = "review_required"
            warnings = list(document.get("parse_warnings") or [])
            if "retroactive_deferred_adjudication" not in warnings:
                warnings.append("retroactive_deferred_adjudication")
            document["parse_warnings"] = warnings
        return document

    def documents() -> Iterator[dict[str, Any]]:
        for path in document_paths:
            yield visible_document(path.stem)

    document_count = dump_jsonl(layout.documents, documents())

    def chunks() -> Iterator[dict[str, Any]]:
        for digest in sorted(accepted):
            yield from iter_jsonl(layout.chunk_shards / f"{digest}.jsonl")

    chunk_count = dump_jsonl(layout.chunks, chunks())
    review_count = dump_jsonl(
        layout.review_queue,
        (
            visible_document(digest)
            for digest, row in sorted(state.items())
            if (row.get("status") == "metadata_review_required" or digest in retroactive_deferred) and (layout.document_shards / f"{digest}.json").exists()
        ),
    )
    deferred_count = dump_jsonl(
        layout.deferred_backlog,
        (
            {
                "sha256": digest,
                "source_file": row["source_file"],
                "deferred_at": row["at"],
                "reason": row.get("error"),
            }
            for digest, row in sorted(state.items())
            if row.get("status") == "metadata_deferred"
        ),
    )
    counts: dict[str, int] = {}
    for row in state.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "created_at": utc_now(),
        "complete": bool(final and deferred_count == 0),
        "metadata_deferred_count": deferred_count,
        "openalex_deferred_backlog": str(layout.deferred_backlog),
        "inventory_pdf_count": sum(1 for _ in iter_jsonl(layout.inventory)),
        "document_count": document_count,
        "accepted_document_count": len(accepted),
        "retroactive_deferred_count": len(retroactive_deferred),
        "chunk_count": chunk_count,
        "metadata_review_required_count": review_count,
        "status_counts": counts,
    }
    dump_json(layout.report, report)
    layout.write_binding("parsed" if final else "parsing", ingestion=report)
    return report


def file_digest(path: Path) -> str:
    return sha256_file(path)


def batched(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "document_id": str(chunk["document_id"]), "title": str(chunk["title"]),
        "authors_json": json.dumps(chunk.get("authors", []), ensure_ascii=False),
        "year": int(chunk["year"]) if chunk.get("year") else 0,
        "journal": str(chunk.get("journal") or ""), "publisher": str(chunk.get("publisher") or ""),
        "document_type": str(chunk.get("document_type") or "unknown"), "volume": str(chunk.get("volume") or ""),
        "issue": str(chunk.get("issue") or ""), "doi": str(chunk.get("doi") or ""),
        "theme": str(chunk.get("theme") or ""), "section": str(chunk.get("section") or ""),
        "page_start": int(chunk["page_start"]), "page_end": int(chunk["page_end"]),
        "ordinal": int(chunk["ordinal"]), "token_count": int(chunk.get("token_count") or 0),
    }


def index(layout: Layout, model_name: str, device: str, batch_size: int) -> dict[str, Any]:
    import chromadb
    from giis_prod.indexing import GraniteEmbeddingFunction

    if not layout.chunks.exists() or layout.chunks.stat().st_size == 0:
        raise RuntimeError("No materialized chunks are available for indexing")
    corpus_version = file_digest(layout.chunks)
    collection_name = "giis_full_rag_" + corpus_version[:16]
    embedding = GraniteEmbeddingFunction(model_name, device=device)
    client = chromadb.PersistentClient(path=str(layout.chroma))
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding,
        metadata={"hnsw:space": "cosine", "corpus_version": corpus_version},
    )
    existing = collection.count()
    consumed = 0
    for batch in batched(iter_jsonl(layout.chunks), batch_size):
        next_consumed = consumed + len(batch)
        if next_consumed <= existing:
            consumed = next_consumed
            continue
        if consumed < existing:
            batch = batch[existing - consumed :]
        collection.add(ids=[r["chunk_id"] for r in batch], documents=[r["text"] for r in batch], metadatas=[chroma_metadata(r) for r in batch])
        consumed = next_consumed
        if collection.count() % (batch_size * 10) == 0:
            dump_json(layout.index / "index_progress.json", {"updated_at": utc_now(), "collection": collection_name, "embedded_chunks": collection.count()})
    chunk_count = collection.count()
    if chunk_count != consumed:
        raise RuntimeError(f"Chroma count {chunk_count} does not match frozen chunk count {consumed}")

    lexical_versioned = layout.lexical.with_name(f"lexical.{corpus_version[:16]}.sqlite3")
    temporary = lexical_versioned.with_suffix(".sqlite3.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, payload_json TEXT NOT NULL);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title, section, text, tokenize='unicode61 remove_diacritics 0');
        """)
        for batch in batched(iter_jsonl(layout.chunks), 1000):
            connection.executemany("INSERT INTO chunks(chunk_id, document_id, payload_json) VALUES (?, ?, ?)", [(r["chunk_id"], r["document_id"], json.dumps(r, ensure_ascii=False, sort_keys=True)) for r in batch])
            connection.executemany("INSERT INTO chunks_fts(chunk_id, title, section, text) VALUES (?, ?, ?, ?)", [(r["chunk_id"], r.get("title", ""), r.get("section") or "", r["text"]) for r in batch])
            connection.commit()
    finally:
        connection.close()
    temporary.replace(lexical_versioned)
    if layout.lexical.exists() or layout.lexical.is_symlink():
        layout.lexical.unlink()
    layout.lexical.symlink_to(lexical_versioned.name)
    manifest = {
        "schema_version": "giis.full-rag.index.v1", "activated_at": utc_now(),
        "corpus_version": corpus_version, "collection_name": collection_name,
        "chroma_dir": str(layout.chroma), "lexical_db": str(lexical_versioned),
        "lexical_active_link": str(layout.lexical), "embedding_model": model_name,
        "embedding_device": device, "chunk_count": chunk_count,
        "document_registry": str(layout.documents), "chunk_file": str(layout.chunks),
        "metadata_deferred_count": json.loads(layout.report.read_text(encoding="utf-8")).get("metadata_deferred_count", 0),
        "retroactive_deferred_count": json.loads(layout.report.read_text(encoding="utf-8")).get("retroactive_deferred_count", 0),
        "openalex_deferred_backlog": str(layout.deferred_backlog),
    }
    dump_json(layout.active_index, manifest)
    layout.write_binding("active", index=manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Recursive, resumable GIIS Full-RAG ingestion")
    parser.add_argument("command", choices=("inventory", "parse", "materialize", "index", "all"))
    parser.add_argument("--root", type=Path, default=Path(os.getenv("GIIS_FULL_RAG_ROOT", str(Path(__file__).resolve().parent / "Full_RAG"))))
    parser.add_argument("--settings", type=Path, default=Path(__file__).resolve().parent / "config/settings_full_rag.toml")
    parser.add_argument("--retry-quarantine", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--embedding-model", default="ibm-granite/granite-embedding-30m-english")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    layout = Layout(args.root)
    layout.create()
    os.umask(0o002)
    if args.command in {"inventory", "all"}:
        print(json.dumps(inventory(layout), indent=2))
    if args.command in {"parse", "all"}:
        try:
            print(json.dumps(parse(layout, args.settings, args.retry_quarantine, args.limit, args.selection_manifest), indent=2))
        except Exception as exc:
            if type(exc).__name__ == "MetadataQuotaDeferred":
                print(str(exc), file=sys.stderr)
                return 75
            raise
    if args.command in {"materialize", "all"}:
        print(json.dumps(materialize(layout), indent=2))
    if args.command in {"index", "all"}:
        print(json.dumps(index(layout, args.embedding_model, args.embedding_device, args.batch_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


