from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .io import content_version, read_jsonl, write_json_atomic


def _collection_name(version: str) -> str:
    return "giis_corpus_" + re.sub(r"[^a-zA-Z0-9_-]", "_", version[:16])


def _lexical_version_path(base_path: Path, version: str) -> Path:
    return base_path.with_name(f"{base_path.stem}.{version[:16]}{base_path.suffix}")


def _chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "document_id": str(chunk["document_id"]),
        "title": str(chunk["title"]),
        "authors_json": json.dumps(chunk.get("authors", []), ensure_ascii=False),
        "year": int(chunk["year"]) if chunk.get("year") else 0,
        "journal": str(chunk.get("journal") or ""),
        "publisher": str(chunk.get("publisher") or ""),
        "document_type": str(chunk.get("document_type") or "unknown"),
        "volume": str(chunk.get("volume") or ""),
        "issue": str(chunk.get("issue") or ""),
        "doi": str(chunk.get("doi") or ""),
        "theme": str(chunk.get("theme") or ""),
        "section": str(chunk.get("section") or ""),
        "page_start": int(chunk["page_start"]),
        "page_end": int(chunk["page_end"]),
        "ordinal": int(chunk["ordinal"]),
        "token_count": int(chunk.get("token_count") or 0),
    }


class GraniteEmbeddingFunction:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, normalize_embeddings=True).tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Support Chroma 1.5+'s explicit query-embedding interface."""
        return self(input)

    @staticmethod
    def name() -> str:
        return "giis_granite_embedding"


def build_lexical_index(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                title,
                section,
                text,
                tokenize='unicode61 remove_diacritics 0'
            );
            """
        )
        connection.executemany(
            "INSERT INTO chunks(chunk_id, document_id, payload_json) VALUES (?, ?, ?)",
            [
                (
                    chunk["chunk_id"],
                    chunk["document_id"],
                    json.dumps(chunk, ensure_ascii=False, sort_keys=True),
                )
                for chunk in chunks
            ],
        )
        connection.executemany(
            "INSERT INTO chunks_fts(chunk_id, title, section, text) VALUES (?, ?, ?, ?)",
            [
                (
                    chunk["chunk_id"],
                    chunk.get("title", ""),
                    chunk.get("section") or "",
                    chunk["text"],
                )
                for chunk in chunks
            ],
        )
        connection.commit()
    finally:
        connection.close()


def build_indexes(settings: Settings, batch_size: int = 128) -> dict[str, Any]:
    chunks = list(read_jsonl(settings.path("paths.chunk_file")))
    if not chunks:
        raise ValueError("Cannot index an empty chunk file")
    version = content_version(chunks)
    collection_name = _collection_name(version)

    import chromadb

    embedding_model = str(settings.get("models.embedding"))
    embedding_function = GraniteEmbeddingFunction(embedding_model)
    chroma_client = chromadb.PersistentClient(path=str(settings.path("paths.chroma_dir")))
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "corpus_version": version},
    )
    existing_count = collection.count()
    if existing_count not in {0, len(chunks)}:
        raise RuntimeError(
            f"Immutable collection {collection_name} is partial: {existing_count}/{len(chunks)}"
        )
    if existing_count == 0:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[chunk["chunk_id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[_chroma_metadata(chunk) for chunk in batch],
            )
    if collection.count() != len(chunks):
        raise RuntimeError("Chroma verification failed after ingestion")

    lexical_path = _lexical_version_path(settings.path("paths.lexical_db"), version)
    build_lexical_index(lexical_path, chunks)
    manifest = {
        "activated_at": datetime.now(UTC).isoformat(),
        "corpus_version": version,
        "collection_name": collection_name,
        "chroma_dir": str(settings.path("paths.chroma_dir")),
        "lexical_db": str(lexical_path),
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
    }
    write_json_atomic(settings.path("paths.active_index"), manifest)
    return manifest
