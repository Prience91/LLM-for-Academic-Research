from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .io import content_version, read_jsonl, write_json_atomic


def _usable_documents(path: Path) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in read_jsonl(path)
            if str(row.get("metadata_status")) == "verified"
        ),
        key=lambda row: str(row["document_id"]),
    )


def plan_heldout_split(
    settings: Settings,
    expected_corpus_documents: int | None = None,
    heldout_documents: int | None = None,
) -> dict[str, Any]:
    """Create an immutable configuration-sized paper-level held-out plan."""
    expected = int(
        expected_corpus_documents
        if expected_corpus_documents is not None
        else settings.get("smoke_run_ii.expected_corpus_documents")
    )
    heldout_count = int(
        heldout_documents
        if heldout_documents is not None
        else settings.get("smoke_run_ii.heldout_documents")
    )
    configured_corpus = int(settings.get("smoke_run_ii.expected_corpus_documents"))
    expected_sft = (
        int(settings.get("smoke_run_ii.expected_sft_documents"))
        if expected == configured_corpus and heldout_documents is None
        else expected - heldout_count
    )
    if expected - heldout_count != expected_sft or expected_sft < 2:
        raise ValueError(
            f"Split invariant failed: {expected} - {heldout_count} != {expected_sft}"
        )
    documents = _usable_documents(settings.path("paths.document_registry"))
    if len(documents) < expected:
        raise ValueError(
            f"Held-out planning requires at least {expected} verified documents; found {len(documents)}"
        )

    rng = random.Random(int(settings.get("smoke_run_ii.seed")))
    shuffled = list(documents)
    rng.shuffle(shuffled)
    selected = sorted(shuffled[:expected], key=lambda row: str(row["document_id"]))
    selected_ids = {str(row["document_id"]) for row in selected}
    heldout_order = list(selected)
    rng.shuffle(heldout_order)
    heldout_ids = {str(row["document_id"]) for row in heldout_order[:heldout_count]}
    surplus_ids = sorted(str(row["document_id"]) for row in documents if str(row["document_id"]) not in selected_ids)
    rows = [
        {
            "document_id": str(row["document_id"]),
            "sha256": str(row["sha256"]),
            "source_file": str(row["source_file"]),
            "title": str(row["title"]),
            "partition": (
                "heldout_benchmark" if str(row["document_id"]) in heldout_ids else "sft_pool"
            ),
        }
        for row in selected
    ]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": "deterministic_random_paper_level_single_theme",
        "seed": int(settings.get("smoke_run_ii.seed")),
        "verified_pool_documents": len(documents),
        "corpus_documents": expected,
        "surplus_verified_documents": len(surplus_ids),
        "surplus_document_ids": surplus_ids,
        "sft_documents": expected_sft,
        "heldout_documents": heldout_count,
        "selection_digest": content_version(rows),
        "heldout_document_ids": sorted(heldout_ids),
        "sft_document_ids": sorted(
            str(row["document_id"])
            for row in selected
            if str(row["document_id"]) not in heldout_ids
        ),
        "documents": rows,
    }
    write_json_atomic(settings.path("paths.heldout_manifest"), manifest)
    return manifest
