from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .io import content_version, sha256_file, write_json_atomic


class TrainingMaterializationError(RuntimeError):
    """Raised when the frozen held-out plan cannot be materialized safely."""


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingMaterializationError(
            f"Missing held-out manifest: {path}. Run `giis_pipeline.py plan-heldout` first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingMaterializationError(f"Invalid held-out manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise TrainingMaterializationError("Held-out manifest must contain a documents list")
    return payload


def _validated_plan(
    manifest: dict[str, Any], source_dir: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = manifest["documents"]
    expected_digest = str(manifest.get("selection_digest") or "")
    if not expected_digest or content_version(rows) != expected_digest:
        raise TrainingMaterializationError("Held-out manifest selection digest is invalid")

    selected: list[dict[str, str]] = []
    heldout: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise TrainingMaterializationError("Held-out manifest contains a non-object row")
        document_id = str(raw.get("document_id") or "")
        source_file = str(raw.get("source_file") or "")
        expected_hash = str(raw.get("sha256") or "")
        partition = str(raw.get("partition") or "")
        if not document_id or document_id in seen_ids:
            raise TrainingMaterializationError(f"Missing or duplicate document_id: {document_id!r}")
        if (
            not source_file
            or Path(source_file).name != source_file
            or Path(source_file).suffix.casefold() != ".pdf"
            or source_file.casefold() in seen_names
        ):
            raise TrainingMaterializationError(
                f"Unsafe, non-PDF, or duplicate source filename: {source_file!r}"
            )
        if len(expected_hash) != 64:
            raise TrainingMaterializationError(f"Invalid SHA-256 for {source_file}")
        if partition not in {"sft_pool", "heldout_benchmark"}:
            raise TrainingMaterializationError(
                f"Unknown partition {partition!r} for {source_file}"
            )
        seen_ids.add(document_id)
        seen_names.add(source_file.casefold())
        row = {
            "document_id": document_id,
            "source_file": source_file,
            "sha256": expected_hash,
            "partition": partition,
        }
        (selected if partition == "sft_pool" else heldout).append(row)

    expected_corpus = int(manifest.get("corpus_documents", -1))
    expected_sft = int(manifest.get("sft_documents", -1))
    expected_heldout = int(manifest.get("heldout_documents", -1))
    if len(rows) != expected_corpus:
        raise TrainingMaterializationError(
            f"Manifest row count {len(rows)} != corpus_documents {expected_corpus}"
        )
    if len(selected) != expected_sft or len(heldout) != expected_heldout:
        raise TrainingMaterializationError(
            "Manifest partition counts do not match sft_documents/heldout_documents"
        )

    if not source_dir.is_dir():
        raise TrainingMaterializationError(f"Missing RAG source directory: {source_dir}")
    source_files = [
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.casefold() == ".pdf"
    ]
    actual_names = {path.name.casefold() for path in source_files}
    missing = sorted(seen_names - actual_names)
    if missing:
        raise TrainingMaterializationError(
            f"RAG source no longer contains the frozen selection; missing={missing[:5]}"
        )

    for row in selected + heldout:
        source = source_dir / row["source_file"]
        if source.is_symlink() or not source.is_file():
            raise TrainingMaterializationError(f"Unsafe or missing source PDF: {source}")
        actual_hash = sha256_file(source)
        if actual_hash != row["sha256"]:
            raise TrainingMaterializationError(
                f"Source hash mismatch for {row['source_file']}: "
                f"expected {row['sha256']}, found {actual_hash}"
            )
    return selected, heldout


def materialize_training(settings: Settings) -> dict[str, Any]:
    """Copy only the frozen SFT partition into Papers_Training with hash verification."""
    manifest_path = settings.path("paths.heldout_manifest")
    source_dir = settings.path("paths.pdf_dir")
    destination_dir = settings.path("paths.training_pdf_dir")
    report_path = settings.path("paths.training_materialization_report")
    manifest = _load_manifest(manifest_path)
    selected, heldout = _validated_plan(manifest, source_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {row["source_file"].casefold() for row in selected}
    unexpected_entries = sorted(
        path.name
        for path in destination_dir.iterdir()
        if path.name.casefold() not in expected_names
    )
    if unexpected_entries:
        raise TrainingMaterializationError(
            "Papers_Training contains entries outside the frozen SFT partition: "
            + ", ".join(unexpected_entries[:10])
        )

    existing_count = 0
    missing: list[dict[str, str]] = []
    for row in selected:
        destination = destination_dir / row["source_file"]
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise TrainingMaterializationError(f"Unsafe destination entry: {destination}")
            actual_hash = sha256_file(destination)
            if actual_hash != row["sha256"]:
                raise TrainingMaterializationError(
                    f"Existing training PDF hash mismatch for {row['source_file']}"
                )
            existing_count += 1
        else:
            missing.append(row)

    copied_count = 0
    for row in missing:
        source = source_dir / row["source_file"]
        destination = destination_dir / row["source_file"]
        temporary = destination_dir / f".{destination.name}.giis-materialize.tmp"
        if temporary.exists():
            raise TrainingMaterializationError(f"Stale materialization temporary file: {temporary}")
        try:
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != row["sha256"]:
                raise TrainingMaterializationError(
                    f"Copied training PDF failed hash verification: {row['source_file']}"
                )
            os.replace(temporary, destination)
            copied_count += 1
        finally:
            if temporary.exists():
                temporary.unlink()

    materialized_files = sorted(
        path for path in destination_dir.iterdir() if path.is_file()
    )
    if len(materialized_files) != len(selected):
        raise TrainingMaterializationError(
            f"Materialized {len(materialized_files)} files; expected {len(selected)}"
        )
    for row in selected:
        if sha256_file(destination_dir / row["source_file"]) != row["sha256"]:
            raise TrainingMaterializationError(
                f"Final training hash verification failed: {row['source_file']}"
            )
    heldout_names = {row["source_file"].casefold() for row in heldout}
    leaked = sorted(path.name for path in materialized_files if path.name.casefold() in heldout_names)
    if leaked:
        raise TrainingMaterializationError("Held-out leakage detected: " + ", ".join(leaked))

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_path": str(manifest_path),
        "selection_digest": manifest["selection_digest"],
        "source_directory": str(source_dir),
        "destination_directory": str(destination_dir),
        "corpus_documents": int(manifest["corpus_documents"]),
        "materialized_sft_documents": len(selected),
        "excluded_heldout_documents": len(heldout),
        "copied_documents": copied_count,
        "already_present_documents": existing_count,
        "heldout_leakage_count": 0,
        "verified": True,
    }
    write_json_atomic(report_path, report)
    return report
