from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .io import sha256_file, write_json_atomic

DATASET_RELEASE_SCHEMA_VERSION = 2
SUPPORTED_DATASET_RELEASE_SCHEMA_VERSIONS = {1, 2}
MINIMUM_GENERATION_POLICY_VERSION = 15
DATASET_RELEASE_FILENAME = "dataset_manifest.json"
_RELEASE_FILES = ("train.jsonl", "validation.jsonl", "rejected.jsonl")


class DatasetReleaseError(RuntimeError):
    """The SFT artifacts do not form a complete, immutable dataset release."""


def dataset_release_path(settings: Settings) -> Path:
    return settings.path("paths.sft_dir") / DATASET_RELEASE_FILENAME


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetReleaseError(f"Missing {label}: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetReleaseError(f"Unreadable {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetReleaseError(f"Invalid {label}: expected a JSON object at {path}")
    return payload


def _required_text(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetReleaseError(f"Invalid {label}: {key!r} must be non-empty text")
    return value.strip()


def _required_nonnegative_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetReleaseError(
            f"Invalid {label}: {key!r} must be a non-negative integer"
        )
    return value


def _sha256(path: Path, label: str) -> str:
    try:
        return sha256_file(path)
    except (OSError, ValueError) as exc:
        raise DatasetReleaseError(f"Cannot hash {label}: {path}: {exc}") from exc


def _jsonl_row_count(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetReleaseError(
                        f"Invalid JSONL in released artifact {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise DatasetReleaseError(
                        f"Invalid JSONL in released artifact {path}:{line_number}: "
                        "each row must be a JSON object"
                    )
                count += 1
    except FileNotFoundError as exc:
        raise DatasetReleaseError(f"Missing released dataset artifact: {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise DatasetReleaseError(f"Unreadable released dataset artifact: {path}: {exc}") from exc
    return count


def _file_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetReleaseError(f"Missing released dataset artifact: {path}")
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise DatasetReleaseError(
            f"Cannot inspect released dataset artifact: {path}: {exc}"
        ) from exc
    return {
        "name": path.name,
        "size_bytes": size_bytes,
        "sha256": _sha256(path, "released dataset artifact"),
        "row_count": _jsonl_row_count(path),
    }


def begin_dataset_release(
    settings: Settings,
    *,
    generation_policy_version: int,
    policy_fingerprint: str,
    expected_documents: int | None,
) -> dict[str, Any]:
    """Invalidate any previous release before a new dataset build starts."""
    if (
        isinstance(generation_policy_version, bool)
        or not isinstance(generation_policy_version, int)
        or generation_policy_version < MINIMUM_GENERATION_POLICY_VERSION
    ):
        raise ValueError(
            "generation_policy_version must be an integer at or above "
            f"v{MINIMUM_GENERATION_POLICY_VERSION}"
        )
    if not policy_fingerprint.strip():
        raise ValueError("policy_fingerprint must not be empty")
    effective_expected_documents = (
        settings.get("teacher.expected_training_documents")
        if expected_documents is None
        else expected_documents
    )
    if (
        isinstance(effective_expected_documents, bool)
        or not isinstance(effective_expected_documents, int)
        or effective_expected_documents < 2
    ):
        raise ValueError("effective expected_documents must be an integer of at least two")
    manifest: dict[str, Any] = {
        "schema_version": DATASET_RELEASE_SCHEMA_VERSION,
        "schema": "giis-sft-release-v2",
        "status": "building",
        "started_at": _utc_now(),
        "generation_policy_version": generation_policy_version,
        "policy_fingerprint": policy_fingerprint,
        "expected_documents": effective_expected_documents,
    }
    write_json_atomic(dataset_release_path(settings), manifest)
    return manifest


def finalize_dataset_release(
    settings: Settings,
    *,
    generation_policy_version: int,
    policy_fingerprint: str,
    build_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish a complete release after every JSONL has been written."""
    if (
        isinstance(generation_policy_version, bool)
        or not isinstance(generation_policy_version, int)
        or generation_policy_version < MINIMUM_GENERATION_POLICY_VERSION
    ):
        raise ValueError(
            "generation_policy_version must be an integer at or above "
            f"v{MINIMUM_GENERATION_POLICY_VERSION}"
        )
    if not policy_fingerprint.strip():
        raise ValueError("policy_fingerprint must not be empty")

    selection_path = settings.path("paths.training_selection_manifest")
    selection = _read_json_object(selection_path, "training selection manifest")
    selection_digest = _required_text(
        selection, "selection_digest", "training selection manifest"
    )
    selected_document_count = _required_nonnegative_int(
        selection, "selected_document_count", "training selection manifest"
    )
    expected_document_count = _required_nonnegative_int(
        selection, "expected_document_count", "training selection manifest"
    )
    if selected_document_count < 2:
        raise DatasetReleaseError(
            "Invalid training selection manifest: selected_document_count must be at least two"
        )
    if expected_document_count != selected_document_count:
        raise DatasetReleaseError(
            "Invalid training selection manifest: expected and selected counts differ"
        )

    sft_dir = settings.path("paths.sft_dir")
    files = {name: _file_descriptor(sft_dir / name) for name in _RELEASE_FILES}
    for required_split in ("train.jsonl", "validation.jsonl"):
        if files[required_split]["row_count"] < 1:
            raise DatasetReleaseError(
                f"Cannot release an empty required split: {sft_dir / required_split}"
            )

    previous = _read_json_object(dataset_release_path(settings), "dataset release manifest")
    started_at = _required_text(previous, "started_at", "building dataset release manifest")
    if previous.get("status") != "building":
        raise DatasetReleaseError(
            "Dataset release must be marked building before it can be finalized"
        )
    if previous.get("generation_policy_version") != generation_policy_version:
        raise DatasetReleaseError(
            "Building release generation_policy_version does not match finalization"
        )
    if previous.get("policy_fingerprint") != policy_fingerprint:
        raise DatasetReleaseError(
            "Building release policy_fingerprint does not match finalization"
        )
    if previous.get("expected_documents") != expected_document_count:
        raise DatasetReleaseError(
            "Building release expected document count does not match the training selection"
        )

    manifest: dict[str, Any] = {
        "schema_version": DATASET_RELEASE_SCHEMA_VERSION,
        "schema": "giis-sft-release-v2",
        "status": "complete",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "generation_policy_version": generation_policy_version,
        "policy_fingerprint": policy_fingerprint,
        "training_selection": {
            "name": selection_path.name,
            "sha256": _sha256(selection_path, "training selection manifest"),
            "selection_digest": selection_digest,
            "expected_document_count": expected_document_count,
            "selected_document_count": selected_document_count,
        },
        "files": files,
        "build_metadata": dict(build_metadata or {}),
    }
    # This write is deliberately last. A crash before this point leaves status=building,
    # which prevents stale JSONLs from being mistaken for a complete release.
    write_json_atomic(dataset_release_path(settings), manifest)
    return validate_dataset_release(settings)


def validate_dataset_release(
    settings: Settings,
    *,
    minimum_generation_policy_version: int = MINIMUM_GENERATION_POLICY_VERSION,
) -> dict[str, Any]:
    """Verify release state, selection identity, JSONL counts, and byte hashes."""
    path = dataset_release_path(settings)
    manifest = _read_json_object(path, "dataset release manifest")
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_DATASET_RELEASE_SCHEMA_VERSIONS:
        raise DatasetReleaseError(
            "Unsupported dataset release schema_version in " f"{path}"
        )
    if schema_version == 2:
        schema_name = _required_text(manifest, "schema", "dataset release manifest")
        if not schema_name.endswith("release-v2"):
            raise DatasetReleaseError(
                f"Invalid schema name for dataset release schema version 2: {schema_name}"
            )
    if manifest.get("status") != "complete":
        raise DatasetReleaseError(
            f"Dataset release is not complete ({manifest.get('status')!r}): {path}"
        )
    policy_version = _required_nonnegative_int(
        manifest, "generation_policy_version", "dataset release manifest"
    )
    if policy_version < minimum_generation_policy_version:
        raise DatasetReleaseError(
            "Dataset release generation policy is stale: "
            f"v{policy_version} < required v{minimum_generation_policy_version}"
        )
    _required_text(manifest, "policy_fingerprint", "dataset release manifest")

    recorded_selection = manifest.get("training_selection")
    if not isinstance(recorded_selection, dict):
        raise DatasetReleaseError(
            "Invalid dataset release manifest: training_selection must be an object"
        )
    selection_path = settings.path("paths.training_selection_manifest")
    current_selection = _read_json_object(selection_path, "training selection manifest")
    current_digest = _required_text(
        current_selection, "selection_digest", "training selection manifest"
    )
    recorded_digest = _required_text(
        recorded_selection, "selection_digest", "released training selection"
    )
    if recorded_digest != current_digest:
        raise DatasetReleaseError(
            "Training selection digest differs from the released dataset selection"
        )
    current_selection_count = _required_nonnegative_int(
        current_selection, "selected_document_count", "training selection manifest"
    )
    recorded_selection_count = _required_nonnegative_int(
        recorded_selection, "selected_document_count", "released training selection"
    )
    if recorded_selection_count != current_selection_count:
        raise DatasetReleaseError(
            "Training selection document count differs from the released dataset selection"
        )
    current_expected_count = _required_nonnegative_int(
        current_selection, "expected_document_count", "training selection manifest"
    )
    recorded_expected_count = _required_nonnegative_int(
        recorded_selection, "expected_document_count", "released training selection"
    )
    if current_expected_count != current_selection_count:
        raise DatasetReleaseError(
            "Training selection expected count does not match its selected count"
        )
    if recorded_expected_count != current_expected_count:
        raise DatasetReleaseError(
            "Training selection expected count differs from the released dataset selection"
        )
    recorded_selection_hash = _required_text(
        recorded_selection, "sha256", "released training selection"
    )
    if _sha256(selection_path, "training selection manifest") != recorded_selection_hash:
        raise DatasetReleaseError(
            "Training selection manifest bytes differ from the released dataset selection"
        )

    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, dict):
        raise DatasetReleaseError("Invalid dataset release manifest: files must be an object")
    sft_dir = settings.path("paths.sft_dir")
    verified_files: dict[str, Any] = {}
    for name in _RELEASE_FILES:
        recorded = recorded_files.get(name)
        if not isinstance(recorded, dict):
            raise DatasetReleaseError(
                f"Dataset release manifest is missing the descriptor for {name}"
            )
        if recorded.get("name") != name:
            raise DatasetReleaseError(f"Dataset release filename mismatch for {name}")
        expected_hash = _required_text(recorded, "sha256", f"released file {name}")
        expected_count = _required_nonnegative_int(
            recorded, "row_count", f"released file {name}"
        )
        current = _file_descriptor(sft_dir / name)
        if current["sha256"] != expected_hash:
            raise DatasetReleaseError(f"Released dataset artifact hash mismatch: {name}")
        if current["row_count"] != expected_count:
            raise DatasetReleaseError(f"Released dataset artifact row-count mismatch: {name}")
        recorded_size = _required_nonnegative_int(
            recorded, "size_bytes", f"released file {name}"
        )
        if current["size_bytes"] != recorded_size:
            raise DatasetReleaseError(f"Released dataset artifact size mismatch: {name}")
        verified_files[name] = current
    for required_split in ("train.jsonl", "validation.jsonl"):
        if verified_files[required_split]["row_count"] < 1:
            raise DatasetReleaseError(f"Released required split is empty: {required_split}")
    return manifest
