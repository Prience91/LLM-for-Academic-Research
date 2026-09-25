from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import giis_pipeline
from giis_prod.config import Settings
from giis_prod.dataset_release import (
    DatasetReleaseError,
    begin_dataset_release,
    dataset_release_path,
    finalize_dataset_release,
    validate_dataset_release,
)
from giis_prod.production_readiness import REQUIRED_PRODUCTION_GATES
from giis_prod.training import train_sft


def _settings(root: Path) -> Settings:
    readiness = {key: False for key, _ in REQUIRED_PRODUCTION_GATES}
    return Settings(
        root=root,
        data={
            "paths": {
                "pdf_dir": "Papers_RAG",
                "training_pdf_dir": "Papers_Training",
                "metadata_overrides": "data/metadata.jsonl",
                "document_registry": "artifacts/corpus/documents.jsonl",
                "chunk_file": "artifacts/corpus/chunks.jsonl",
                "active_index": "artifacts/index/active_index.json",
                "sft_dir": "artifacts/sft",
                "training_selection_manifest": "artifacts/sft/training_selection.json",
                "benchmark_file": "benchmark/benchmark.jsonl",
            },
            "models": {
                "adapter_output": "models/adapter",
                "merged_output": "models/merged",
            },
            "teacher": {
                "expected_training_documents": 2500,
                "timed_smoke_max_documents": 5,
            },
            "production_readiness": readiness,
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class DatasetReleaseTests(unittest.TestCase):
    def _prepare_release_inputs(self, settings: Settings) -> None:
        _write_json(
            settings.path("paths.training_selection_manifest"),
            {
                "selection_digest": "selection-v15",
                "selected_document_count": 2,
                "expected_document_count": 2,
                "documents": [{"document_id": "doc-1"}, {"document_id": "doc-2"}],
            },
        )
        sft_dir = settings.path("paths.sft_dir")
        _write_jsonl(sft_dir / "train.jsonl", [{"id": "train-1"}])
        _write_jsonl(sft_dir / "validation.jsonl", [{"id": "validation-1"}])
        _write_jsonl(sft_dir / "rejected.jsonl", [])

    def _complete_release(self, settings: Settings) -> dict:
        self._prepare_release_inputs(settings)
        begin_dataset_release(
            settings,
            generation_policy_version=15,
            policy_fingerprint="policy-v15",
            expected_documents=2,
        )
        return finalize_dataset_release(
            settings,
            generation_policy_version=15,
            policy_fingerprint="policy-v15",
            build_metadata={"train_examples": 1, "validation_examples": 1},
        )

    def test_complete_release_hashes_counts_and_selection_validate(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            manifest = self._complete_release(settings)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["schema"], "giis-sft-release-v2")
            self.assertEqual(manifest["files"]["train.jsonl"]["row_count"], 1)
            self.assertEqual(manifest["files"]["rejected.jsonl"]["row_count"], 0)
            self.assertEqual(
                manifest["training_selection"]["selection_digest"], "selection-v15"
            )
            self.assertEqual(validate_dataset_release(settings), manifest)

    def test_legacy_schema_version_one_remains_readable(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            self._complete_release(settings)
            path = dataset_release_path(settings)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 1
            manifest.pop("schema", None)
            _write_json(path, manifest)
            self.assertEqual(validate_dataset_release(settings)["schema_version"], 1)

    def test_absent_and_building_manifests_block_consumption(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            self._prepare_release_inputs(settings)
            with self.assertRaisesRegex(DatasetReleaseError, "Missing dataset release"):
                validate_dataset_release(settings)
            begin_dataset_release(
                settings,
                generation_policy_version=15,
                policy_fingerprint="policy-v15",
                expected_documents=2,
            )
            with self.assertRaisesRegex(DatasetReleaseError, "not complete"):
                validate_dataset_release(settings)

    def test_mutated_jsonl_with_same_row_count_is_rejected_by_hash(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            self._complete_release(settings)
            _write_jsonl(
                settings.path("paths.sft_dir") / "train.jsonl",
                [{"id": "tampered"}],
            )
            with self.assertRaisesRegex(DatasetReleaseError, "hash mismatch"):
                validate_dataset_release(settings)

    def test_manifest_row_count_mismatch_is_rejected(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            self._complete_release(settings)
            path = dataset_release_path(settings)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["files"]["train.jsonl"]["row_count"] += 1
            _write_json(path, manifest)
            with self.assertRaisesRegex(DatasetReleaseError, "row-count mismatch"):
                validate_dataset_release(settings)

    def test_release_from_an_older_generation_policy_is_rejected(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            self._complete_release(settings)
            path = dataset_release_path(settings)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["generation_policy_version"] = 14
            _write_json(path, manifest)
            with self.assertRaisesRegex(DatasetReleaseError, "policy is stale"):
                validate_dataset_release(settings)

    def test_changed_selection_digest_blocks_consumption(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            self._complete_release(settings)
            selection_path = settings.path("paths.training_selection_manifest")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["selection_digest"] = "different-selection"
            _write_json(selection_path, selection)
            with self.assertRaisesRegex(DatasetReleaseError, "selection digest differs"):
                validate_dataset_release(settings)

    def test_finalization_rejects_selection_count_different_from_requested_build(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            self._prepare_release_inputs(settings)
            begin_dataset_release(
                settings,
                generation_policy_version=15,
                policy_fingerprint="policy-v15",
                expected_documents=5,
            )
            with self.assertRaisesRegex(
                DatasetReleaseError, "expected document count does not match"
            ):
                finalize_dataset_release(
                    settings,
                    generation_policy_version=15,
                    policy_fingerprint="policy-v15",
                )

    def test_training_preflight_fails_before_heavy_framework_imports(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            imported: list[str] = []
            original_import = __import__

            def tracking_import(name, *args, **kwargs):
                imported.append(name)
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=tracking_import):
                with self.assertRaises(DatasetReleaseError):
                    train_sft(settings)
            self.assertNotIn("torch", imported)
            self.assertNotIn("transformers", imported)
            self.assertNotIn("trl", imported)

    def test_status_recommends_dataset_when_stale_jsonls_have_no_release(self):
        with TemporaryDirectory() as temporary:
            settings = _settings(Path(temporary))
            for directory in ("Papers_RAG", "Papers_Training"):
                for index in (1, 2):
                    path = Path(temporary) / directory / f"paper-{index}.pdf"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"%PDF-smoke")
            for dotted_key in (
                "paths.document_registry",
                "paths.chunk_file",
                "paths.active_index",
            ):
                _write_json(settings.path(dotted_key), {})
            _write_jsonl(
                settings.path("paths.sft_dir") / "train.jsonl", [{"id": "old-train"}]
            )
            _write_jsonl(
                settings.path("paths.sft_dir") / "validation.jsonl",
                [{"id": "old-validation"}],
            )

            status = giis_pipeline.pipeline_status(settings)

            self.assertFalse(status["dataset_release"]["valid"])
            self.assertIn("dataset --expected-documents 2", status["next_step"])


if __name__ == "__main__":
    unittest.main()
