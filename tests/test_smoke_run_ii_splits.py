from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from giis_prod.config import Settings
from giis_prod.io import write_jsonl_atomic
from giis_prod.splits import plan_heldout_split
from giis_prod.training_materialization import (
    TrainingMaterializationError,
    materialize_training,
)


class SmokeRunIISplitTests(unittest.TestCase):
    @staticmethod
    def _materialization_settings(root: Path) -> Settings:
        return Settings(
            root=root,
            data={
                "paths": {
                    "pdf_dir": "Papers_RAG",
                    "training_pdf_dir": "Papers_Training",
                    "document_registry": "artifacts/corpus/documents.jsonl",
                    "heldout_manifest": "artifacts/splits/heldout_manifest.json",
                    "training_materialization_report": (
                        "artifacts/splits/training_materialization.json"
                    ),
                },
                "smoke_run_ii": {
                    "expected_corpus_documents": 4,
                    "expected_sft_documents": 3,
                    "heldout_documents": 1,
                    "seed": 3407,
                },
            },
        )

    @staticmethod
    def _prepare_materialization(root: Path, settings: Settings) -> dict[str, object]:
        rag = settings.path("paths.pdf_dir")
        rag.mkdir(parents=True)
        rows = []
        for index in range(4):
            payload = f"paper-{index}".encode()
            source_file = f"paper-{index}.pdf"
            (rag / source_file).write_bytes(payload)
            rows.append(
                {
                    "document_id": f"d{index}",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source_file": source_file,
                    "title": f"Paper {index}",
                    "metadata_status": "verified",
                }
            )
        write_jsonl_atomic(settings.path("paths.document_registry"), rows)
        return plan_heldout_split(settings)

    def test_canary_split_is_paper_level_deterministic_and_disjoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                data={
                    "paths": {
                        "document_registry": "artifacts/corpus/documents.jsonl",
                        "heldout_manifest": "artifacts/splits/heldout_manifest.json",
                    },
                    "smoke_run_ii": {
                        "expected_corpus_documents": 700,
                        "expected_sft_documents": 630,
                        "heldout_documents": 70,
                        "seed": 3407,
                    },
                },
            )
            write_jsonl_atomic(
                settings.path("paths.document_registry"),
                (
                    {
                        "document_id": f"d{index}",
                        "sha256": f"hash-{index}",
                        "source_file": f"paper-{index}.pdf",
                        "title": f"Paper {index}",
                        "metadata_status": "verified",
                    }
                    for index in range(10)
                ),
            )
            first = plan_heldout_split(
                settings, expected_corpus_documents=10, heldout_documents=2
            )
            second = plan_heldout_split(
                settings, expected_corpus_documents=10, heldout_documents=2
            )
            stored = json.loads(
                settings.path("paths.heldout_manifest").read_text(encoding="utf-8")
            )
        self.assertEqual(first["heldout_document_ids"], second["heldout_document_ids"])
        self.assertFalse(
            set(first["heldout_document_ids"]) & set(first["sft_document_ids"])
        )
        self.assertEqual(len(first["heldout_document_ids"]), 2)
        self.assertEqual(len(first["sft_document_ids"]), 8)
        self.assertEqual(stored["selection_digest"], first["selection_digest"])

    def test_split_selects_exact_release_from_larger_verified_pool(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = self._materialization_settings(root)
            rows = [
                {
                    "document_id": f"d{index}",
                    "sha256": f"hash-{index}",
                    "source_file": f"paper-{index}.pdf",
                    "title": f"Paper {index}",
                    "metadata_status": "verified",
                }
                for index in range(6)
            ]
            write_jsonl_atomic(settings.path("paths.document_registry"), rows)
            first = plan_heldout_split(settings)
            second = plan_heldout_split(settings)
        self.assertEqual(len(first["documents"]), 4)
        self.assertEqual(first["verified_pool_documents"], 6)
        self.assertEqual(first["surplus_verified_documents"], 2)
        self.assertEqual(first["selection_digest"], second["selection_digest"])
        self.assertEqual(first["surplus_document_ids"], second["surplus_document_ids"])

    def test_materialization_allows_unselected_raw_intake_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = self._materialization_settings(root)
            self._prepare_materialization(root, settings)
            (settings.path("paths.pdf_dir") / "unselected.pdf").write_bytes(b"surplus")
            report = materialize_training(settings)
        self.assertEqual(report["materialized_sft_documents"], 3)
        self.assertEqual(report["heldout_leakage_count"], 0)

    def test_materializes_only_sft_pool_and_is_idempotent(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = self._materialization_settings(root)
            manifest = self._prepare_materialization(root, settings)
            first = materialize_training(settings)
            second = materialize_training(settings)
            names = {path.name for path in settings.path("paths.training_pdf_dir").iterdir()}
            heldout_names = {
                row["source_file"]
                for row in manifest["documents"]
                if row["partition"] == "heldout_benchmark"
            }
        self.assertEqual(first["materialized_sft_documents"], 3)
        self.assertEqual(first["copied_documents"], 3)
        self.assertEqual(second["copied_documents"], 0)
        self.assertEqual(second["already_present_documents"], 3)
        self.assertEqual(len(names), 3)
        self.assertFalse(names & heldout_names)

    def test_rejects_unexpected_training_file_without_deleting_it(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = self._materialization_settings(root)
            self._prepare_materialization(root, settings)
            training = settings.path("paths.training_pdf_dir")
            training.mkdir(parents=True)
            unexpected = training / "unexpected.pdf"
            unexpected.write_bytes(b"user-owned")
            with self.assertRaisesRegex(TrainingMaterializationError, "outside"):
                materialize_training(settings)
            self.assertEqual(unexpected.read_bytes(), b"user-owned")

    def test_rejects_source_hash_drift_before_copying(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = self._materialization_settings(root)
            manifest = self._prepare_materialization(root, settings)
            selected = next(
                row for row in manifest["documents"] if row["partition"] == "sft_pool"
            )
            (settings.path("paths.pdf_dir") / selected["source_file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(TrainingMaterializationError, "hash mismatch"):
                materialize_training(settings)
            self.assertFalse(settings.path("paths.training_pdf_dir").exists())


if __name__ == "__main__":
    unittest.main()
