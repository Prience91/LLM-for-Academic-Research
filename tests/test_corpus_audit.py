from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from giis_prod.config import Settings
from giis_prod.corpus_audit import CorpusAuditError, CorpusAuditor
from giis_prod.io import write_jsonl_atomic


def _settings(root: Path) -> Settings:
    return Settings(
        root=root,
        data={
            "paths": {
                "document_registry": "artifacts/corpus/documents.jsonl",
                "chunk_file": "artifacts/corpus/chunks.jsonl",
                "heldout_manifest": "artifacts/splits/heldout_manifest.json",
                "active_index": "artifacts/index/active_index.json",
                "corpus_audit_report": "artifacts/audits/corpus_audit.json",
            },
            "corpus_audit": {"minimum_source_title_similarity": 0.80},
            "metadata": {
                "minimum_title_fragment_tokens": 6,
                "minimum_title_fragment_characters": 40,
            },
            "ingestion": {"max_chunk_tokens": 512},
        },
    )


def _document(title: str = "Climate Risk and Coffee") -> dict[str, object]:
    return {
        "document_id": "doc_1",
        "source_file": "Climate Risk and Coffee.pdf",
        "sha256": "a" * 64,
        "title": title,
        "authors": ["A. Author"],
        "year": 2024,
        "journal": "Climate Journal",
        "doi": "10.1234/example",
        "document_type": "journal-article",
        "publisher": "Publisher",
        "volume": "1",
        "issue": "2",
        "page_count": 10,
        "metadata_status": "verified",
        "metadata_provenance": {
            "title": "crossref_doi",
            "authors": "crossref_doi",
            "year": "crossref_doi",
        },
    }


def _chunk(title: str = "Climate Risk and Coffee") -> dict[str, object]:
    return {
        "chunk_id": "chk_1",
        "document_id": "doc_1",
        "text": "Evidence text",
        "title": title,
        "authors": ["A. Author"],
        "year": 2024,
        "journal": "Climate Journal",
        "doi": "10.1234/example",
        "document_type": "journal-article",
        "publisher": "Publisher",
        "volume": "1",
        "issue": "2",
        "page_start": 1,
        "page_end": 1,
        "token_count": 20,
    }


class CorpusAuditTests(unittest.TestCase):
    def _write(self, root: Path, document: dict[str, object], chunk: dict[str, object]) -> None:
        write_jsonl_atomic(root / "artifacts/corpus/documents.jsonl", [document])
        write_jsonl_atomic(root / "artifacts/corpus/chunks.jsonl", [chunk])

    def test_valid_registry_and_chunk_pass_pre_index_audit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, _document(), _chunk())
            report = CorpusAuditor(_settings(root)).run(check_indexes=False)
        self.assertTrue(report["passed"])
        self.assertEqual(report["error_count"], 0)

    def test_collision_prefix_is_ignored_for_source_title_audit(self):
        document = _document()
        document["source_file"] = "034839e995f3__Climate Risk and Coffee.pdf"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, document, _chunk())
            report = CorpusAuditor(_settings(root)).run(check_indexes=False)
        self.assertTrue(report["passed"])

    def test_verified_title_supersedes_tentative_filename(self):
        verified = "A 38-year-old woman with zosteriform skin lesions"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, _document(verified), _chunk(verified))
            report = CorpusAuditor(_settings(root)).run(check_indexes=False)
        self.assertTrue(report["passed"])
        self.assertIn(
            "filename_differs_from_canonical_title",
            {row["code"] for row in report["warnings"]},
        )

    def test_distinctive_truncated_filename_is_a_nonblocking_warning(self):
        document = _document(
            "The relationship among biodiversity, governance, wealth, and scientific "
            "capacity at a country level: Disaggregation and prioritization"
        )
        document["source_file"] = (
            "and scientific capacity at a country level Disaggregation.pdf"
        )
        chunk = _chunk(str(document["title"]))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, document, chunk)
            report = CorpusAuditor(_settings(root)).run(check_indexes=False)
        self.assertTrue(report["passed"])
        self.assertIn(
            "verified_filename_title_truncation",
            {row["code"] for row in report["warnings"]},
        )

    def test_chunk_metadata_substitution_is_blocking(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, _document(), _chunk("Different title"))
            with self.assertRaises(CorpusAuditError):
                CorpusAuditor(_settings(root)).run(check_indexes=False)
            report = __import__("json").loads(
                (root / "artifacts/audits/corpus_audit.json").read_text()
            )
        self.assertIn("chunk_metadata_mismatch", {row["code"] for row in report["errors"]})


if __name__ == "__main__":
    unittest.main()
