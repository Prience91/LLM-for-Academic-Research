from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from giis_prod.config import Settings
from giis_prod.training import _load_and_validate_sft_exclusions, _sha256_file


class TrainingExclusionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Settings:
        (root / "artifacts/sft").mkdir(parents=True)
        (root / "artifacts/audits/semantic").mkdir(parents=True)
        (root / "artifacts/sft/train.jsonl").write_text(
            json.dumps({"example_id": "keep"}) + "\n"
            + json.dumps({"example_id": "drop"}) + "\n",
            encoding="utf-8",
        )
        (root / "artifacts/sft/validation.jsonl").write_text(
            json.dumps({"example_id": "valid"}) + "\n", encoding="utf-8"
        )
        queue = root / "artifacts/sft/semantic_audit_queue.jsonl"
        queue.write_text(json.dumps({"example_id": "drop"}) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "source_audit_queue_sha256": _sha256_file(queue),
            "expected_exclusion_count": 1,
            "excluded_examples": [
                {"example_id": "drop", "reason_codes": ["semantic_failure"]}
            ],
        }
        (root / "artifacts/audits/semantic/sft_exclusions.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return Settings(
            root=root,
            data={
                "paths": {
                    "sft_dir": "artifacts/sft",
                    "semantic_audit_queue": "artifacts/sft/semantic_audit_queue.jsonl",
                    "sft_exclusions": "artifacts/audits/semantic/sft_exclusions.json",
                }
            },
        )

    def test_valid_overlay_identifies_exact_release_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = _load_and_validate_sft_exclusions(self._fixture(Path(directory)))
            self.assertEqual(result["excluded_ids"], ["drop"])
            self.assertEqual(result["split_counts"], {"train": 1, "validation": 0})

    def test_overlay_fails_closed_when_queue_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._fixture(root)
            with (root / "artifacts/sft/semantic_audit_queue.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "queue digest"):
                _load_and_validate_sft_exclusions(settings)

    def test_overlay_fails_closed_when_id_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._fixture(root)
            (root / "artifacts/sft/train.jsonl").write_text(
                json.dumps({"example_id": "keep"}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "exactly once"):
                _load_and_validate_sft_exclusions(settings)


if __name__ == "__main__":
    unittest.main()
