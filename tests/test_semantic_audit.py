from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from giis_prod.config import Settings
from giis_prod.io import read_jsonl, write_jsonl_atomic
from giis_prod.semantic_audit import plan_semantic_audit


class SemanticAuditPlanTests(unittest.TestCase):
    def test_audit_uses_ten_percent_below_cap(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "smoke_run_ii": {
                        "semantic_audit_fraction": 0.10,
                        "semantic_audit_maximum": 400,
                        "seed": 3407,
                    },
                },
            )
            rows = [
                {
                    "example_id": f"sft_{index}",
                    "split": "train",
                    "dataset_kind": "core_single_source",
                    "document_ids": [f"d{index % 10}"],
                    "answer_style": "table" if index % 5 == 0 else "prose",
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Supported claim [E1]."},
                    ],
                }
                for index in range(90)
            ]
            write_jsonl_atomic(settings.path("paths.sft_dir") / "train.jsonl", rows)
            write_jsonl_atomic(settings.path("paths.sft_dir") / "validation.jsonl", [])
            manifest = plan_semantic_audit(settings)
            queue = list(
                read_jsonl(settings.path("paths.sft_dir") / "semantic_audit_queue.jsonl")
            )
        self.assertEqual(manifest["audit_candidates"], 9)
        self.assertEqual(len(queue), 9)
        self.assertTrue(all("audit_stratum" in row for row in queue))


if __name__ == "__main__":
    unittest.main()
