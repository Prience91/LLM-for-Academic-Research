from __future__ import annotations

import unittest

from giis_prod.teacher import EvidenceBundle
from giis_prod.teacher_ab import (
    _refined_request_options,
    _validate_support_ledger,
    assert_context_fit,
)


class TeacherABTests(unittest.TestCase):
    def test_context_preflight_accepts_request_without_changing_allowances(self):
        self.assertEqual(assert_context_fit(8000, 5000, 64, 16384), 13064)

    def test_context_preflight_rejects_any_request_that_would_require_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "without truncation"):
            assert_context_fit(12000, 5000, 64, 16384)

    def _bundle(self):
        return EvidenceBundle(
            bundle_id="factorial",
            document_ids=("doc_a", "doc_b"),
            chunks=[
                {"document_id": "doc_a"},
                {"document_id": "doc_a"},
                {"document_id": "doc_b"},
                {"document_id": "doc_b"},
            ],
        )

    def _row(self):
        return {
            "policy_lane": "refined_multisource_v1",
            "source_document_ids": ["doc_a", "doc_b"],
            "relation_mode": "methodological_contrast",
        }

    def _base_request(self):
        return {
            "messages": [{"role": "user", "content": "Rules\nEVIDENCE:\nblocks"}],
            "response_format": {
                "json_schema": {
                    "name": "base",
                    "schema": {
                        "properties": {
                            "examples": {
                                "items": {
                                    "required": ["question", "answer"],
                                    "properties": {
                                        "question": {"type": "string"},
                                        "answer": {"type": "string"},
                                    },
                                }
                            }
                        }
                    },
                }
            },
        }

    def test_refined_lane_adds_ledger_without_mutating_base_request(self):
        base = self._base_request()
        refined = _refined_request_options(base, self._bundle(), self._row())
        self.assertNotIn(
            "support_ledger",
            base["response_format"]["json_schema"]["schema"]["properties"]
            ["examples"]["items"]["properties"],
        )
        item = refined["response_format"]["json_schema"]["schema"]["properties"][
            "examples"
        ]["items"]
        self.assertIn("support_ledger", item["required"])
        self.assertEqual(
            item["properties"]["support_ledger"]["properties"]["relation_type"][
                "enum"
            ],
            ["methodological_contrast"],
        )

    def test_support_ledger_requires_both_sources_and_answer_markers(self):
        candidate = {
            "answer": "First claim [E1]. Second claim [E3].",
            "support_ledger": {
                "source_a_document_id": "doc_a",
                "source_a_claim": "First claim",
                "source_a_evidence": ["[E1]"],
                "source_b_document_id": "doc_b",
                "source_b_claim": "Second claim",
                "source_b_evidence": ["[E3]"],
                "relation_type": "methodological_contrast",
                "relationship_statement": "The methods differ",
                "relationship_evidence": ["[E1]", "[E3]"],
                "unsupported_transfer_present": False,
            },
        }
        self.assertEqual(
            _validate_support_ledger(self._bundle(), self._row(), candidate),
            (True, "ok"),
        )
        candidate["support_ledger"]["relationship_evidence"] = ["[E1]"]
        self.assertEqual(
            _validate_support_ledger(self._bundle(), self._row(), candidate)[1],
            "relationship_ledger_does_not_use_both_sources",
        )


if __name__ == "__main__":
    unittest.main()
