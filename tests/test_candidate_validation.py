from __future__ import annotations

import unittest

from giis_prod.candidate_validation import validate_answer_style


class CandidateStyleValidationTests(unittest.TestCase):
    def test_grounded_markdown_table_is_accepted(self):
        valid, reason = validate_answer_style(
            {
                "answer_style": "table",
                "answer": (
                    "| Dimension | Finding |\n"
                    "|---|---|\n"
                    "| Exposure | Increased heat [E1] |\n"
                    "| Response | Earlier planting [E2] |"
                ),
            }
        )
        self.assertTrue(valid, reason)

    def test_uncited_table_row_is_rejected(self):
        valid, reason = validate_answer_style(
            {
                "answer_style": "table",
                "answer": "| A | B |\n|---|---|\n| Heat | Adaptation |",
            }
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "table_row_missing_evidence_marker")

    def test_safe_grounded_mermaid_subset_is_accepted(self):
        answer = """```mermaid
flowchart TD
A[\"Heat exposure [E1]\"]
B[\"Yield loss [E2]\"]
A --> B
```"""
        valid, reason = validate_answer_style(
            {"answer_style": "concept_diagram", "answer": answer}
        )
        self.assertTrue(valid, reason)

    def test_mermaid_click_directive_is_rejected(self):
        answer = """```mermaid
flowchart TD
A[\"Heat [E1]\"]
B[\"Loss [E2]\"]
A --> B
click A https://example.org
```"""
        valid, reason = validate_answer_style(
            {"answer_style": "concept_diagram", "answer": answer}
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "unsafe_mermaid_directive")


if __name__ == "__main__":
    unittest.main()
