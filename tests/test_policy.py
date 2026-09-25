from __future__ import annotations

import unittest

from giis_prod.models import Evidence, ResponseMode
from giis_prod.policy import evidence_is_sufficient, select_response_mode


def evidence(document_id: str) -> Evidence:
    return Evidence(
        evidence_id="E1",
        chunk_id=f"chunk-{document_id}",
        document_id=document_id,
        text="Evidence text",
        title="Title",
        authors=["Ada Researcher"],
        year=2025,
        journal="Journal",
        page_start=1,
        page_end=1,
    )


class PolicyTests(unittest.TestCase):
    def test_closed_question_remains_standard(self):
        self.assertEqual(
            select_response_mode("Does the study report a statistically significant effect?"),
            ResponseMode.STANDARD,
        )

    def test_open_research_synthesis_is_long_form(self):
        self.assertEqual(
            select_response_mode(
                "How can local communities adapt to climate change for addressing water security problems?"
            ),
            ResponseMode.LONG_FORM,
        )

    def test_long_form_requires_distinct_sources(self):
        self.assertFalse(evidence_is_sufficient([evidence("d1")], ResponseMode.LONG_FORM, 2))
        self.assertTrue(
            evidence_is_sufficient(
                [evidence("d1"), evidence("d2")], ResponseMode.LONG_FORM, 2
            )
        )


if __name__ == "__main__":
    unittest.main()

