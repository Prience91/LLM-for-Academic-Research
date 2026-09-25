from __future__ import annotations

import unittest
from types import SimpleNamespace

from giis_prod.generation import GroundedGenerator, _completion_result
from giis_prod.models import Evidence, ResponseMode


class GenerationCitationPolicyTests(unittest.TestCase):
    def test_brief_source_summary_stream_yields_early_and_finishes_without_markers(self):
        evidence = [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="The paper reports a supported purpose and finding.",
                title="Paper",
                authors=[],
                year=2024,
                journal=None,
                page_start=1,
                page_end=1,
            )
        ]
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Purpose: Supported purpose."),
                        finish_reason=None,
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=" Findings: Supported finding."),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return iter(chunks)

        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "brief_summary_max_chunks": 8,
            "brief_summary_target_words": 250,
            "brief_summary_max_tokens": 550,
            "no_evidence_message": "No corpus evidence.",
        }
        generator.retriever = SimpleNamespace(
            document_evidence=lambda document_id, max_chunks: evidence
        )
        generator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        generator.model = "merged-model"

        updates = list(generator.summarize_source_stream("doc-1"))

        self.assertTrue(calls[0]["stream"])
        self.assertEqual(
            calls[0]["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(updates[0], "Purpose: Supported purpose.")
        self.assertEqual(
            updates[-1],
            "Purpose: Supported purpose. Findings: Supported finding.",
        )
        self.assertNotIn("[E1]", updates[-1])

    def test_streamed_summary_finishes_with_validation_message_when_truncated(self):
        evidence = [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="Supported evidence.",
                title="Paper",
                authors=[],
                year=None,
                journal=None,
                page_start=1,
                page_end=1,
            )
        ]
        chunks = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Purpose: Incomplete"),
                            finish_reason="length",
                        )
                    ]
                )
            ]
        )
        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "brief_summary_max_chunks": 8,
            "brief_summary_target_words": 250,
            "brief_summary_max_tokens": 550,
            "no_evidence_message": "No corpus evidence.",
            "summary_validation_failure_message": "Summary incomplete.",
        }
        generator.retriever = SimpleNamespace(
            document_evidence=lambda document_id, max_chunks: evidence
        )
        generator.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: chunks)
            )
        )
        generator.model = "merged-model"

        updates = list(generator.summarize_source_stream("doc-1"))

        self.assertEqual(updates, ["Purpose: Incomplete", "Summary incomplete."])

    def test_brief_source_summary_uses_one_bounded_call_and_no_repair(self):
        evidence = [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="The paper reports a supported method and finding.",
                title="Paper",
                authors=["Ada Researcher"],
                year=2024,
                journal="Journal",
                page_start=1,
                page_end=1,
            )
        ]
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Purpose: The paper reports its supported purpose clearly. "
                            "Findings: The paper reports its supported finding clearly."
                        )
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=60, total_tokens=180),
        )
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return completion

        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "brief_summary_max_chunks": 8,
            "brief_summary_target_words": 250,
            "brief_summary_max_tokens": 550,
            "minimum_citation_coverage": 0.8,
            "no_evidence_message": "No corpus evidence.",
        }
        generator.retriever = SimpleNamespace(
            document_evidence=lambda document_id, max_chunks: evidence
        )
        generator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        generator.model = "merged-model"

        result = generator.summarize_source("doc-1")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_tokens"], 550)
        self.assertEqual(
            calls[0]["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertFalse(result.abstained)
        self.assertEqual(result.mode, ResponseMode.SOURCE_SUMMARY)
        self.assertEqual(result.repair_attempts, 0)
        self.assertNotIn("[E1]", result.answer)
        self.assertNotIn("#giis-source-", result.answer)
        self.assertIn("Purpose:", result.answer)
        prompt = calls[0]["messages"][0]["content"] + calls[0]["messages"][1]["content"]
        self.assertIn("Do not emit [E#] markers", prompt)

    def test_brief_source_summary_rejects_token_truncation_without_retry(self):
        evidence = [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="Supported evidence.",
                title="Paper",
                authors=[],
                year=2024,
                journal=None,
                page_start=1,
                page_end=1,
            )
        ]
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Purpose: An incomplete summary"),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "brief_summary_max_chunks": 8,
            "brief_summary_target_words": 250,
            "brief_summary_max_tokens": 550,
            "no_evidence_message": "No corpus evidence.",
            "summary_validation_failure_message": "Summary incomplete.",
        }
        generator.retriever = SimpleNamespace(
            document_evidence=lambda document_id, max_chunks: evidence
        )
        generator.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: completion)
            )
        )
        generator.model = "merged-model"

        result = generator.summarize_source("doc-1")

        self.assertTrue(result.abstained)
        self.assertEqual(result.answer, "Summary incomplete.")
        self.assertIn("truncated_source_summary", result.rejection_reason_codes)

    def test_deployment_prompt_prohibits_second_hand_sources(self):
        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "target_long_form_words": 850,
            "maximum_long_form_words": 1100,
        }

        prompt = generator._system_prompt(ResponseMode.STANDARD)

        self.assertIn("own Document ID", prompt)
        self.assertIn("omit all second-hand citations", prompt)
        self.assertIn("exact evidence markers", prompt)

    def test_completion_usage_is_preserved_for_evaluation(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" Answer [E1]. "))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

        result = _completion_result(completion)

        self.assertEqual(result.text, "Answer [E1].")
        self.assertIsNone(result.finish_reason)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 20)
        self.assertEqual(result.total_tokens, 120)

    def test_rejection_reasons_identify_failed_response_gates(self):
        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "minimum_citation_coverage": 0.8,
            "minimum_long_form_words": 350,
        }
        evidence = [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="Evidence",
                title="Paper",
                authors=[],
                year=None,
                journal=None,
                page_start=1,
                page_end=1,
            )
        ]

        reasons = generator._response_rejection_reasons(
            "This unsupported response has enough words to count as an uncited claim.",
            evidence,
            ResponseMode.STANDARD,
        )

        self.assertIn("invalid_or_missing_evidence_markers", reasons)
        self.assertIn("insufficient_citation_coverage", reasons)

    def test_token_limit_is_a_rejection_reason(self):
        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "minimum_citation_coverage": 0.8,
            "minimum_long_form_words": 350,
        }
        evidence = [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="Evidence",
                title="Paper",
                authors=[],
                year=None,
                journal=None,
                page_start=1,
                page_end=1,
            )
        ]

        reasons = generator._response_rejection_reasons(
            "Supported response [E1].",
            evidence,
            ResponseMode.STANDARD,
            finish_reason="length",
        )

        self.assertIn("truncated_at_completion_limit", reasons)

    def test_truncated_standard_repair_uses_long_form_capacity(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Repaired answer [E1]."),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

        generator = GroundedGenerator.__new__(GroundedGenerator)
        generator.response = {
            "minimum_long_form_words": 350,
            "standard_max_tokens": 900,
            "summary_max_tokens": 550,
            "long_form_max_tokens": 2200,
        }
        generator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        generator.model = "merged-model"
        evidence = [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                text="Evidence",
                title="Paper",
                authors=[],
                year=None,
                journal=None,
                page_start=1,
                page_end=1,
            )
        ]

        generator._repair_once(
            "question",
            "truncated draft [E1].",
            evidence,
            ResponseMode.STANDARD,
            expanded_capacity=True,
        )

        self.assertEqual(calls[0]["max_tokens"], 2200)


if __name__ == "__main__":
    unittest.main()
