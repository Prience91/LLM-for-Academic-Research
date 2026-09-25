import json
import tempfile
import unittest
from pathlib import Path

from giis_prod.config import Settings
from giis_prod.evaluation import (
    BenchmarkCase,
    SystemEvaluator,
    _paired_bootstrap_bounds,
    percentile,
)
from giis_prod.generation import AnswerResult
from giis_prod.models import Evidence, ResponseMode


class EvaluationMathTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = _paired_bootstrap_bounds(
            [0.0, 0.1, -0.1, 0.0], confidence=0.95, samples=500, seed=3407
        )
        second = _paired_bootstrap_bounds(
            [0.0, 0.1, -0.1, 0.0], confidence=0.95, samples=500, seed=3407
        )
        self.assertEqual(first, second)

    def test_identical_pairs_have_zero_bounds(self) -> None:
        self.assertEqual(
            _paired_bootstrap_bounds(
                [0.0] * 30, confidence=0.95, samples=100, seed=3407
            ),
            (0.0, 0.0, 0.0),
        )


class _Settings:
    def get(self, key: str):
        values = {
            "response.minimum_long_form_words": 350,
        }
        return values[key]


class _Generator:
    def __init__(self, result: AnswerResult):
        self.result = result

    def answer(self, *args, **kwargs) -> AnswerResult:
        return self.result


class EvaluationAuditTests(unittest.TestCase):
    def test_case_record_preserves_answer_provenance_and_usage(self) -> None:
        evidence = Evidence(
            evidence_id="E1",
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Supported evidence.",
            title="Paper",
            authors=["Researcher"],
            year=2024,
            journal="Journal",
            page_start=2,
            page_end=2,
        )
        result = AnswerResult(
            answer="Supported claim [E1].",
            references_markdown="## Reference Panel",
            mode=ResponseMode.STANDARD,
            abstained=False,
            evidence=[evidence],
            rewritten_query="question",
            raw_marker_answer="Supported claim [E1].",
            rejection_reason_codes=[],
            repair_attempts=1,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        evaluator = SystemEvaluator.__new__(SystemEvaluator)
        evaluator.settings = _Settings()
        evaluator.generator = _Generator(result)

        detail = evaluator.evaluate_case(
            BenchmarkCase(
                case_id="case-1",
                query="question",
                category="fact",
                expected_document_ids=["doc-1"],
                expect_abstention=False,
                requested_mode=ResponseMode.STANDARD,
            )
        )

        self.assertEqual(detail["answer_text"], "Supported claim [E1].")
        self.assertEqual(detail["raw_marker_answer"], "Supported claim [E1].")
        self.assertEqual(detail["retrieved_chunk_ids"], ["chunk-1"])
        self.assertEqual(detail["evidence"][0]["document_id"], "doc-1")
        self.assertEqual(detail["repair_attempts"], 1)
        self.assertEqual(detail["total_tokens"], 120)
        self.assertTrue(detail["raw_citation_markers_valid"])

    def test_unexercised_long_form_metric_is_null(self) -> None:
        evaluator = SystemEvaluator.__new__(SystemEvaluator)
        evaluator.evaluate_case = lambda case: {
            "case_id": case.case_id,
            "expect_abstention": False,
            "abstention_correct": True,
            "retrieval_recall": 1.0,
            "citation_markers_valid": True,
            "citation_coverage": 1.0,
            "long_form_compliant": None,
            "latency_seconds": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "query": "question",
                        "category": "fact",
                        "expected_document_ids": ["doc-1"],
                        "expect_abstention": False,
                        "requested_mode": "standard",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = evaluator.run(benchmark, root / "results")

        self.assertIsNone(summary["long_form_compliance"])

    def test_supported_benchmark_must_use_frozen_heldout_documents(self) -> None:
        evaluator = SystemEvaluator.__new__(SystemEvaluator)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator.settings = Settings(
                root=root,
                data={"paths": {"heldout_manifest": "artifacts/splits/heldout.json"}},
            )
            heldout = evaluator.settings.path("paths.heldout_manifest")
            heldout.parent.mkdir(parents=True)
            heldout.write_text(
                json.dumps({"heldout_document_ids": ["doc-heldout"]}),
                encoding="utf-8",
            )
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "case_id": "leaked",
                        "query": "question",
                        "category": "fact",
                        "expected_document_ids": ["doc-trained"],
                        "expect_abstention": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outside the frozen held-out"):
                evaluator.run(benchmark, root / "results")


if __name__ == "__main__":
    unittest.main()
