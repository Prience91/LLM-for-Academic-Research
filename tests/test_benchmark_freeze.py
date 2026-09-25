from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from giis_prod.benchmark_freeze import BenchmarkFreezeError, validate_benchmark_freeze
from giis_prod.config import Settings


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BenchmarkFreezeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Settings:
        benchmark = root / "benchmark/benchmark.jsonl"
        reference = root / "benchmark/benchmark_reference.jsonl"
        heldout = root / "artifacts/splits/heldout_manifest.json"
        chunks = root / "artifacts/corpus/chunks.jsonl"

        _write_jsonl(
            benchmark,
            [{
                "case_id": "case-1",
                "query": "What did the study find?",
                "expected_document_ids": ["doc-1"],
                "expect_abstention": False,
                "requested_mode": "standard",
            }],
        )
        _write_jsonl(
            reference,
            [{
                "case_id": "case-1",
                "expected_document_ids": ["doc-1"],
                "supporting_chunk_ids": ["chunk-1"],
                "required_points": ["The supported finding."],
                "semantic_guardrails": ["Do not overstate the finding."],
                "reference_answer_supplied_to_model": False,
            }],
        )
        _write_jsonl(
            chunks,
            [{"chunk_id": "chunk-1", "document_id": "doc-1", "text": "Evidence."}],
        )
        heldout.parent.mkdir(parents=True, exist_ok=True)
        heldout.write_text(
            json.dumps({
                "heldout_document_ids": ["doc-1"],
                "selection_digest": "selection-1",
            }),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "status": "frozen",
            "authoring_method": "independently_curated_from_frozen_corpus_evidence",
            "teacher_model_used_for_question_authoring": False,
            "reference_answers_supplied_to_evaluation_model": False,
            "benchmark_sha256": _digest(benchmark),
            "reference_sha256": _digest(reference),
            "heldout_manifest_sha256": _digest(heldout),
            "heldout_selection_digest": "selection-1",
            "heldout_document_ids": ["doc-1"],
            "case_count": 1,
            "supported_case_count": 1,
            "abstention_case_count": 0,
            "heldout_document_coverage": 1,
            "case_ids": ["case-1"],
        }
        (root / "benchmark/benchmark_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return Settings(root=root, data={"paths": {
            "benchmark_file": "benchmark/benchmark.jsonl",
            "benchmark_reference": "benchmark/benchmark_reference.jsonl",
            "benchmark_manifest": "benchmark/benchmark_manifest.json",
            "heldout_manifest": "artifacts/splits/heldout_manifest.json",
            "chunk_file": "artifacts/corpus/chunks.jsonl",
        }})

    def test_valid_freeze_covers_heldout_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate_benchmark_freeze(self._fixture(Path(directory)))
            self.assertTrue(result["valid"])
            self.assertEqual(result["heldout_document_coverage"], 1)

    def _enable_qwen_v17_turbo_mode(self, root: Path, settings: Settings) -> None:
        model = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
        method = "qwen_122b_obligation_mapped_from_frozen_heldout_evidence"
        routing = {
            "single_source": "GPT-OSS-120B",
            "multi_source": "Codex gold",
            "table": "Codex gold",
            "comparison": "Codex gold",
        }
        settings.data["v17_turbo_heldout_benchmark"] = {
            "authoring_model": model,
            "authoring_method": method,
            "single_source_reviewer": routing["single_source"],
            "multi_source_reviewer": routing["multi_source"],
            "table_reviewer": routing["table"],
            "comparison_reviewer": routing["comparison"],
        }
        benchmark = root / "benchmark/benchmark.jsonl"
        case = json.loads(benchmark.read_text(encoding="utf-8").strip())
        case.update({"candidate_type": "single_source", "question_authoring_model": model})
        _write_jsonl(benchmark, [case])
        reference = root / "benchmark/benchmark_reference.jsonl"
        rubric = json.loads(reference.read_text(encoding="utf-8").strip())
        rubric.update({
            "candidate_type": "single_source",
            "reviewer": "GPT-OSS-120B",
            "review_verdict": "complete",
            "reference_answer": "The supported finding.",
            "coverage_obligations": [{
                "obligation_id": "O1",
                "requirement": "Report the supported finding.",
                "supporting_chunk_ids": ["chunk-1"],
            }],
        })
        _write_jsonl(reference, [rubric])
        manifest_path = root / "benchmark/benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "authoring_method": method,
            "authoring_model": model,
            "teacher_model_used_for_question_authoring": True,
            "reviewer_routing": routing,
            "benchmark_sha256": _digest(benchmark),
            "reference_sha256": _digest(reference),
        })
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_qwen_authored_v17_turbo_freeze_enforces_reviewer_routing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._fixture(root)
            self._enable_qwen_v17_turbo_mode(root, settings)
            result = validate_benchmark_freeze(settings)
            self.assertEqual(result["authoring_mode"], "qwen_122b")
            reference = root / "benchmark/benchmark_reference.jsonl"
            rubric = json.loads(reference.read_text(encoding="utf-8").strip())
            rubric["reviewer"] = "Codex gold"
            _write_jsonl(reference, [rubric])
            manifest_path = root / "benchmark/benchmark_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reference_sha256"] = _digest(reference)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkFreezeError, "reviewer routing mismatch"):
                validate_benchmark_freeze(settings)

    def test_boolean_exception_flag_cannot_bypass_v17_turbo_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._fixture(root)
            self._enable_qwen_v17_turbo_mode(root, settings)
            manifest_path = root / "benchmark/benchmark_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["authoring_method"] = "independently_curated_from_frozen_corpus_evidence"
            manifest["teacher_model_used_for_question_authoring"] = False
            manifest["operator_authorized_authoring_exception"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkFreezeError, "authoring method differs"):
                validate_benchmark_freeze(settings)

    def test_question_mutation_invalidates_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._fixture(root)
            with (root / "benchmark/benchmark.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(BenchmarkFreezeError, "benchmark_sha256"):
                validate_benchmark_freeze(settings)

    def test_supporting_chunk_must_belong_to_expected_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._fixture(root)
            reference = root / "benchmark/benchmark_reference.jsonl"
            _write_jsonl(reference, [{
                "case_id": "case-1",
                "expected_document_ids": ["doc-1"],
                "supporting_chunk_ids": ["chunk-wrong"],
                "required_points": ["The supported finding."],
                "semantic_guardrails": ["Do not overstate the finding."],
                "reference_answer_supplied_to_model": False,
            }])
            manifest_path = root / "benchmark/benchmark_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reference_sha256"] = _digest(reference)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkFreezeError, "invalid supporting chunks"):
                validate_benchmark_freeze(settings)


if __name__ == "__main__":
    unittest.main()
