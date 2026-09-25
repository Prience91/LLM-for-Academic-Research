from __future__ import annotations

import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .benchmark_freeze import validate_benchmark_freeze
from .citations import citation_coverage, validate_markers
from .config import Settings
from .generation import GroundedGenerator, _evidence_prompt
from .io import read_jsonl, write_json_atomic, write_jsonl_atomic
from .models import ResponseMode
from .policy import select_response_mode


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    query: str
    category: str
    expected_document_ids: list[str]
    expect_abstention: bool
    requested_mode: ResponseMode = ResponseMode.AUTO

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(row["case_id"]),
            query=str(row["query"]),
            category=str(row["category"]),
            expected_document_ids=list(row.get("expected_document_ids", [])),
            expect_abstention=bool(row.get("expect_abstention", False)),
            requested_mode=ResponseMode(row.get("requested_mode", "auto")),
        )


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class SystemEvaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.generator = GroundedGenerator(settings)

    def evaluate_case(self, case: BenchmarkCase) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.generator.answer(
            case.query,
            history=None,
            requested_mode=case.requested_mode,
        )
        latency = time.perf_counter() - started
        retrieved_documents = sorted({item.document_id for item in result.evidence})
        expected = set(case.expected_document_ids)
        recall = len(expected & set(retrieved_documents)) / len(expected) if expected else None
        markers_valid, _ = validate_markers(result.raw_marker_answer, result.evidence)
        coverage = citation_coverage(result.raw_marker_answer) if result.raw_marker_answer else 0.0
        requested = select_response_mode(case.query, case.requested_mode)
        long_form_compliant = (
            None
            if requested not in {ResponseMode.LONG_FORM, ResponseMode.SOURCE_SUMMARY}
            or case.expect_abstention
            else (
                not result.abstained
                and len(result.raw_marker_answer.split())
                >= int(self.settings.get("response.minimum_long_form_words"))
            )
        )
        return {
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "expect_abstention": case.expect_abstention,
            "actual_abstention": result.abstained,
            "abstention_reason": result.abstention_reason,
            "rejection_reason_codes": list(result.rejection_reason_codes),
            "abstention_correct": result.abstained == case.expect_abstention,
            "requested_mode": requested.value,
            "response_mode": result.mode.value,
            "answer_text": result.answer,
            "raw_marker_answer": result.raw_marker_answer,
            "references_markdown": result.references_markdown,
            "expected_document_ids": case.expected_document_ids,
            "retrieved_document_ids": retrieved_documents,
            "retrieved_chunk_ids": [item.chunk_id for item in result.evidence],
            "evidence": [asdict(item) for item in result.evidence],
            "retrieval_recall": recall,
            "raw_citation_markers_valid": (
                markers_valid if result.raw_marker_answer else None
            ),
            "citation_markers_valid": (
                None
                if case.expect_abstention
                else (not result.abstained and markers_valid)
            ),
            "citation_coverage": None if case.expect_abstention else coverage,
            "long_form_compliant": long_form_compliant,
            "word_count": len(result.raw_marker_answer.split()),
            "repair_attempts": result.repair_attempts,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "latency_seconds": latency,
            "rewritten_query": result.rewritten_query,
            "obligations": result.obligations,
            "obligation_coverage": result.obligation_coverage,
            "obligation_coverage_complete": result.coverage_complete,
            "atomic_planning_succeeded": result.atomic_planning_succeeded,
            "retrieval_passes": result.retrieval_passes,
        }

    def run(self, benchmark_path: Path, output_dir: Path) -> dict[str, Any]:
        configured_settings = getattr(self, "settings", None)
        configured_paths = (
            configured_settings.data.get("paths", {})
            if configured_settings is not None
            else {}
        )
        if "benchmark_manifest" in configured_paths:
            configured_benchmark = self.settings.path("paths.benchmark_file")
            if benchmark_path.resolve() == configured_benchmark.resolve():
                validate_benchmark_freeze(self.settings)
        cases = [BenchmarkCase.from_dict(row) for row in read_jsonl(benchmark_path)]
        if not cases:
            raise ValueError(f"Benchmark is empty: {benchmark_path}")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("Benchmark case_id values must be unique")
        try:
            heldout_path = self.settings.path("paths.heldout_manifest")
        except (AttributeError, KeyError):
            heldout_path = None
        if heldout_path is not None and heldout_path.exists():
            heldout_payload = json.loads(heldout_path.read_text(encoding="utf-8"))
            heldout_ids = {
                str(value) for value in heldout_payload.get("heldout_document_ids", [])
            }
            leaked = {
                document_id
                for case in cases
                if not case.expect_abstention
                for document_id in case.expected_document_ids
                if document_id not in heldout_ids
            }
            if leaked:
                raise ValueError(
                    "Supported benchmark cases reference documents outside the frozen held-out "
                    f"pool: {sorted(leaked)[:20]}"
                )
        details = [self.evaluate_case(case) for case in cases]
        supported = [row for row in details if not row["expect_abstention"]]
        recalls = [row["retrieval_recall"] for row in supported if row["retrieval_recall"] is not None]
        coverage = [row["citation_coverage"] for row in supported]
        long_form = [row for row in supported if row["long_form_compliant"] is not None]
        latencies = [row["latency_seconds"] for row in details]
        obligation_cases = [
            row
            for row in supported
            if row.get("obligation_coverage_complete") is not None
        ]
        obligation_rows = [
            obligation
            for row in obligation_cases
            for obligation in row.get("obligation_coverage", [])
        ]
        atomic_planning_cases = [
            row
            for row in supported
            if row.get("atomic_planning_succeeded") is not None
        ]
        summary = {
            "case_count": len(details),
            "abstention_accuracy": statistics.fmean(row["abstention_correct"] for row in details),
            "mean_retrieval_recall": statistics.fmean(recalls) if recalls else 0.0,
            "valid_citation_rate": (
                statistics.fmean(bool(row["citation_markers_valid"]) for row in supported)
                if supported
                else 0.0
            ),
            "mean_citation_coverage": statistics.fmean(coverage) if coverage else 0.0,
            "long_form_compliance": (
                statistics.fmean(bool(row["long_form_compliant"]) for row in long_form)
                if long_form
                else None
            ),
            "p50_latency_seconds": percentile(latencies, 0.50),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "obligation_case_coverage_rate": (
                statistics.fmean(
                    bool(row["obligation_coverage_complete"]) for row in obligation_cases
                )
                if obligation_cases
                else None
            ),
            "obligation_coverage_rate": (
                statistics.fmean(bool(row["covered"]) for row in obligation_rows)
                if obligation_rows
                else None
            ),
            "atomic_planning_success_rate": (
                statistics.fmean(
                    bool(row["atomic_planning_succeeded"])
                    for row in atomic_planning_cases
                )
                if atomic_planning_cases
                else None
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(output_dir / "details.jsonl", details)
        write_json_atomic(output_dir / "summary.json", summary)
        return summary

    def streaming_probe(self, case: BenchmarkCase) -> dict[str, float | int | str]:
        from transformers import AutoTokenizer

        retrieval = self.generator.retriever.retrieve(case.query)
        if retrieval.abstained:
            raise RuntimeError("Streaming probe case produced no corpus evidence")
        mode = select_response_mode(case.query, case.requested_mode)
        started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        stream = self.generator.client.chat.completions.create(
            model=self.generator.model,
            messages=[
                {"role": "system", "content": self.generator._system_prompt(mode)},
                {
                    "role": "user",
                    "content": f"Evidence:\n{_evidence_prompt(retrieval.evidence)}\n\nQuestion: {case.query}",
                },
            ],
            temperature=0.0,
            max_tokens=int(self.settings.get("response.long_form_max_tokens")),
            stream=True,
            **self.generator._chat_template_options(),
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(content)
        finished = time.perf_counter()
        if first_token_at is None:
            raise RuntimeError("Streaming probe returned no output tokens")
        tokenizer = AutoTokenizer.from_pretrained(self.settings.get("models.base"))
        completion_tokens = len(tokenizer.encode("".join(pieces), add_special_tokens=False))
        decode_seconds = max(finished - first_token_at, 1e-9)
        return {
            "case_id": case.case_id,
            "ttft_seconds": first_token_at - started,
            "decode_seconds": decode_seconds,
            "completion_tokens": completion_tokens,
            "decode_tokens_per_second": completion_tokens / decode_seconds,
            "total_seconds": finished - started,
        }


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    mapping = {
        "abstention_accuracy": "abstention_correct",
        "mean_retrieval_recall": "retrieval_recall",
        "valid_citation_rate": "citation_markers_valid",
        "mean_citation_coverage": "citation_coverage",
        "long_form_compliance": "long_form_compliant",
    }
    value = row.get(mapping[metric])
    return None if value is None else float(value)


def _paired_bootstrap_bounds(
    differences: list[float],
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not differences:
        raise ValueError("Cannot compare a metric without paired benchmark observations")
    point = statistics.fmean(differences)
    if len(differences) == 1:
        return point, point, point
    randomizer = random.Random(seed)
    means = sorted(
        statistics.fmean(randomizer.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    alpha = 1.0 - confidence
    one_sided_lower = percentile(means, alpha)
    central_lower = percentile(means, alpha / 2.0)
    central_upper = percentile(means, 1.0 - alpha / 2.0)
    return one_sided_lower, central_lower, central_upper


def compare_runs(
    settings: Settings, baseline_dir: Path, candidate_dir: Path
) -> dict[str, Any]:
    baseline_path = baseline_dir / "summary.json"
    candidate_path = candidate_dir / "summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_rows = {row["case_id"]: row for row in read_jsonl(baseline_dir / "details.jsonl")}
    candidate_rows = {row["case_id"]: row for row in read_jsonl(candidate_dir / "details.jsonl")}
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Baseline and candidate case_id sets differ")
    maximum_regression = float(settings.get("quality_gates.maximum_material_metric_regression"))
    confidence = float(settings.get("quality_gates.confidence_level"))
    bootstrap_samples = int(settings.get("quality_gates.bootstrap_samples"))
    seed = int(settings.get("quality_gates.bootstrap_seed"))
    minimum_paired_cases = int(settings.get("quality_gates.minimum_paired_cases_per_metric"))
    quality_metrics = [
        "abstention_accuracy",
        "mean_retrieval_recall",
        "valid_citation_rate",
        "mean_citation_coverage",
        "long_form_compliance",
    ]
    comparisons: dict[str, dict[str, float | int | bool]] = {}
    failures: list[str] = []
    for metric_index, metric in enumerate(quality_metrics):
        paired_differences = []
        for case_id in sorted(baseline_rows):
            baseline_value = _metric_value(baseline_rows[case_id], metric)
            candidate_value = _metric_value(candidate_rows[case_id], metric)
            if baseline_value is not None and candidate_value is not None:
                paired_differences.append(candidate_value - baseline_value)
        lower, central_lower, central_upper = _paired_bootstrap_bounds(
            paired_differences,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed + metric_index,
        )
        point = statistics.fmean(paired_differences)
        enough_cases = len(paired_differences) >= minimum_paired_cases
        noninferior = enough_cases and lower >= -maximum_regression
        comparisons[metric] = {
            "paired_cases": len(paired_differences),
            "minimum_paired_cases": minimum_paired_cases,
            "point_difference": point,
            "one_sided_lower_bound": lower,
            "central_interval_lower": central_lower,
            "central_interval_upper": central_upper,
            "noninferior": noninferior,
        }
        if not enough_cases:
            failures.append(
                f"{metric} has {len(paired_differences)} paired cases; "
                f"at least {minimum_paired_cases} are required"
            )
        elif not noninferior:
            failures.append(
                f"{metric} lower confidence bound {lower:.4f} is below "
                f"the {-maximum_regression:.4f} non-inferiority boundary"
            )
    latency_difference = float(candidate["p95_latency_seconds"]) - float(
        baseline["p95_latency_seconds"]
    )
    if latency_difference > 0:
        failures.append(f"candidate p95 latency is {latency_difference:.4f}s slower")
    return {
        "passed": not failures,
        "maximum_allowed_regression": maximum_regression,
        "confidence_level": confidence,
        "bootstrap_samples": bootstrap_samples,
        "minimum_paired_cases_per_metric": minimum_paired_cases,
        "quality_comparisons": comparisons,
        "p95_latency_difference_seconds": latency_difference,
        "failures": failures,
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
    }


def concurrent_probe(
    evaluator: SystemEvaluator,
    case: BenchmarkCase,
    users: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=users) as executor:
        futures = [executor.submit(evaluator.evaluate_case, case) for _ in range(users)]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    return {
        "users": users,
        "wall_seconds": elapsed,
        "requests_per_second": users / elapsed,
        "p95_request_latency_seconds": percentile(
            [row["latency_seconds"] for row in results], 0.95
        ),
        "all_non_abstaining": all(not row["actual_abstention"] for row in results),
    }
