from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import Settings
from .io import read_jsonl


class BenchmarkFreezeError(ValueError):
    """Raised when a frozen held-out benchmark is missing, stale, or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_file(settings: Settings, key: str) -> Path:
    path = settings.path(key)
    if not path.is_file():
        raise BenchmarkFreezeError(f"Required frozen-benchmark file is missing: {path}")
    return path


def validate_benchmark_freeze(settings: Settings) -> dict[str, Any]:
    """Validate the immutable benchmark, evidence rubric, and held-out binding."""
    benchmark_path = _required_file(settings, "paths.benchmark_file")
    reference_path = _required_file(settings, "paths.benchmark_reference")
    manifest_path = _required_file(settings, "paths.benchmark_manifest")
    heldout_path = _required_file(settings, "paths.heldout_manifest")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "frozen":
        raise BenchmarkFreezeError("Benchmark manifest must use schema_version 1 and status=frozen")
    turbo_policy = settings.data.get("v17_turbo_heldout_benchmark")
    exception = manifest.get("one_time_authoring_exception")
    operator_exception = isinstance(exception, dict)
    if operator_exception:
        if exception.get("exception_id") != "current_canary_codex_complete_three_v1":
            raise BenchmarkFreezeError("Unrecognized one-time held-out authoring exception")
        if exception.get("author") != "Codex gold" or exception.get("reviewer") != "Codex gold":
            raise BenchmarkFreezeError("One-time exception must bind Codex authoring and review")
        if exception.get("policy_precedent") is not False:
            raise BenchmarkFreezeError("One-time exception cannot establish policy precedent")
        if exception.get("future_v17_turbo_authoring_policy_unchanged") is not True:
            raise BenchmarkFreezeError("One-time exception must preserve future V17-Turbo policy")
    qwen_authored = False
    if turbo_policy and not operator_exception:
        qwen_authored = True
        if manifest.get("authoring_method") != turbo_policy.get("authoring_method"):
            raise BenchmarkFreezeError("Benchmark authoring method differs from V17-Turbo policy")
        if manifest.get("authoring_model") != turbo_policy.get("authoring_model"):
            raise BenchmarkFreezeError("Benchmark authoring model differs from V17-Turbo policy")
        if not bool(manifest.get("teacher_model_used_for_question_authoring", False)):
            raise BenchmarkFreezeError("V17-Turbo benchmark must declare Qwen teacher authorship")
        expected_routing = {
            "single_source": turbo_policy.get("single_source_reviewer"),
            "multi_source": turbo_policy.get("multi_source_reviewer"),
            "table": turbo_policy.get("table_reviewer"),
            "comparison": turbo_policy.get("comparison_reviewer"),
        }
        if manifest.get("reviewer_routing") != expected_routing:
            raise BenchmarkFreezeError("Benchmark reviewer routing differs from V17-Turbo policy")
    else:
        if manifest.get("authoring_method") != "independently_curated_from_frozen_corpus_evidence":
            raise BenchmarkFreezeError("Benchmark must declare independent evidence-based authoring")
        if bool(manifest.get("teacher_model_used_for_question_authoring", True)):
            raise BenchmarkFreezeError("Independent benchmark cannot declare teacher question authorship")
    if bool(manifest.get("reference_answers_supplied_to_evaluation_model", True)):
        raise BenchmarkFreezeError("The hidden evidence rubric cannot be supplied to the model")

    expected_hashes = {
        "benchmark_sha256": _sha256_file(benchmark_path),
        "reference_sha256": _sha256_file(reference_path),
        "heldout_manifest_sha256": _sha256_file(heldout_path),
    }
    for field, actual in expected_hashes.items():
        if str(manifest.get(field) or "") != actual:
            raise BenchmarkFreezeError(f"Frozen benchmark digest mismatch: {field}")

    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    heldout_ids = {str(value) for value in heldout.get("heldout_document_ids", [])}
    if not heldout_ids:
        raise BenchmarkFreezeError("Held-out manifest contains no benchmark document IDs")
    if str(manifest.get("heldout_selection_digest") or "") != str(
        heldout.get("selection_digest") or ""
    ):
        raise BenchmarkFreezeError("Benchmark freeze is bound to the wrong held-out selection")
    if set(map(str, manifest.get("heldout_document_ids", []))) != heldout_ids:
        raise BenchmarkFreezeError("Benchmark manifest held-out IDs do not match the frozen split")
    exception_authorized: set[str] = set()
    exception_legacy: set[str] = set()
    if operator_exception:
        exception_authorized = {str(value) for value in exception.get("authorized_document_ids", [])}
        exception_legacy = {str(value) for value in exception.get("preserved_legacy_document_ids", [])}
        if len(exception_authorized) != 3 or len(exception_legacy) != 5:
            raise BenchmarkFreezeError("One-time exception must bind exactly three new and five legacy documents")
        if exception_authorized & exception_legacy or exception_authorized | exception_legacy != heldout_ids:
            raise BenchmarkFreezeError("One-time exception document scope differs from the held-out split")

    cases = list(read_jsonl(benchmark_path))
    references = list(read_jsonl(reference_path))
    case_ids = [str(row.get("case_id") or "") for row in cases]
    if not cases or any(not value for value in case_ids) or len(case_ids) != len(set(case_ids)):
        raise BenchmarkFreezeError("Benchmark case IDs must be non-empty and unique")
    if int(manifest.get("case_count", -1)) != len(cases):
        raise BenchmarkFreezeError("Benchmark case count differs from its frozen manifest")
    if list(map(str, manifest.get("case_ids", []))) != case_ids:
        raise BenchmarkFreezeError("Benchmark case order or membership differs from its manifest")

    supported = [row for row in cases if not bool(row.get("expect_abstention", False))]
    abstention = [row for row in cases if bool(row.get("expect_abstention", False))]
    if int(manifest.get("supported_case_count", -1)) != len(supported):
        raise BenchmarkFreezeError("Supported benchmark count differs from its manifest")
    if int(manifest.get("abstention_case_count", -1)) != len(abstention):
        raise BenchmarkFreezeError("Abstention benchmark count differs from its manifest")

    covered: list[str] = []
    allowed_types = {"single_source", "multi_source", "table", "comparison"}
    for row in supported:
        expected = [str(value) for value in row.get("expected_document_ids", [])]
        if qwen_authored:
            candidate_type = str(row.get("candidate_type") or "")
            if candidate_type not in allowed_types:
                raise BenchmarkFreezeError("Qwen-authored benchmark case has invalid candidate type")
            required_sources = 1 if candidate_type == "single_source" else 2
            if len(expected) < required_sources or len(expected) != len(set(expected)):
                raise BenchmarkFreezeError("Held-out case has an invalid source count")
            if not set(expected).issubset(heldout_ids):
                raise BenchmarkFreezeError("Held-out case references a non-held-out document")
            if row.get("question_authoring_model") != turbo_policy.get("authoring_model"):
                raise BenchmarkFreezeError("Held-out case does not bind the Qwen authoring model")
        elif len(expected) != 1 or expected[0] not in heldout_ids:
            raise BenchmarkFreezeError(
                "Each independently curated case must target exactly one frozen held-out document"
            )
        elif operator_exception:
            document_id = expected[0]
            if document_id in exception_authorized:
                if row.get("question_authoring_model") != "Codex gold":
                    raise BenchmarkFreezeError("Exception-authored case does not bind Codex gold")
                if row.get("authoring_exception_id") != exception.get("exception_id"):
                    raise BenchmarkFreezeError("Exception-authored case has the wrong exception ID")
            elif document_id in exception_legacy:
                provenance = str(row.get("question_authoring_provenance") or "")
                if not provenance.startswith("legacy_five_preserved"):
                    raise BenchmarkFreezeError("Preserved legacy case lacks provenance")
        covered.extend(expected)
    if qwen_authored:
        if set(covered) != heldout_ids:
            raise BenchmarkFreezeError("Qwen-authored cases must cover every held-out document")
    elif len(covered) != len(set(covered)) or set(covered) != heldout_ids:
        raise BenchmarkFreezeError(
            "Supported benchmark cases must cover every held-out document exactly once"
        )
    if int(manifest.get("heldout_document_coverage", -1)) != len(set(covered)):
        raise BenchmarkFreezeError("Held-out coverage count differs from its manifest")

    reference_by_id = {str(row.get("case_id") or ""): row for row in references}
    if set(reference_by_id) != set(case_ids) or len(references) != len(reference_by_id):
        raise BenchmarkFreezeError("Evidence-rubric cases do not match the benchmark cases")

    chunk_path = _required_file(settings, "paths.chunk_file")
    chunk_documents: dict[str, str] = {}
    for row in read_jsonl(chunk_path):
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id:
            chunk_documents[chunk_id] = str(row.get("document_id") or "")
    for case in supported:
        case_id = str(case["case_id"])
        expected_documents = [str(value) for value in case.get("expected_document_ids", [])]
        rubric = reference_by_id[case_id]
        if list(map(str, rubric.get("expected_document_ids", []))) != expected_documents:
            raise BenchmarkFreezeError(f"Evidence rubric targets the wrong document: {case_id}")
        if bool(rubric.get("reference_answer_supplied_to_model", True)):
            raise BenchmarkFreezeError(f"Reference-answer isolation is not declared: {case_id}")
        supporting = [str(value) for value in rubric.get("supporting_chunk_ids", [])]
        supporting_documents = {chunk_documents.get(value) for value in supporting}
        if not supporting or None in supporting_documents or not supporting_documents.issubset(
            set(expected_documents)
        ):
            raise BenchmarkFreezeError(f"Evidence rubric contains invalid supporting chunks: {case_id}")
        if qwen_authored and supporting_documents != set(expected_documents):
            raise BenchmarkFreezeError(f"Evidence rubric does not support every source: {case_id}")
        if not rubric.get("required_points") or not rubric.get("semantic_guardrails"):
            raise BenchmarkFreezeError(f"Evidence rubric is incomplete: {case_id}")
        if qwen_authored:
            candidate_type = str(case.get("candidate_type"))
            expected_reviewer = manifest["reviewer_routing"][candidate_type]
            if str(rubric.get("candidate_type")) != candidate_type:
                raise BenchmarkFreezeError(f"Reference candidate type differs from case: {case_id}")
            if str(rubric.get("reviewer")) != expected_reviewer:
                raise BenchmarkFreezeError(f"Held-out reviewer routing mismatch: {case_id}")
            if str(rubric.get("review_verdict")) != "complete":
                raise BenchmarkFreezeError(f"Held-out case is not semantically complete: {case_id}")
            if not str(rubric.get("reference_answer") or "").strip():
                raise BenchmarkFreezeError(f"Held-out reference answer is absent: {case_id}")
            if not rubric.get("coverage_obligations"):
                raise BenchmarkFreezeError(f"Held-out obligation rubric is absent: {case_id}")
        elif operator_exception:
            if str(case.get("candidate_type")) != "single_source":
                raise BenchmarkFreezeError(f"Current exception only permits single-source cases: {case_id}")
            if str(rubric.get("candidate_type")) != "single_source":
                raise BenchmarkFreezeError(f"Exception reference candidate type differs from case: {case_id}")
            if str(rubric.get("reviewer")) != "Codex gold" or str(rubric.get("review_verdict")) != "complete":
                raise BenchmarkFreezeError(f"Exception case lacks Codex complete review: {case_id}")
            if not str(rubric.get("reference_answer") or "").strip():
                raise BenchmarkFreezeError(f"Exception reference answer is absent: {case_id}")
            obligations = rubric.get("coverage_obligations")
            if not isinstance(obligations, list) or not obligations:
                raise BenchmarkFreezeError(f"Exception obligation rubric is absent: {case_id}")
            if any(not item.get("requirement") or not item.get("supporting_chunk_ids") for item in obligations):
                raise BenchmarkFreezeError(f"Exception obligation rubric is incomplete: {case_id}")

    return {
        "valid": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "benchmark_sha256": expected_hashes["benchmark_sha256"],
        "reference_sha256": expected_hashes["reference_sha256"],
        "heldout_manifest_sha256": expected_hashes["heldout_manifest_sha256"],
        "case_count": len(cases),
        "supported_case_count": len(supported),
        "abstention_case_count": len(abstention),
        "heldout_document_coverage": len(set(covered)),
        "authoring_mode": (
            "qwen_122b" if qwen_authored else "codex_one_time_exception" if operator_exception else "independent"
        ),
    }
