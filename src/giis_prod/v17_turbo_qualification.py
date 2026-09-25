"""Paired table/comparison qualification generation for V17-Turbo."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from .creative_audit import _deterministic_review, _request_options, _source_parts
from .io import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from .teacher_ab import running_teacher_profile
from .v17_turbo_contract import qualification_cases


def _passing_rows(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def _coverage_requirements(source: dict[str, Any]) -> str:
    obligations = source.get("candidate", {}).get("coverage_obligations") or []
    if not obligations:
        raise ValueError("Qualification source has no V17-Turbo coverage obligations")
    return "\n".join(
        f"- {str(row['obligation_id'])}: {str(row['requirement']).strip()}"
        for row in obligations
    )


def _qualified_request_options(
    settings: Any, *, source: dict[str, Any], answer_style: str
) -> tuple[dict[str, Any], str, str]:
    evidence, question = _source_parts({"messages": source["student_messages"]})
    instruction = (
        "Answer as one compact Markdown comparison table. Every substantive row must carry "
        "one or more evidence markers."
        if answer_style == "table"
        else "Answer as an explicit point-by-point comparison with evidence markers in every factual point."
    )
    options = _request_options(
        settings,
        evidence=evidence,
        question=question,
        answer_style=answer_style,
        instruction=instruction,
    )
    required = _coverage_requirements(source)
    insertion = (
        "\nRequired coverage inherited from the passing V17-Turbo prose source; address every item "
        "without adding unsupported requirements:\n" + required + "\n"
    )
    options["messages"][1]["content"] = options["messages"][1]["content"].replace(
        "\nEVIDENCE:\n", insertion + "\nEVIDENCE:\n", 1
    )
    return options, evidence, question


def run_v17_turbo_qualification(
    settings: Any,
    *,
    passing_candidates_dir: Path,
    output_dir: Path,
    plan_digest: str,
) -> dict[str, Any]:
    """Generate configured paired variants and freeze every deterministic pass for review."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_digest") != plan_digest:
            raise RuntimeError("Qualification manifest belongs to another V17-Turbo plan")
        return manifest

    configuration = settings.section("v17_turbo_structured_qualification")
    if not bool(configuration.get("enabled")):
        raise RuntimeError("V17-Turbo structured qualification lane is disabled")
    passing_rows = _passing_rows(passing_candidates_dir)
    mismatched = [
        str(row.get("example_id") or "unknown")
        for row in passing_rows
        if row.get("plan_digest") != plan_digest
    ]
    if mismatched:
        raise RuntimeError(
            "Qualification input contains candidates outside the active plan: "
            + ", ".join(sorted(mismatched)[:10])
        )
    cases = qualification_cases(
        passing_rows,
        seed=int(settings.get("teacher.seed")),
        bundle_count=int(configuration["source_bundles"]),
    )
    expected = int(configuration["planned_total_candidates"])
    if len(cases) != expected:
        raise RuntimeError(f"Expected {expected} qualification cases; found {len(cases)}")

    profile = running_teacher_profile(settings)
    expected_profile = {"max_model_len": 16384, "max_num_seqs": 2}
    actual_profile = {key: int(profile[key]) for key in expected_profile}
    if actual_profile != expected_profile:
        raise RuntimeError(
            f"V17-Turbo qualification requires the selected 16K/2 teacher profile; found {actual_profile}"
        )
    api_key = os.getenv("TEACHER_API_KEY")
    if not api_key:
        raise RuntimeError("TEACHER_API_KEY is unavailable in this shell/tmux session")
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=str(settings.get("teacher.base_url")),
        timeout=float(settings.get("teacher.request_timeout_seconds")),
    )
    retries = int(settings.get("teacher.maximum_transient_retries_per_request"))
    minimum_coverage = float(settings.get("teacher.minimum_training_citation_coverage"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "cases"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def execute(case: dict[str, Any]) -> dict[str, Any]:
        checkpoint = checkpoint_dir / f"{case['case_id']}.json"
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            expected_identity = {
                "case_id": case["case_id"],
                "answer_style": case["answer_style"],
                "source_example_id": case["source_example_id"],
                "bundle_id": case["bundle_id"],
                "document_ids": case["document_ids"],
                "plan_digest": plan_digest,
            }
            if all(cached.get(key) == value for key, value in expected_identity.items()):
                return cached
            checkpoint.unlink()
        source = case["source"]
        options, evidence, question = _qualified_request_options(
            settings, source=source, answer_style=str(case["answer_style"])
        )
        started = time.perf_counter()
        result: dict[str, Any] = {
            key: case[key]
            for key in ("case_id", "answer_style", "source_example_id", "bundle_id", "document_ids")
        }
        result.update(
            {
                "plan_digest": plan_digest,
                "question": question,
                "required_coverage_obligations": source["candidate"]["coverage_obligations"],
                "semantic_review": None,
                "training_eligible": False,
            }
        )
        for attempt in range(1, retries + 2):
            try:
                response = client.chat.completions.create(**options)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Teacher returned empty structured output")
                examples = list(json.loads(content).get("examples") or [])
                if len(examples) != 1:
                    raise ValueError("Teacher changed the requested qualification count")
                candidate = examples[0]
                result["candidate"] = candidate
                result["deterministic_review"] = _deterministic_review(
                    candidate,
                    question=question,
                    answer_style=str(case["answer_style"]),
                    evidence=evidence,
                    minimum_coverage=minimum_coverage,
                )
                usage = getattr(response, "usage", None)
                result["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                result["attempts"] = attempt
                break
            except Exception as exc:
                if attempt > retries:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    result["attempts"] = attempt
                    result["deterministic_review"] = {
                        "passed": False,
                        "issues": ["generation_or_parse_failure"],
                        "semantic_status": "not_queued",
                    }
                else:
                    time.sleep(float(settings.get("teacher.transient_retry_backoff_seconds")))
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json_atomic(checkpoint, result)
        return result

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    concurrency = int(settings.get("teacher.generation_concurrency"))
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute, case) for case in cases]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["case_id"]))
    results_path = output_dir / "results.jsonl"
    write_jsonl_atomic(results_path, results)
    semantic_queue = [row for row in results if row["deterministic_review"].get("passed")]
    semantic_path = output_dir / "manual_semantic_queue.jsonl"
    write_jsonl_atomic(semantic_path, semantic_queue)
    by_style = {
        style: sum(1 for row in results if row["answer_style"] == style)
        for style in ("table", "comparison")
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "plan_digest": plan_digest,
        "purpose": "V17-Turbo paired table/comparison qualification lane",
        "source_bundle_count": len({row["bundle_id"] for row in results}),
        "case_count": len(results),
        "cases_by_style": by_style,
        "deterministic_passed": len(semantic_queue),
        "deterministic_failed": len(results) - len(semantic_queue),
        "semantic_review_required": len(semantic_queue),
        "semantic_review_fraction": 1.0,
        "counts_toward_primary_candidate_minima": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results_file": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "manual_semantic_queue": {
            "path": str(semantic_path),
            "row_count": len(semantic_queue),
            "sha256": sha256_file(semantic_path),
        },
        "training_release": False,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
