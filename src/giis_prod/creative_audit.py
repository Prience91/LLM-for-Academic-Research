from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .candidate_validation import validate_answer_style
from .citations import citation_coverage, contains_non_corpus_citation
from .config import Settings
from .dataset_release_ab import _load_release
from .io import sha256_file, write_json_atomic, write_jsonl_atomic
from .teacher import _teacher_schema
from .teacher_ab import running_teacher_profile

_QUESTION = re.compile(r"\nQuestion:\s*(.+?)\s*$", re.DOTALL)
_MARKER = re.compile(r"\[E(\d+)\]")

# Each source example already passed the v15-RC deterministic release policy. Its exact
# evidence and question are reused so this test isolates creative rendering rather than
# evidence selection or question generation.
_CASES = (
    {
        "case_id": "creative_01_table",
        "source_example_id": "sft_1669bf669d5bdb0b857f6024",
        "answer_style": "table",
        "instruction": "Answer as one compact Markdown comparison table.",
    },
    {
        "case_id": "creative_02_comparison",
        "source_example_id": "sft_664f77fb707b58253afab2a5",
        "answer_style": "comparison",
        "instruction": "Answer as an explicit point-by-point comparison.",
    },
    {
        "case_id": "creative_03_causal_chain",
        "source_example_id": "sft_c0819926307ee98664aea5ae",
        "answer_style": "causal_chain",
        "instruction": "Answer as a numbered causal chain with at least two evidence-cited steps.",
    },
    {
        "case_id": "creative_04_concept_diagram",
        "source_example_id": "sft_c1ed2a0d7d0f44f0c9c6b03c",
        "answer_style": "concept_diagram",
        "instruction": "Answer only as one safe Mermaid flowchart showing the supported process.",
    },
    {
        "case_id": "creative_05_concept_diagram",
        "source_example_id": "sft_3e4287ed0e1672ccb3f8b5cd",
        "answer_style": "concept_diagram",
        "instruction": "Answer only as one safe Mermaid flowchart showing the supported process.",
    },
)


def _source_parts(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages") or []
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Invalid source chat structure: {row.get('example_id')}")
    user = str(messages[1].get("content") or "")
    match = _QUESTION.search(user)
    if not match:
        raise ValueError(f"Missing source question: {row.get('example_id')}")
    evidence = user[: match.start()].removeprefix("Evidence:\n").strip()
    return evidence, match.group(1).strip()


def _request_options(
    settings: Settings,
    *,
    evidence: str,
    question: str,
    answer_style: str,
    instruction: str,
) -> dict[str, Any]:
    allowed = sorted({int(number) for number in _MARKER.findall(evidence)})
    allowed_markers = ", ".join(f"[E{number}]" for number in allowed)
    prompt = f"""Render one grounded scientific answer to the exact supplied question.

Rules:
- Copy the question exactly, character for character, into the question field.
- Set response_mode to standard and answer_style to {answer_style}.
- {instruction}
- Use only the supplied evidence; omit any claim the evidence cannot support.
- Cite every factual sentence, table data row, causal step, or Mermaid node with one or more
  exact evidence markers. The only allowed markers are: {allowed_markers}.
- Never reproduce a nested citation or attribute a claim to a source merely mentioned in the text.
- Stay below 500 words and do not expose hidden reasoning.
- For a table, provide a Markdown header, separator, and at least one evidence-cited data row.
- For a comparison, make the compared dimensions explicit and use at least two distinct evidence
  markers across the response.
- For a causal chain, provide at least two numbered or arrow-linked evidence-cited steps.
- For a concept_diagram, return exactly one Mermaid flowchart (TD, TB, LR, or RL), define 2-20
  nodes, put an evidence marker inside every node label, use at least one unlabelled arrow, and use
  no URL, click, HTML, comment, script, or styling directive.

EVIDENCE:
{evidence}

EXACT QUESTION:
{question}
"""
    options: dict[str, Any] = {}
    if "enable_thinking" in settings.section("teacher"):
        options["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": bool(settings.get("teacher.enable_thinking"))
            }
        }
    return {
        "model": settings.get("models.teacher"),
        "messages": [
            {
                "role": "system",
                "content": "You render strictly grounded scientific answers in the requested schema.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": _teacher_schema(exact_items=1),
        "temperature": float(settings.get("teacher.temperature")),
        "max_tokens": int(settings.get("teacher.max_output_tokens")),
        **options,
    }


def _deterministic_review(
    example: dict[str, Any], *, question: str, answer_style: str, evidence: str, minimum_coverage: float
) -> dict[str, Any]:
    answer = str(example.get("answer") or "").strip()
    issues: list[str] = []
    if str(example.get("question") or "").strip() != question:
        issues.append("question_changed")
    if str(example.get("response_mode") or "") != "standard":
        issues.append("response_mode_not_standard")
    if str(example.get("answer_style") or "") != answer_style:
        issues.append("answer_style_changed")
    style_valid, style_reason = validate_answer_style(example)
    if not style_valid:
        issues.append(style_reason)
    allowed = set(_MARKER.findall(evidence))
    used = set(_MARKER.findall(answer))
    if not used:
        issues.append("missing_evidence_markers")
    elif not used.issubset(allowed):
        issues.append("out_of_range_evidence_markers")
    if contains_non_corpus_citation(answer):
        issues.append("non_corpus_secondary_citation")
    coverage = citation_coverage(answer)
    if coverage < minimum_coverage:
        issues.append("insufficient_citation_coverage")
    word_count = len(answer.split())
    if word_count > 500:
        issues.append("standard_answer_too_long")
    if not answer:
        issues.append("empty_answer")
    return {
        "passed": not issues,
        "issues": sorted(set(issues)),
        "citation_coverage": round(coverage, 6),
        "word_count": word_count,
        "used_evidence_markers": sorted(f"E{number}" for number in used),
        "format_validator": style_reason,
        "semantic_status": "pending_independent_manual_review" if not issues else "not_queued",
    }


def run_creative_audit(
    settings: Settings,
    *,
    release_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from openai import OpenAI

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Creative audit already exists at {manifest_path}; choose a fresh output directory"
        )
    profile = running_teacher_profile(settings)
    expected_profile = {"max_model_len": 16384, "max_num_seqs": 2}
    actual_profile = {key: int(profile[key]) for key in expected_profile}
    if actual_profile != expected_profile:
        raise RuntimeError(
            f"Creative audit requires the selected 16K/2 teacher profile; found {actual_profile}"
        )
    api_key = os.getenv("TEACHER_API_KEY")
    if not api_key:
        raise RuntimeError("TEACHER_API_KEY is unavailable in this shell/tmux session")

    release_manifest, accepted, _rejected = _load_release(release_dir)
    by_id = {str(row.get("example_id")): row for row in accepted}
    missing = [case["source_example_id"] for case in _CASES if case["source_example_id"] not in by_id]
    if missing:
        raise RuntimeError(f"Creative audit source examples are missing: {missing}")
    client = OpenAI(
        api_key=api_key,
        base_url=str(settings.get("teacher.base_url")),
        timeout=float(settings.get("teacher.request_timeout_seconds")),
    )
    minimum_coverage = float(settings.get("teacher.minimum_training_citation_coverage"))

    def execute(case: dict[str, str]) -> dict[str, Any]:
        row = by_id[case["source_example_id"]]
        evidence, question = _source_parts(row)
        started = time.perf_counter()
        result: dict[str, Any] = {
            **case,
            "bundle_id": row.get("bundle_id"),
            "document_ids": row.get("document_ids"),
            "question": question,
            "evidence": evidence,
        }
        try:
            response = client.chat.completions.create(
                **_request_options(
                    settings,
                    evidence=evidence,
                    question=question,
                    answer_style=case["answer_style"],
                    instruction=case["instruction"],
                )
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Teacher returned empty structured output")
            payload = json.loads(content)
            examples = list(payload.get("examples") or [])
            if len(examples) != 1:
                raise ValueError("Teacher changed the requested creative-candidate count")
            example = examples[0]
            result["candidate"] = example
            result["deterministic_review"] = _deterministic_review(
                example,
                question=question,
                answer_style=case["answer_style"],
                evidence=evidence,
                minimum_coverage=minimum_coverage,
            )
            usage = getattr(response, "usage", None)
            result["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception as exc:  # preserve every case outcome for an auditable five-case run
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["deterministic_review"] = {
                "passed": False,
                "issues": ["generation_or_parse_failure"],
                "semantic_status": "not_queued",
            }
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(execute, case): case["case_id"] for case in _CASES}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["case_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    write_jsonl_atomic(results_path, results)
    semantic_queue = [
        row for row in results if bool(row["deterministic_review"].get("passed"))
    ]
    semantic_path = output_dir / "manual_semantic_queue.jsonl"
    write_jsonl_atomic(semantic_path, semantic_queue)
    passed = len(semantic_queue)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Independent five-case creative rendering audit; frozen A/B releases unchanged",
        "teacher_model": settings.get("models.teacher_source"),
        "teacher_profile": actual_profile,
        "generation_concurrency": 2,
        "source_release": str(release_dir),
        "source_release_manifest_status": release_manifest.get("status"),
        "source_release_manifest_sha256": sha256_file(release_dir / "dataset_manifest.json"),
        "case_count": len(results),
        "deterministic_passed": passed,
        "deterministic_failed": len(results) - passed,
        "all_deterministic_checks_passed": passed == len(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results_file": {"path": str(results_path), "sha256": sha256_file(results_path)},
        "manual_semantic_queue": {
            "path": str(semantic_path),
            "row_count": passed,
            "sha256": sha256_file(semantic_path),
        },
        "interpretation_guardrail": (
            "Python verifies structure, citation syntax/coverage, and safety only; semantic "
            "entailment remains pending independent manual evidence review."
        ),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest

