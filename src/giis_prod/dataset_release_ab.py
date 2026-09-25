from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from .citations import citation_coverage, contains_non_corpus_citation
from .io import content_version, read_jsonl, sha256_file, stable_digest, write_json_atomic, write_jsonl_atomic
from .teacher import _question_similarity, validate_answer_style

_EVIDENCE = re.compile(
    r"(?m)^\[E(\d+)\]\s*\nDocument ID:\s*([^\n]+)"
)
_ANSWER_MARKER = re.compile(r"\[E(\d+)\]")
_QUESTION = re.compile(r"\nQuestion:\s*(.+?)\s*$", re.DOTALL)


def _load_release(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = directory / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Dataset release is not complete: {manifest_path}")
    accepted: list[dict[str, Any]] = []
    for name in ("train.jsonl", "validation.jsonl"):
        path = directory / name
        expected = manifest["files"][name]
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Immutable release hash mismatch: {path}")
        rows = list(read_jsonl(path))
        if len(rows) != int(expected["row_count"]):
            raise RuntimeError(f"Immutable release row-count mismatch: {path}")
        accepted.extend(rows)
    rejected_path = directory / "rejected.jsonl"
    expected_rejected = manifest["files"]["rejected.jsonl"]
    if sha256_file(rejected_path) != expected_rejected["sha256"]:
        raise RuntimeError(f"Immutable release hash mismatch: {rejected_path}")
    rejected = list(read_jsonl(rejected_path))
    if len(rejected) != int(expected_rejected["row_count"]):
        raise RuntimeError(f"Immutable release row-count mismatch: {rejected_path}")
    return manifest, accepted, rejected


def _record_parts(row: dict[str, Any]) -> tuple[str, str, str, dict[str, str]]:
    messages = row.get("messages") or []
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        return "", "", "", {}
    user = str(messages[1].get("content") or "")
    match = _QUESTION.search(user)
    question = match.group(1).strip() if match else ""
    evidence = user[: match.start()].strip() if match else user
    answer = str(messages[2].get("content") or "").strip()
    marker_documents = {
        number: document_id.strip()
        for number, document_id in _EVIDENCE.findall(evidence)
    }
    return question, answer, evidence, marker_documents


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def _deterministic_arm(
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    minimum_coverage: float,
    minimum_long_form_words: int,
    maximum_tokens: int,
    maximum_question_similarity: float,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    answer_words: list[int] = []
    question_words: list[int] = []
    normalized_questions: dict[str, list[str]] = defaultdict(list)
    questions_by_bundle: dict[str, list[tuple[str, str]]] = defaultdict(list)
    document_yield: Counter[str] = Counter()

    for row in rows:
        issues: list[str] = []
        question, answer, _evidence, marker_documents = _record_parts(row)
        bundle_id = str(row.get("bundle_id") or "")
        document_ids = [str(value) for value in (row.get("document_ids") or [])]
        if not question or not answer or not marker_documents:
            issues.append("invalid_chat_or_evidence_structure")
        used = set(_ANSWER_MARKER.findall(answer))
        allowed = set(marker_documents)
        if not used:
            issues.append("missing_evidence_markers")
        elif not used.issubset(allowed):
            issues.append("out_of_range_evidence_markers")
        cited_documents = {marker_documents[number] for number in used if number in marker_documents}
        if not set(document_ids).issubset(set(marker_documents.values())):
            issues.append("record_documents_do_not_match_evidence")
        kind = str(row.get("dataset_kind") or "")
        if kind == "core_single_source" and len(document_ids) != 1:
            issues.append("invalid_single_source_cardinality")
        if kind == "multi_source_extra" and (
            len(document_ids) < 2 or not set(document_ids).issubset(cited_documents)
        ):
            issues.append("multi_source_does_not_cite_every_source")
        coverage = citation_coverage(answer)
        coverage_values.append(coverage)
        if coverage < minimum_coverage:
            issues.append("insufficient_citation_coverage")
        if contains_non_corpus_citation(answer):
            issues.append("non_corpus_secondary_citation")
        style_valid, style_reason = validate_answer_style(
            {"answer": answer, "answer_style": row.get("answer_style", "prose")}
        )
        if not style_valid:
            issues.append(style_reason)
        words = len(answer.split())
        answer_words.append(words)
        question_words.append(len(question.split()))
        mode = str(row.get("response_mode") or "")
        if mode not in {"standard", "long_form"}:
            issues.append("invalid_response_mode")
        if mode == "standard" and words > 500:
            issues.append("standard_answer_too_long")
        if mode == "long_form" and words < minimum_long_form_words:
            issues.append("long_form_too_short")
        token_count = int(row.get("token_count") or 0)
        if token_count < 1 or token_count > maximum_tokens:
            issues.append("invalid_training_token_count")
        expected_id = "sft_" + stable_digest(bundle_id, question, answer)
        if row.get("example_id") != expected_id:
            issues.append("example_id_mismatch")
        normalized = re.sub(r"\s+", " ", question.casefold()).strip()
        if normalized:
            normalized_questions[normalized].append(str(row.get("example_id") or ""))
            questions_by_bundle[bundle_id].append((str(row.get("example_id") or ""), question))
        for document_id in document_ids:
            document_yield[document_id] += 1
        if issues:
            failures.append(
                {
                    "example_id": row.get("example_id"),
                    "bundle_id": bundle_id,
                    "issues": sorted(set(issues)),
                }
            )

    exact_duplicates = {
        question: ids for question, ids in normalized_questions.items() if len(ids) > 1
    }
    lexical_duplicates: list[dict[str, Any]] = []
    for bundle_id, candidates in sorted(questions_by_bundle.items()):
        for index, (left_id, left) in enumerate(candidates):
            for right_id, right in candidates[index + 1 :]:
                score = _question_similarity(left, right)
                if score >= maximum_question_similarity:
                    lexical_duplicates.append(
                        {
                            "bundle_id": bundle_id,
                            "left_example_id": left_id,
                            "right_example_id": right_id,
                            "similarity": round(score, 6),
                        }
                    )

    def stats(values: list[float | int]) -> dict[str, float]:
        return {
            "minimum": round(float(min(values)), 6),
            "mean": round(float(mean(values)), 6),
            "median": round(float(median(values)), 6),
            "maximum": round(float(max(values)), 6),
        }

    reason_counts = Counter(str(row.get("reason") or "unknown") for row in rejected)
    issue_counts = Counter(issue for row in failures for issue in row["issues"])
    return {
        "accepted_candidates": len(rows),
        "rejected_candidates": len(rejected),
        "candidate_yield_rate": round(len(rows) / (len(rows) + len(rejected)), 6),
        "full_reaudit_passed": len(rows) - len(failures),
        "full_reaudit_failed": len(failures),
        "full_reaudit_pass_rate": round((len(rows) - len(failures)) / len(rows), 6),
        "reaudit_issue_counts": dict(sorted(issue_counts.items())),
        "reaudit_failures": failures,
        "rejection_counts": dict(sorted(reason_counts.items())),
        "dataset_kind": _distribution(rows, "dataset_kind"),
        "split": _distribution(rows, "split"),
        "response_mode": _distribution(rows, "response_mode"),
        "answer_style": _distribution(rows, "answer_style"),
        "citation_coverage": stats(coverage_values),
        "answer_word_count": stats(answer_words),
        "question_word_count": stats(question_words),
        "exact_duplicate_question_groups": exact_duplicates,
        "within_bundle_lexical_duplicate_pairs": lexical_duplicates,
        "document_yield": dict(sorted(document_yield.items())),
    }


def _choose_pair(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], rng: random.Random
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[int, float, float, dict[str, Any], dict[str, Any]]] = []
    for left in left_rows:
        left_question, *_ = _record_parts(left)
        for right in right_rows:
            right_question, *_ = _record_parts(right)
            exact_stratum = int(
                left.get("response_mode") == right.get("response_mode")
                and left.get("answer_style", "prose") == right.get("answer_style", "prose")
            )
            candidates.append(
                (
                    exact_stratum,
                    _question_similarity(left_question, right_question),
                    rng.random(),
                    left,
                    right,
                )
            )
    _, _, _, left, right = max(candidates, key=lambda item: item[:3])
    return left, right


def _paired_sample(
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    *,
    single_count: int,
    multi_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_arm: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    for arm, rows in (("baseline", baseline), ("challenger", challenger)):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            mode = "multi" if row.get("dataset_kind") == "multi_source_extra" else "single"
            grouped[(mode, str(row["bundle_id"]))].append(row)
        by_arm[arm] = grouped

    queue: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    selected_bundle_ids: dict[str, list[str]] = {}
    for mode, count in (("single", single_count), ("multi", multi_count)):
        common = sorted(
            key[1]
            for key in set(by_arm["baseline"]) & set(by_arm["challenger"])
            if key[0] == mode
        )
        if len(common) < count:
            raise RuntimeError(f"Need {count} common {mode} bundles but found {len(common)}")
        rng.shuffle(common)
        selected = common[:count]
        selected_bundle_ids[mode] = selected
        for bundle_id in selected:
            baseline_row, challenger_row = _choose_pair(
                by_arm["baseline"][(mode, bundle_id)],
                by_arm["challenger"][(mode, bundle_id)],
                rng,
            )
            baseline_question, baseline_answer, baseline_evidence, _ = _record_parts(baseline_row)
            challenger_question, challenger_answer, challenger_evidence, _ = _record_parts(challenger_row)
            if baseline_evidence != challenger_evidence:
                raise RuntimeError(f"Evidence changed between arms for bundle {bundle_id}")
            swap = bool(rng.getrandbits(1))
            arm_a, row_a, question_a, answer_a = (
                ("challenger", challenger_row, challenger_question, challenger_answer)
                if swap
                else ("baseline", baseline_row, baseline_question, baseline_answer)
            )
            arm_b, row_b, question_b, answer_b = (
                ("baseline", baseline_row, baseline_question, baseline_answer)
                if swap
                else ("challenger", challenger_row, challenger_question, challenger_answer)
            )
            pair_id = f"semantic_pair_{len(queue) + 1:03d}"
            queue.append(
                {
                    "pair_id": pair_id,
                    "source_mode": mode,
                    "bundle_id": bundle_id,
                    "document_ids": baseline_row.get("document_ids"),
                    "evidence": baseline_evidence,
                    "response_a": {"question": question_a, "answer": answer_a},
                    "response_b": {"question": question_b, "answer": answer_b},
                    "audit": {
                        "response_a": None,
                        "response_b": None,
                        "preference": None,
                        "notes": None,
                    },
                }
            )
            key_rows.append(
                {
                    "pair_id": pair_id,
                    "response_a_arm": arm_a,
                    "response_b_arm": arm_b,
                    "response_a_example_id": row_a.get("example_id"),
                    "response_b_example_id": row_b.get("example_id"),
                }
            )
    rng.shuffle(queue)
    return queue, key_rows, selected_bundle_ids


def build_comparison(
    root: Path,
    baseline_dir: Path,
    challenger_dir: Path,
    output_dir: Path,
    *,
    single_count: int = 25,
    multi_count: int = 10,
    seed: int = 3407,
) -> dict[str, Any]:
    from .config import Settings

    settings = Settings.load(root / "config" / "settings.toml")
    baseline_manifest, baseline, baseline_rejected = _load_release(baseline_dir)
    challenger_manifest, challenger, challenger_rejected = _load_release(challenger_dir)
    left_digest = baseline_manifest["build_metadata"]["training_selection_digest"]
    right_digest = challenger_manifest["build_metadata"]["training_selection_digest"]
    if left_digest != right_digest:
        raise RuntimeError("A/B releases use different training document selections")
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_selection_digest": left_digest,
        "baseline": {
            "directory": str(baseline_dir.resolve()),
            "manifest_sha256": sha256_file(baseline_dir / "dataset_manifest.json"),
            "files": baseline_manifest["files"],
        },
        "challenger": {
            "directory": str(challenger_dir.resolve()),
            "manifest_sha256": sha256_file(challenger_dir / "dataset_manifest.json"),
            "files": challenger_manifest["files"],
        },
    }
    freeze["freeze_digest"] = content_version([freeze])
    write_json_atomic(output_dir / "freeze_manifest.json", freeze)

    minimum_coverage = float(settings.get("teacher.minimum_training_citation_coverage"))
    minimum_long_form_words = int(settings.get("response.minimum_long_form_words"))
    maximum_tokens = int(settings.get("training.max_length"))
    maximum_similarity = float(settings.get("teacher.maximum_question_similarity"))
    baseline_metrics = _deterministic_arm(
        baseline,
        baseline_rejected,
        minimum_coverage=minimum_coverage,
        minimum_long_form_words=minimum_long_form_words,
        maximum_tokens=maximum_tokens,
        maximum_question_similarity=maximum_similarity,
    )
    challenger_metrics = _deterministic_arm(
        challenger,
        challenger_rejected,
        minimum_coverage=minimum_coverage,
        minimum_long_form_words=minimum_long_form_words,
        maximum_tokens=maximum_tokens,
        maximum_question_similarity=maximum_similarity,
    )
    baseline_seconds = float(baseline_manifest["build_metadata"]["dataset_build_seconds"])
    # The challenger required a checkpointed recovery. Include both active process durations.
    challenger_seconds = float(challenger_manifest["build_metadata"]["dataset_build_seconds"])
    prior_failed_seconds = 0.0
    audit_dir = challenger_dir / "audit"
    for journal in sorted(audit_dir.glob("dataset_audit_*.jsonl")):
        for event in read_jsonl(journal):
            if event.get("event") == "run_failed":
                prior_failed_seconds += float(event.get("dataset_build_seconds") or 0.0)
    challenger_total_seconds = challenger_seconds + prior_failed_seconds
    deterministic = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": {
            "minimum_citation_coverage": minimum_coverage,
            "minimum_long_form_words": minimum_long_form_words,
            "maximum_training_tokens": maximum_tokens,
            "maximum_lexical_question_similarity": maximum_similarity,
        },
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "comparative": {
            "accepted_candidate_change": len(challenger) - len(baseline),
            "candidate_yield_rate_change": round(
                challenger_metrics["candidate_yield_rate"] - baseline_metrics["candidate_yield_rate"], 6
            ),
            "baseline_wall_seconds": baseline_seconds,
            "challenger_total_active_wall_seconds": round(challenger_total_seconds, 3),
            "challenger_prior_failed_wall_seconds": round(prior_failed_seconds, 3),
            "active_wall_time_speedup": round(baseline_seconds / challenger_total_seconds, 6),
            "baseline_candidates_per_hour": round(len(baseline) * 3600 / baseline_seconds, 6),
            "challenger_candidates_per_hour": round(len(challenger) * 3600 / challenger_total_seconds, 6),
        },
    }
    write_json_atomic(output_dir / "deterministic_comparison.json", deterministic)

    queue, key_rows, selected_bundles = _paired_sample(
        baseline,
        challenger,
        single_count=single_count,
        multi_count=multi_count,
        seed=seed,
    )
    write_jsonl_atomic(output_dir / "semantic_blinded_queue.jsonl", queue)
    write_jsonl_atomic(output_dir / "semantic_blinding_key.jsonl", key_rows)
    semantic_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": "35 matched pairs: 25 single-source and 10 multi-source; configuration labels blinded",
        "seed": seed,
        "pair_count": len(queue),
        "candidate_reviews": len(queue) * 2,
        "single_source_pairs": single_count,
        "multi_source_pairs": multi_count,
        "selected_bundle_ids": selected_bundles,
        "queue_sha256": sha256_file(output_dir / "semantic_blinded_queue.jsonl"),
        "blinding_key_sha256": sha256_file(output_dir / "semantic_blinding_key.jsonl"),
        "status": "pending_blinded_semantic_review",
    }
    write_json_atomic(output_dir / "semantic_audit_manifest.json", semantic_manifest)
    return {
        "freeze_manifest": str(output_dir / "freeze_manifest.json"),
        "deterministic_comparison": str(output_dir / "deterministic_comparison.json"),
        "semantic_queue": str(output_dir / "semantic_blinded_queue.jsonl"),
        "semantic_manifest": semantic_manifest,
        "baseline_accepted": len(baseline),
        "challenger_accepted": len(challenger),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two immutable SFT dataset releases")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-pairs", type=int, default=25)
    parser.add_argument("--multi-pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    result = build_comparison(
        args.root.resolve(),
        args.baseline.resolve(),
        args.challenger.resolve(),
        args.output.resolve(),
        single_count=args.single_pairs,
        multi_count=args.multi_pairs,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
