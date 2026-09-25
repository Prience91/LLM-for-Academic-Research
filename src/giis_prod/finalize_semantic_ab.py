from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .io import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic

_DIMENSIONS = (
    "grounding",
    "citation_entailment",
    "completeness",
    "relevance",
    "clarity",
    "synthesis",
)


def _exact_sign_test(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if not trials:
        return None
    tail = sum(math.comb(trials, index) for index in range(0, min(wins, losses) + 1))
    return round(min(1.0, 2.0 * tail / (2**trials)), 6)


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = sum(bool(row["review"]["clean"]) for row in rows)
    dimensions: dict[str, Any] = {}
    for dimension in _DIMENSIONS:
        values = [
            float(row["review"][dimension])
            for row in rows
            if row["review"].get(dimension) is not None
        ]
        dimensions[dimension] = {
            "count": len(values),
            "mean": round(mean(values), 6) if values else None,
        }
    normalized = [float(row["normalized_score"]) for row in rows]
    issues = Counter(
        issue for row in rows for issue in (row["review"].get("issues") or [])
    )
    return {
        "reviewed_candidates": len(rows),
        "clean_candidates": clean,
        "clean_rate": round(clean / len(rows), 6),
        "mean_normalized_score": round(mean(normalized), 6),
        "dimension_scores": dimensions,
        "issue_counts": dict(sorted(issues.items())),
    }


def finalize(queue_path: Path, key_path: Path, judgments_path: Path, output_dir: Path) -> dict[str, Any]:
    queue = {row["pair_id"]: row for row in read_jsonl(queue_path)}
    keys = {row["pair_id"]: row for row in read_jsonl(key_path)}
    judgments = {row["pair_id"]: row for row in read_jsonl(judgments_path)}
    if len(queue) != 35 or len(keys) != 35 or len(judgments) != 35:
        raise RuntimeError("Semantic A/B finalization requires exactly 35 unique pairs")
    if set(queue) != set(keys) or set(queue) != set(judgments):
        raise RuntimeError("Queue, key, and judgment pair identities do not match")

    arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unblinded: list[dict[str, Any]] = []
    preference_counts: Counter[str] = Counter()
    paired_differences: list[float] = []
    strata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair_id in sorted(queue):
        queued = queue[pair_id]
        key = keys[pair_id]
        judgment = judgments[pair_id]
        if judgment["source_mode"] != queued["source_mode"]:
            raise RuntimeError(f"Source-mode mismatch for {pair_id}")
        per_pair: dict[str, Any] = {
            "pair_id": pair_id,
            "source_mode": queued["source_mode"],
            "bundle_id": queued["bundle_id"],
            "responses": {},
        }
        score_by_arm: dict[str, float] = {}
        for label in ("a", "b"):
            arm = str(key[f"response_{label}_arm"])
            review = judgment[f"response_{label}"]
            dimensions = [
                float(review[name])
                for name in _DIMENSIONS
                if review.get(name) is not None
            ]
            normalized = sum(dimensions) / (2 * len(dimensions))
            row = {
                "pair_id": pair_id,
                "source_mode": queued["source_mode"],
                "arm": arm,
                "blinded_label": label,
                "example_id": key[f"response_{label}_example_id"],
                "review": review,
                "normalized_score": round(normalized, 6),
            }
            arm_rows[arm].append(row)
            strata[queued["source_mode"]][arm].append(row)
            score_by_arm[arm] = normalized
            per_pair["responses"][arm] = row
        paired_differences.append(score_by_arm["challenger"] - score_by_arm["baseline"])
        preference = str(judgment.get("preference") or "tie")
        if preference in {"a", "b"}:
            preference_arm = str(key[f"response_{preference}_arm"])
        else:
            preference_arm = "tie"
        preference_counts[preference_arm] += 1
        per_pair["preference"] = preference_arm
        per_pair["notes"] = judgment.get("notes")
        unblinded.append(per_pair)

    wins = preference_counts["challenger"]
    losses = preference_counts["baseline"]
    difference_mean = mean(paired_differences)
    difference_se = (
        stdev(paired_differences) / math.sqrt(len(paired_differences))
        if len(paired_differences) > 1
        else 0.0
    )
    result = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "reviewer": "Codex blinded manual evidence audit",
        "rubric": {
            "dimensions": {
                "grounding": "0-2",
                "citation_entailment": "0-2",
                "completeness": "0-2",
                "relevance": "0-2",
                "clarity": "0-2",
                "synthesis": "0-2 for multi-source only",
            },
            "clean": "No substantive unsupported, inaccurate, or citation-mismatched claim",
        },
        "design": {
            "pairs": 35,
            "candidate_reviews": 70,
            "single_source_pairs": 25,
            "multi_source_pairs": 10,
            "labels_hidden_until_all_judgments_complete": True,
            "questions_may_differ_within_pair": True,
            "pairing_controls": ["exact evidence bundle", "source mode", "response mode and answer style when available"],
        },
        "inputs": {
            "queue_sha256": sha256_file(queue_path),
            "blinding_key_sha256": sha256_file(key_path),
            "judgments_sha256": sha256_file(judgments_path),
        },
        "arms": {arm: _arm_summary(rows) for arm, rows in sorted(arm_rows.items())},
        "strata": {
            mode: {arm: _arm_summary(rows) for arm, rows in sorted(arms.items())}
            for mode, arms in sorted(strata.items())
        },
        "paired": {
            "challenger_minus_baseline_mean_normalized_score": round(difference_mean, 6),
            "normal_approximation_95_percent_interval": [
                round(difference_mean - 1.96 * difference_se, 6),
                round(difference_mean + 1.96 * difference_se, 6),
            ],
            "preferences": dict(sorted(preference_counts.items())),
            "two_sided_exact_sign_test_excluding_ties": _exact_sign_test(wins, losses),
        },
        "interpretation_guardrails": [
            "The deliberately enriched multi-source stratum is not population weighted.",
            "Pair members share evidence but may answer different generated questions; aggregate arm scores are primary and direct pair preferences are secondary.",
            "The sample supports canary selection, not a claim of population-level statistical superiority.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "semantic_unblinded_reviews.jsonl", unblinded)
    write_json_atomic(output_dir / "semantic_comparison.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.queue.resolve(), args.key.resolve(), args.judgments.resolve(), args.output.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
