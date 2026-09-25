from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

from .citations import citation_coverage
from .config import Settings
from .io import content_version, read_jsonl, write_json_atomic, write_jsonl_atomic


def _coverage_band(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    answer = str(messages[-1].get("content") or "") if messages else ""
    coverage = citation_coverage(answer)
    if coverage < 0.60:
        return "40-59"
    if coverage < 0.80:
        return "60-79"
    return "80-100"


def plan_semantic_audit(settings: Settings) -> dict[str, Any]:
    sft_dir = settings.path("paths.sft_dir")
    rows = list(read_jsonl(sft_dir / "train.jsonl")) + list(
        read_jsonl(sft_dir / "validation.jsonl")
    )
    if not rows:
        raise ValueError("Cannot plan a semantic audit for an empty dataset")
    fraction = float(settings.get("smoke_run_ii.semantic_audit_fraction"))
    maximum = int(settings.get("smoke_run_ii.semantic_audit_maximum"))
    audit_count = min(maximum, math.ceil(len(rows) * fraction))
    seed = int(settings.get("smoke_run_ii.seed"))
    rng = random.Random(seed)

    strata: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        document_count = "multi" if len(row.get("document_ids") or []) > 1 else "single"
        key = (
            str((row.get("document_ids") or ["unknown"])[0]),
            str(row.get("answer_style") or "prose"),
            document_count,
            str(row.get("split") or "unknown"),
            _coverage_band(row),
        )
        strata[key].append(row)
    queues: list[tuple[tuple[str, str, str, str, str], deque[dict[str, Any]]]] = []
    for key in sorted(strata):
        rng.shuffle(strata[key])
        queues.append((key, deque(strata[key])))
    rng.shuffle(queues)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < audit_count and queues:
        next_round: list[
            tuple[tuple[str, str, str, str, str], deque[dict[str, Any]]]
        ] = []
        for key, queue in queues:
            while queue:
                row = queue.popleft()
                example_id = str(row["example_id"])
                if example_id not in selected_ids:
                    selected_ids.add(example_id)
                    selected.append(
                        {
                            **row,
                            "audit_stratum": {
                                "primary_document_id": key[0],
                                "answer_style": key[1],
                                "source_mode": key[2],
                                "split": key[3],
                                "citation_coverage_band": key[4],
                            },
                            "semantic_verdict": None,
                            "semantic_notes": None,
                        }
                    )
                    break
            if queue:
                next_round.append((key, queue))
            if len(selected) >= audit_count:
                break
        queues = next_round

    queue_path = sft_dir / "semantic_audit_queue.jsonl"
    write_jsonl_atomic(queue_path, selected)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "min(400, ceil(10_percent_of_python_accepted))",
        "seed": seed,
        "accepted_candidates": len(rows),
        "audit_candidates": len(selected),
        "audit_fraction": len(selected) / len(rows),
        "queue_path": str(queue_path),
        "queue_digest": content_version(selected),
        "strata": [
            {
                "primary_document_id": key[0],
                "answer_style": key[1],
                "source_mode": key[2],
                "split": key[3],
                "citation_coverage_band": key[4],
                "population": len(value),
            }
            for key, value in sorted(strata.items())
        ],
    }
    write_json_atomic(sft_dir / "semantic_audit_manifest.json", manifest)
    return manifest
