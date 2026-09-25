#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from giis_prod import heldout_oss_review as protocol

ROOT = Path(os.getenv("GIIS_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
OUTPUT_ROOT = (ROOT / "artifacts/v17_turbo").resolve()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def configure(run: Path) -> tuple[Path, Path]:
    run = run.resolve()
    if not run.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Run must remain below artifacts/v17_turbo")
    out = run / "oss120b_single_source_semantic_census"
    packets = out / "census_packets_blinded.jsonl"
    protocol.OUT = out
    protocol.PACKETS = packets
    protocol.RESULTS = out / "census_axis_results.jsonl"
    protocol.SUMMARY = out / "census_summary.json"
    protocol.CONCURRENCY = 1
    return out, packets


def latest() -> dict:
    values = {}
    if protocol.RESULTS.exists():
        for row in read_jsonl(protocol.RESULTS):
            values[(row["review_id"], row["axis"])] = row
    return values


def review_with_retries(task):
    result = {}
    for _ in range(3):
        result = protocol.review_axis(task)
        if result.get("status") == "ok":
            return result
    return result


def execute(packets_path: Path) -> None:
    packets = read_jsonl(packets_path)
    done = {key for key, value in latest().items() if value.get("status") == "ok"}
    tasks = [
        (packet, axis)
        for packet in packets
        for axis in ("coverage", "support")
        if (packet["review_id"], axis) not in done
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(review_with_retries, tasks))


def score(run: Path, packets_path: Path, out: Path) -> dict:
    plan_digest = json.loads((run / "plan.json").read_text())["plan_digest"]
    packets = read_jsonl(packets_path)
    axes = latest()
    counts = Counter()
    cases = []
    errors = 0
    for packet in packets:
        rid = packet["review_id"]
        coverage = axes.get((rid, "coverage"))
        support = axes.get((rid, "support"))
        if not coverage or not support or coverage.get("status") != "ok" or support.get("status") != "ok":
            verdict = "error_fail_closed"
            errors += 1
        else:
            cov = coverage["review"]["coverage"]
            sup = support["review"]["support"]
            verdict = "unsupported" if sup == "unsupported" else cov
        counts[verdict] += 1
        cases.append({
            "review_id": rid,
            "candidate_sha256": packet["candidate_sha256"],
            "plan_digest": plan_digest,
            "lane": "single_source",
            "verdict": verdict,
            "disposition": "eligible_for_operator_release" if verdict == "complete" else "quarantine",
            "coverage": coverage.get("review", {}).get("coverage") if coverage else None,
            "support": support.get("review", {}).get("support") if support else None,
        })
    summary = {
        "schema": "giis_v17_turbo_oss_census_v1",
        "protocol": "gpt_oss_120b_component_mapped_dual_axis",
        "source_plan_digest": plan_digest,
        "packet_sha256": hashlib.sha256(packets_path.read_bytes()).hexdigest(),
        "total": len(packets),
        "counts": dict(counts),
        "unresolved_errors": errors,
        "cases": cases,
        "single_source_gold_review_fraction": 0.0,
        "training_admission_effect": "none",
        "training_release": False,
    }
    temporary = out / "census_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(out / "census_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    run = parser.parse_args().run.resolve()
    out, packets = configure(run)
    if not packets.is_file():
        raise FileNotFoundError(packets)
    execute(packets)
    summary = score(run, packets, out)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
