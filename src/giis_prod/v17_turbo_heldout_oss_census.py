#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from giis_prod import heldout_oss_review as protocol


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest():
    rows = {}
    if protocol.RESULTS.exists():
        for row in read_jsonl(protocol.RESULTS):
            rows[(row["review_id"], row["axis"])] = row
    return rows


def review_with_retries(task):
    result = {}
    for _ in range(3):
        result = protocol.review_axis(task)
        if result.get("status") == "ok":
            return result
    return result


def run():
    packets = read_jsonl(protocol.PACKETS)
    completed = {key for key, value in latest().items() if value.get("status") == "ok"}
    tasks = [
        (packet, axis)
        for packet in packets
        for axis in ("coverage", "support")
        if (packet["review_id"], axis) not in completed
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(review_with_retries, tasks))


def configure(run):
    run = run.resolve()
    status = json.loads((run / "status.json").read_text())
    if status.get("review_routing_revision") == "split_review_v2":
        raise ValueError("Held-out candidates require gold review; OSS census is historical only")
    root = protocol.ROOT.resolve()
    allowed = (root / "artifacts/v17_turbo").resolve()
    if not run.is_relative_to(allowed):
        raise ValueError("Held-out run must remain below artifacts/v17_turbo")
    protocol.OUT = run / "oss120b_single_source_review"
    protocol.PACKETS = run / "review_packets_oss_single_source.jsonl"
    protocol.RESULTS = protocol.OUT / "axis_results.jsonl"
    protocol.SUMMARY = protocol.OUT / "summary.json"
    protocol.CONCURRENCY = 1
    return run


def score(run):
    packets = read_jsonl(protocol.PACKETS)
    axes = latest()
    counts = Counter()
    cases = []
    for packet in packets:
        rid = packet["review_id"]
        coverage = axes.get((rid, "coverage"))
        support = axes.get((rid, "support"))
        if not coverage or not support or coverage.get("status") != "ok" or support.get("status") != "ok":
            verdict = "error_fail_closed"
        else:
            cov = coverage["review"]["coverage"]
            sup = support["review"]["support"]
            verdict = "unsupported" if sup == "unsupported" else cov
        counts[verdict] += 1
        cases.append({
            "review_id": rid,
            "candidate_sha256": packet["candidate_sha256"],
            "lane": "single_source",
            "verdict": verdict,
            "disposition": "eligible_for_benchmark_freeze" if verdict == "complete" else "quarantine",
            "coverage": coverage.get("review", {}).get("coverage") if coverage else None,
            "support": support.get("review", {}).get("support") if support else None,
        })
    status = json.loads((run / "status.json").read_text())
    summary = {
        "schema": "giis_v17_turbo_heldout_oss_review_v2",
        "protocol": "gpt_oss_120b_component_mapped_dual_axis",
        "selection_digest": status["selection_digest"],
        "allocation_digest": status["allocation_digest"],
        "packet_sha256": hashlib.sha256(protocol.PACKETS.read_bytes()).hexdigest(),
        "total": len(packets),
        "counts": dict(counts),
        "unresolved_errors": counts["error_fail_closed"],
        "cases": cases,
        "single_source_gold_review_fraction": 0.0,
        "benchmark_freeze": False,
    }
    temporary = protocol.SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(protocol.SUMMARY)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    run_path = configure(parser.parse_args().run)
    protocol.OUT.mkdir(parents=True, exist_ok=True)
    if not protocol.PACKETS.is_file():
        raise FileNotFoundError(protocol.PACKETS)
    run()
    result = score(run_path)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
