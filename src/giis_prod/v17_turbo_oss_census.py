#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path

from giis_prod import heldout_oss_review as protocol
from giis_prod.v17_turbo_review_routing import REVISION, SEMANTIC_REJECTS, write_queue, write_bound

ROOT = Path("/home/codeaccounting/Documents/Mini run")
OUTPUT_ROOT = (ROOT / "artifacts/v17_turbo").resolve()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def configure(run: Path) -> tuple[Path, Path]:
    run = run.resolve()
    if not run.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Run must remain below artifacts/v17_turbo")
    out = run / "oss120b_all_lane_semantic_census"
    packets = out / "census_packets_blinded.jsonl"
    manifest = json.loads((out / "census_manifest.json").read_text())
    if manifest.get("routing_revision") != REVISION or manifest["packet_sha256"] != hashlib.sha256(packets.read_bytes()).hexdigest():
        raise ValueError("OSS packet freeze mismatch")
    plan = json.loads((run / "plan.json").read_text())
    if plan["configuration"].get("v17_turbo_review", {}).get("routing_revision") != REVISION or manifest["source_plan_digest"] != plan["plan_digest"]:
        raise ValueError("Historical or foreign OSS review plan")
    for rel, expected in plan["bindings"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError("OSS runtime source binding drift: " + rel)
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
    packet, axis = task
    history = [r for r in read_jsonl(protocol.RESULTS) if r.get("review_id") == packet["review_id"] and r.get("axis") == axis] if protocol.RESULTS.exists() else []
    if any(r.get("candidate_sha256") != packet["candidate_sha256"] for r in history):
        raise ValueError("Retry history candidate identity mismatch")
    if history and history[-1].get("status") == "ok": return history[-1]
    result = history[-1] if history else {}
    for _ in range(max(0, 2 - len(history))):
        result = protocol.review_axis(task)
        if result.get("status") == "ok": return result
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
    manifest = json.loads((out / "census_manifest.json").read_text())
    if manifest["source_plan_digest"] != plan_digest or manifest["packet_sha256"] != hashlib.sha256(packets_path.read_bytes()).hexdigest():
        raise ValueError("OSS plan or packet binding mismatch")
    if len(packets) != manifest["candidate_count"]:
        raise ValueError("OSS census population mismatch")
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
        for axis in (coverage, support):
            if axis and (axis.get("candidate_sha256") != packet["candidate_sha256"] or axis.get("lane") != packet["lane"]):
                raise ValueError("OSS axis candidate identity mismatch")
        counts[verdict] += 1
        cases.append({
            "review_id": rid,
            "candidate_sha256": packet["candidate_sha256"],
            "plan_digest": plan_digest,
            "lane": packet["lane"],
            "partition": packet["partition"],
            "verdict": verdict,
            "disposition": "eligible_for_operator_release" if verdict == "complete" else "quarantine",
            "coverage": coverage.get("review", {}).get("coverage") if coverage else None,
            "support": support.get("review", {}).get("support") if support else None,
        })
    summary = {
        "schema": "giis_v17_turbo_oss_census_v2",
        "protocol": "gpt_oss_120b_component_mapped_dual_axis",
        "source_plan_digest": plan_digest,
        "packet_sha256": hashlib.sha256(packets_path.read_bytes()).hexdigest(),
        "total": len(packets),
        "counts": dict(counts),
        "counts_by_lane_split": {key: dict(Counter(c["verdict"] for c in cases if c["partition"] + "/" + c["lane"] == key))
                                 for key in sorted({c["partition"] + "/" + c["lane"] for c in cases})},
        "unresolved_errors": errors,
        "cases": cases,
        "terminal_fail_closed_count": errors,
        "repair_of_exhausted_errors_allowed": False,
        "routing_revision": REVISION,
        "gold_selection": "completed_oss_semantic_rejects",
        "training_admission_effect": "none",
        "training_release": False,
    }
    write_bound(out / "census_summary.json", summary)
    nonadmitted = {row["review_id"] for row in cases if row["verdict"] in SEMANTIC_REJECTS}
    queue = write_queue(run / "codex_gold_review_oss_non_admissions",
                        [p for p in packets if p["review_id"] in nonadmitted],
                        {"scope": "training_validation_oss_non_admissions", "source_plan_digest": plan_digest,
                         "oss_summary_sha256": hashlib.sha256((out / "census_summary.json").read_bytes()).hexdigest()})
    terminal = [row for row in cases if row["verdict"] == "error_fail_closed"]
    write_bound(run / "quarantine/oss_fail_closed_terminal/manifest.json",
                {"schema": "giis_terminal_oss_failures_v1", "source_plan_digest": plan_digest,
                 "cases": terminal, "count": len(terminal), "retry_allowed": False,
                 "repair_allowed": False, "gold_salvage_allowed": False})
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
