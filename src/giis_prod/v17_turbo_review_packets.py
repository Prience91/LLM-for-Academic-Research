#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from giis_prod.v17_turbo_review_routing import REVISION, write_bound

ROOT = Path("/home/codeaccounting/Documents/Mini run")
OUTPUT_ROOT = (ROOT / "artifacts/v17_turbo").resolve()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evidence_from(row: dict) -> str:
    user = next(x["content"] for x in row["student_messages"] if x.get("role") == "user")
    if "Evidence:\n" not in user or "\n\nQuestion: " not in user:
        raise ValueError("Cannot recover evidence for " + row["example_id"])
    return user.split("Evidence:\n", 1)[1].rsplit("\n\nQuestion: ", 1)[0].strip()


def packet_digest(packet: dict) -> str:
    body = {key: packet[key] for key in (
        "lane", "question", "evidence", "answer", "teacher_declared_obligations"
    )}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare(run: Path) -> dict:
    run = run.resolve()
    if not run.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Run must remain below artifacts/v17_turbo")
    plan = read(run / "plan.json")
    progress = read(run / "progress.json")
    structured = read(run / "structured_qualification/manifest.json")
    digest = plan["plan_digest"]
    if plan["configuration"].get("v17_turbo_review", {}).get("routing_revision") != REVISION:
        raise RuntimeError("Frozen plan uses historical review routing; create a fresh plan")
    for rel, expected in plan["bindings"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen source binding drift: " + rel)
    expected_structured = int(
        plan["configuration"]["v17_turbo_structured_qualification"]["planned_total_candidates"]
    )
    if progress.get("plan_digest") != digest:
        raise RuntimeError("Progress/plan digest mismatch")
    if structured.get("plan_digest") != digest or structured.get("status") != "complete":
        raise RuntimeError("Structured qualification is incomplete or foreign")
    if int(structured.get("case_count", -1)) != expected_structured:
        raise RuntimeError("Structured qualification count mismatch")
    files = sorted((run / "passing_candidates").glob("*.json"))
    if len(files) != int(progress["python_passing_candidates"]):
        raise RuntimeError("Passing-candidate materialized view mismatch")
    rows = []
    for path in files:
        row = read(path)
        if row.get("plan_digest") != digest:
            raise RuntimeError("Foreign passing candidate: " + str(path))
        if row.get("dataset_kind") not in {"core_single_source", "multi_source_extra"}:
            raise RuntimeError("Unknown primary dataset kind")
        if row.get("partition") not in {"train", "validation"}:
            raise RuntimeError("Invalid primary partition")
        candidate = row["candidate"]
        packet = {
            "review_id": row["example_id"],
            "plan_digest": digest,
            "partition": row["partition"],
            "lane": "single_source" if row["dataset_kind"] == "core_single_source" else "multi_source",
            "document_ids": row["document_ids"],
            "bundle_id": row["bundle_id"],
            "question": candidate["question"],
            "evidence": evidence_from(row),
            "answer": candidate["answer"],
            "teacher_declared_obligations": candidate["coverage_obligations"],
            "answer_units": candidate.get("answer_units", []),
        }
        packet["candidate_sha256"] = packet_digest(packet)
        rows.append(packet)
    primary = {read(path)["example_id"]: read(path) for path in files}
    queue_info = structured["manual_semantic_queue"]
    creative_path = Path(queue_info["path"])
    if not creative_path.resolve().is_relative_to(run):
        raise RuntimeError("Foreign creative queue")
    if hashlib.sha256(creative_path.read_bytes()).hexdigest() != queue_info["sha256"]:
        raise RuntimeError("Creative queue digest drift")
    creative = [json.loads(x) for x in creative_path.read_text().splitlines() if x.strip()]
    if len(creative) != int(queue_info["row_count"]) or len(creative) != int(structured["deterministic_passed"]):
        raise RuntimeError("Creative queue count drift")
    for row in creative:
        parent = primary[row["source_example_id"]]
        if row.get("plan_digest") != digest or row.get("document_ids") != parent["document_ids"] or row.get("bundle_id") != parent["bundle_id"]:
            raise RuntimeError("Creative candidate/parent identity mismatch")
        if not row["deterministic_review"].get("passed") or row["answer_style"] not in {"table", "comparison"}:
            raise RuntimeError("Unqualified creative candidate")
        packet = {"review_id": row["case_id"], "plan_digest": digest,
                  "partition": parent["partition"], "lane": row["answer_style"],
                  "document_ids": row["document_ids"], "bundle_id": row["bundle_id"],
                  "source_example_id": row["source_example_id"], "question": row["question"],
                  "evidence": evidence_from(parent), "answer": row["candidate"]["answer"],
                  "teacher_declared_obligations": row["required_coverage_obligations"],
                  "answer_units": row["candidate"].get("answer_units", [])}
        packet["candidate_sha256"] = packet_digest(packet)
        rows.append(packet)
    ids = [row["review_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise RuntimeError("All-lane packet set is empty or contains duplicate IDs")
    out = run / "oss120b_all_lane_semantic_census"
    out.mkdir(parents=True, exist_ok=True)
    packets = out / "census_packets_blinded.jsonl"
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    if packets.exists() and packets.read_bytes() != data:
        raise RuntimeError("Immutable OSS packet set changed")
    if not packets.exists():
        with packets.open("xb") as handle: handle.write(data)
    manifest = {
        "schema": "giis_v17_turbo_all_lane_packets_v2",
        "protocol": "V17-Turbo GPT-OSS-120B component-mapped dual-axis census",
        "source_plan_digest": digest,
        "candidate_count": len(rows),
        "routing_revision": REVISION,
        "gold_selection": "completed_oss_semantic_rejects",
        "packet_sha256": hashlib.sha256(packets.read_bytes()).hexdigest(),
        "gold_labels_present": False,
        "training_release": False,
    }
    write_bound(out / "census_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().run), indent=2))


if __name__ == "__main__":
    main()
