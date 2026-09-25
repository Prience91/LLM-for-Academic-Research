#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.getenv("GIIS_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
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
        if row.get("dataset_kind") != "core_single_source":
            continue
        candidate = row["candidate"]
        packet = {
            "review_id": row["example_id"],
            "plan_digest": digest,
            "partition": row["partition"],
            "lane": "single_source",
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
    ids = [row["review_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise RuntimeError("Single-source packet set is empty or contains duplicate IDs")
    out = run / "oss120b_single_source_semantic_census"
    out.mkdir(parents=True, exist_ok=True)
    packets = out / "census_packets_blinded.jsonl"
    temporary = Path(str(packets) + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(packets)
    manifest = {
        "schema": "giis_v17_turbo_single_source_packets_v1",
        "protocol": "V17-Turbo GPT-OSS-120B component-mapped dual-axis census",
        "source_plan_digest": digest,
        "candidate_count": len(rows),
        "single_source_gold_review_fraction": 0.0,
        "packet_sha256": hashlib.sha256(packets.read_bytes()).hexdigest(),
        "gold_labels_present": False,
        "training_release": False,
    }
    atomic_json(out / "census_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().run), indent=2))


if __name__ == "__main__":
    main()
