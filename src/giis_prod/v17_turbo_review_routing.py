"""Split-specific review queues, authorization, and admission provenance."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REVISION = "split_review_v2"
SEMANTIC_REJECTS = {"minor_omission", "materially_incomplete", "unsupported"}
LANES = {"single_source", "multi_source", "table", "comparison"}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write_bound(path, value):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Immutable review artifact drift: " + str(path))
        return
    with path.open("xb") as f:
        f.write(data)

def write_queue(directory, packets, context):
    directory = Path(directory)
    ids = [p["review_id"] for p in packets]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gold packet IDs")
    for p in packets:
        if p["lane"] not in LANES:
            raise ValueError("Unknown review lane")
        if any(k.startswith("oss_") or k in {"verdict", "rationale", "axis_outputs"} for k in p):
            raise ValueError("Gold packet leaks machine verdict")
    path = directory / "review_packets_blinded.jsonl"
    data = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in packets).encode()
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Frozen gold queue changed")
    if not path.exists():
        with path.open("xb") as f: f.write(data)
    manifest = {"schema": "giis_v17_turbo_gold_queue_v2", "routing_revision": REVISION,
                "packet_count": len(packets), "packet_sha256": digest(path),
                "gold_review_requires_explicit_authorization": True,
                "blinded_to_oss": True, **context}
    write_bound(directory / "queue_manifest.json", manifest)
    return manifest

def validate_queue(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "queue_manifest.json").read_text())
    path = directory / "review_packets_blinded.jsonl"
    packets = read_rows(path)
    if manifest.get("routing_revision") != REVISION or digest(path) != manifest["packet_sha256"]:
        raise ValueError("Gold queue binding mismatch")
    if len(packets) != manifest["packet_count"] or len({p["review_id"] for p in packets}) != len(packets):
        raise ValueError("Gold queue census mismatch")
    return manifest, packets

def authorize(directory, statement):
    manifest, _ = validate_queue(directory)
    if not statement.strip(): raise ValueError("Explicit operator authorization statement required")
    auth = {"schema": "giis_gold_authorization_v1", "authorized": True,
            "queue_manifest_sha256": digest(Path(directory) / "queue_manifest.json"),
            "packet_sha256": manifest["packet_sha256"], "operator_statement": statement}
    write_bound(Path(directory) / "authorization.json", auth)
    return auth

def bind_gold(directory):
    directory = Path(directory)
    manifest, packets = validate_queue(directory)
    auth = json.loads((directory / "authorization.json").read_text())
    if auth.get("authorized") is not True or auth.get("queue_manifest_sha256") != digest(directory / "queue_manifest.json") or auth.get("packet_sha256") != manifest["packet_sha256"]:
        raise ValueError("Gold authorization is absent or foreign")
    reviews = read_rows(directory / "reviews.jsonl")
    by = {r["review_id"]: r for r in reviews}
    if len(by) != len(reviews) or set(by) != {p["review_id"] for p in packets}:
        raise ValueError("Gold disposition census is incomplete or foreign")
    for packet in packets:
        r = by[packet["review_id"]]
        if r.get("candidate_sha256") != packet["candidate_sha256"] or r.get("lane") != packet["lane"]:
            raise ValueError("Gold decision candidate identity mismatch")
        if r.get("reviewer") not in {"codex_gold", "Codex gold"}:
            raise ValueError("Unauthorized gold reviewer")
        if r.get("verdict") == "error_fail_closed":
            terminal_path = directory / "gold_fail_closed_disposition_authorization.json"
            if not terminal_path.exists(): raise ValueError("Gold fail-closed disposition awaits operator authorization; no retry")
            terminal = json.loads(terminal_path.read_text())
            if terminal.get("reviews_sha256") != digest(directory / "reviews.jsonl") or terminal.get("packet_sha256") != manifest["packet_sha256"] or terminal.get("authorized") is not True or terminal.get("disposition") not in {"quarantine", "archive"}:
                raise ValueError("Gold failure disposition authorization is foreign")
            continue
        if r.get("verdict") not in {"complete", "minor_omission", "materially_incomplete", "unsupported"}:
            raise ValueError("Unresolved gold verdict")
    summary = {"schema": "giis_gold_disposition_census_v2", "status": "complete",
               "routing_revision": REVISION, "reviewed": len(reviews),
               "counts": dict(Counter(r["verdict"] for r in reviews)),
               "packet_sha256": manifest["packet_sha256"],
               "queue_manifest_sha256": digest(directory / "queue_manifest.json"),
               "reviews_sha256": digest(directory / "reviews.jsonl"),
               "authorization_sha256": digest(directory / "authorization.json")}
    failures = [r for r in reviews if r.get("verdict") == "error_fail_closed"]
    if failures:
        terminal = json.loads((directory / "gold_fail_closed_disposition_authorization.json").read_text())
        summary["gold_fail_closed_disposition"] = terminal["disposition"]
        summary["gold_fail_closed_authorization_sha256"] = digest(directory / "gold_fail_closed_disposition_authorization.json")
        write_bound(directory / "quarantine/gold_fail_closed_terminal/manifest.json",
                    {"cases":failures,"count":len(failures),"disposition":terminal["disposition"],
                     "authorization_sha256":summary["gold_fail_closed_authorization_sha256"],
                     "retry_allowed":False,"repair_allowed":False,"eligible":False})
    write_bound(directory / "disposition_manifest.json", summary)
    return summary

class _GoldContext:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        bind_gold(self.directory)
        _, packets = validate_queue(self.directory)
        self.packets = {p["review_id"]: p for p in packets}
        self.reviews = {r["review_id"]: r for r in read_rows(self.directory / "reviews.jsonl")}

def gold_context(directory):
    return _GoldContext(directory)

def admission(packet, oss=None, gold=None, gold_queue=None, context=None):
    """Validate reviewer ownership and identity before selecting an eligible row."""
    if packet.get("partition") == "heldout":
        if not gold or gold.get("verdict") != "complete": raise ValueError("Held-out reference requires gold complete")
        path = "gold_direct_complete"
    else:
        if packet.get("partition") not in {"train", "validation"}: raise ValueError("Unknown partition")
        if not oss: raise ValueError("Training/validation candidate lacks OSS review")
        for k in ("review_id", "candidate_sha256", "lane", "partition", "plan_digest"):
            if oss.get(k) != packet.get(k): raise ValueError("OSS candidate identity mismatch: " + k)
        if oss.get("verdict") == "complete":
            if oss.get("coverage") != "complete" or oss.get("support") != "supported": raise ValueError("OSS complete lacks successful axes")
            return "oss_direct_complete"
        if oss.get("verdict") not in SEMANTIC_REJECTS or oss.get("coverage") not in {"complete", "minor_omission", "materially_incomplete"} or oss.get("support") not in {"supported", "unsupported"}:
            raise ValueError("OSS fail-closed outcome is terminal and cannot be gold salvaged")
        if not gold or gold.get("verdict") != "complete" or not gold_queue:
            raise ValueError("Completed OSS semantic rejection requires authorized gold complete")
        path = "gold_salvaged_after_oss_non_admission"
    if gold_queue is None: raise ValueError("Gold queue binding required")
    if context is None:
        context = gold_context(gold_queue)
    if not isinstance(context, _GoldContext) or context.directory != Path(gold_queue).resolve():
        raise ValueError("Foreign gold validation context")
    original = context.packets.get(packet["review_id"], {})
    if original.get("candidate_sha256") != packet["candidate_sha256"]:
        raise ValueError("Candidate absent from authorized gold queue")
    for k in ("review_id", "candidate_sha256", "lane"):
        if gold.get(k) != packet.get(k): raise ValueError("Gold candidate identity mismatch")
    if context.reviews[packet["review_id"]] != gold: raise ValueError("Gold decision differs from committed census")
    return path

def authorize_gold_failures(directory, statement, disposition):
    manifest, packets = validate_queue(directory)
    if not statement.strip() or disposition not in {"quarantine", "archive"}:
        raise ValueError("Explicit gold failure disposition authorization required")
    reviews = read_rows(Path(directory) / "reviews.jsonl")
    failures = [r for r in reviews if r.get("verdict") == "error_fail_closed"]
    if not failures: raise ValueError("No gold fail-closed outcomes to authorize")
    value = {"authorized": True, "operator_statement": statement, "disposition": disposition,
             "packet_sha256": manifest["packet_sha256"], "reviews_sha256": digest(Path(directory) / "reviews.jsonl"),
             "candidate_ids": [r["review_id"] for r in failures], "retry_allowed": False, "repair_allowed": False}
    write_bound(Path(directory) / "gold_fail_closed_disposition_authorization.json", value)
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["authorize", "bind", "authorize-gold-failures"])
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--operator-statement", default="")
    parser.add_argument("--disposition", choices=["quarantine", "archive"], default="archive")
    args = parser.parse_args()
    if args.action == "authorize-gold-failures": result = authorize_gold_failures(args.queue, args.operator_statement, args.disposition)
    else: result = authorize(args.queue, args.operator_statement) if args.action == "authorize" else bind_gold(args.queue)
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
