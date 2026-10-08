"""Freeze held-out gold queues and bind split-specific semantic dispositions."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from giis_prod.v17_turbo_review_routing import REVISION, SEMANTIC_REJECTS, digest, read_rows, write_bound, write_queue, bind_gold, admission, gold_context

def prepare_heldout(run, configuration):
    run = Path(run)
    from giis_prod.v17_turbo_pre_gold import heldout_authoring_census
    status, packets, terminal_ids = heldout_authoring_census(run, configuration)
    for p in packets:
        if p.get("assigned_reviewer") != "Codex gold":
            raise ValueError("Held-out candidate has wrong reviewer")
        p["partition"] = "heldout"
    return write_queue(run / "codex_gold_review_all_lanes", packets,
                       {"scope": "heldout_all_generated", "selection_digest": status["selection_digest"],
                        "allocation_digest": status["allocation_digest"],
                        "source_packet_sha256": digest(run / "review_packets.jsonl"),
                        "review_policy_fingerprint": status["review_policy_fingerprint"],
                        "planned_case_count": status["total"],
                        "terminal_authoring_case_ids": terminal_ids,
                        "canonical_full_count_satisfied": not terminal_ids})

def bind_main(run):
    run = Path(run)
    plan = json.loads((run / "plan.json").read_text())
    if plan["configuration"]["v17_turbo_review"].get("routing_revision") != REVISION:
        raise ValueError("Historical main routing; fresh plan required")
    out = run / "oss120b_all_lane_semantic_census"
    packets = read_rows(out / "census_packets_blinded.jsonl")
    census = json.loads((out / "census_summary.json").read_text())
    if census.get("source_plan_digest") != plan["plan_digest"] or census["packet_sha256"] != digest(out / "census_packets_blinded.jsonl"):
        raise ValueError("Main census binding drift")
    packet_manifest = json.loads((out / "census_manifest.json").read_text())
    if packet_manifest["candidate_count"] != len(packets) or census["total"] != len(packets) or packet_manifest["packet_sha256"] != digest(out / "census_packets_blinded.jsonl"):
        raise ValueError("Main frozen packet census drift")
    decisions = {r["review_id"]: r for r in census["cases"]}
    if len(decisions) != len(census["cases"]) or set(decisions) != {p["review_id"] for p in packets}:
        raise ValueError("Main OSS census incomplete")
    queue = run / "codex_gold_review_oss_non_admissions"
    rejected_count = sum(r["verdict"] in SEMANTIC_REJECTS for r in decisions.values())
    if rejected_count:
        gold_summary = bind_gold(queue)
        gold = {r["review_id"]: r for r in read_rows(queue / "reviews.jsonl")}
        gold_digest = digest(queue / "disposition_manifest.json")
        context = gold_context(queue)
    else:
        from giis_prod.v17_turbo_review_routing import validate_queue
        _, empty = validate_queue(queue)
        if empty: raise ValueError("Unexpected non-empty gold queue")
        gold = {}
        gold_digest = None
        context = None
    expected_rejects = {rid for rid, r in decisions.items() if r["verdict"] in SEMANTIC_REJECTS}
    if set(gold) != expected_rejects:
        raise ValueError("Gold queue differs from completed OSS semantic rejects")
    rows=[]
    for p in packets:
        oss=decisions[p["review_id"]]; g=gold.get(p["review_id"])
        eligible=oss["verdict"]=="complete" or bool(g and g["verdict"]=="complete")
        path=admission(p,oss=oss,gold=g,gold_queue=queue,context=context) if eligible else None
        verdict=g["verdict"] if g else oss["verdict"]
        rows.append({"review_id":p["review_id"],"candidate_sha256":p["candidate_sha256"],
                     "plan_digest":p["plan_digest"],"lane":p["lane"],"partition":p["partition"],
                     "verdict":verdict,"eligible":eligible,"admission_path":path,
                     "disposition":"eligible" if eligible else "hold" if verdict=="minor_omission" else ("terminal_" + gold_summary.get("gold_fail_closed_disposition", "quarantine")) if verdict=="error_fail_closed" and g else "terminal_quarantine" if verdict=="error_fail_closed" else "discard",
                     "repair_allowed":False if verdict=="error_fail_closed" else verdict=="minor_omission"})
    manifest={"schema":"giis_v17_turbo_main_semantic_review_v2","status":"complete",
              "routing_revision":REVISION,"plan_digest":plan["plan_digest"],"population":len(packets),
              "gold_review_authorized":bool(rejected_count),"gold_review_required":bool(rejected_count),"unreviewed_candidates":0,
              "unreviewed_multi_or_creative_candidates":0,"counts":dict(Counter(r["verdict"] for r in rows)),
              "counts_by_lane_split":{key:dict(Counter(r["verdict"] for r in rows if r["partition"]+"/"+r["lane"]==key)) for key in sorted({r["partition"]+"/"+r["lane"] for r in rows})},
              "oss_summary_sha256":digest(out/"census_summary.json"),
              "gold_disposition_manifest_sha256":gold_digest,"cases":rows}
    write_bound(run/"main_semantic_review_manifest.json",manifest)
    write_bound(run/"semantic_workflow_completion.json",{"stage":"gold_review_complete",
                "routing_revision":REVISION,"plan_digest":plan["plan_digest"],
                "manifest_sha256":digest(run/"main_semantic_review_manifest.json")})
    return manifest

def bind_heldout(run):
    run=Path(run); queue=run/"codex_gold_review_all_lanes"
    summary=bind_gold(queue)
    from giis_prod.v17_turbo_review_routing import validate_queue
    qm,packets=validate_queue(queue)
    if qm.get("source_packet_sha256")!=digest(run/"review_packets.jsonl"):
        raise ValueError("Held-out source packets changed")
    reviews={r["review_id"]:r for r in read_rows(queue/"reviews.jsonl")}
    context=gold_context(queue)
    cases=[]
    for p in packets:
        r=reviews[p["review_id"]]; eligible=r["verdict"]=="complete"
        cases.append({"review_id":p["review_id"],"candidate_sha256":p["candidate_sha256"],
                      "lane":p["lane"],"verdict":r["verdict"],"eligible":eligible,
                      "disposition":"eligible" if eligible else "hold" if r["verdict"]=="minor_omission" else "terminal_"+summary.get("gold_fail_closed_disposition","quarantine") if r["verdict"]=="error_fail_closed" else "discard",
                      "repair_allowed":r["verdict"]=="minor_omission",
                      "admission_path":admission(p,gold=r,gold_queue=queue,context=context) if eligible else None})
    manifest={"schema":"giis_v17_turbo_heldout_semantic_review_v2","status":"complete",
              "routing_revision":REVISION,"selection_digest":qm["selection_digest"],
              "allocation_digest":qm["allocation_digest"],"population":len(packets),
              "gold_review_authorized":True,"unreviewed_candidates":0,"counts":summary["counts"],
              "gold_disposition_manifest_sha256":digest(queue/"disposition_manifest.json"),"cases":cases}
    write_bound(run/"heldout_semantic_review_manifest.json",manifest)
    return manifest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare-heldout","bind-main","bind-heldout"])
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--config",type=Path,default=Path("config/settings_v17_turbo.toml"))
    args=parser.parse_args()
    if args.action=="prepare-heldout":
        import tomllib
        with args.config.open("rb") as f: config=tomllib.load(f)
        result=prepare_heldout(args.run,config)
    else: result=bind_main(args.run) if args.action=="bind-main" else bind_heldout(args.run)
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))

if __name__=="__main__":main()
