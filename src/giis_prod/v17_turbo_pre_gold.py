"""Bind the complete OSS and held-out gold queues before the sole authorization pause."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from giis_prod.v17_turbo_review_routing import REVISION, SEMANTIC_REJECTS, authorize, digest, read_rows, validate_queue, write_bound


def main_oss_ready(run: Path) -> dict:
    run=Path(run).resolve()
    plan=json.loads((run/"plan.json").read_text())
    plan_digest=plan["plan_digest"]
    if plan["configuration"]["v17_turbo_review"].get("routing_revision")!=REVISION:
        raise ValueError("Historical main review policy")
    out=run/"oss120b_all_lane_semantic_census"
    source=out/"census_packets_blinded.jsonl"
    packet_manifest=json.loads((out/"census_manifest.json").read_text())
    census=json.loads((out/"census_summary.json").read_text())
    packets=read_rows(source)
    if (packet_manifest.get("routing_revision")!=REVISION or packet_manifest.get("source_plan_digest")!=plan_digest
            or packet_manifest.get("candidate_count")!=len(packets) or packet_manifest.get("packet_sha256")!=digest(source)
            or census.get("routing_revision")!=REVISION or census.get("source_plan_digest")!=plan_digest
            or census.get("packet_sha256")!=digest(source) or census.get("total")!=len(packets)
            or census.get("gold_selection")!="completed_oss_semantic_rejects"):
        raise ValueError("Incomplete or foreign OSS census")
    indexed={p["review_id"]:p for p in packets}
    decisions={c["review_id"]:c for c in census["cases"]}
    if len(indexed)!=len(packets) or len(decisions)!=len(census["cases"]) or set(indexed)!=set(decisions):
        raise ValueError("OSS decision census is incomplete")
    for rid,p in indexed.items():
        c=decisions[rid]
        if (c.get("candidate_sha256")!=p.get("candidate_sha256") or c.get("lane")!=p.get("lane")
                or c.get("partition")!=p.get("partition") or c.get("plan_digest")!=plan_digest
                or c.get("verdict") not in SEMANTIC_REJECTS|{"complete","error_fail_closed"}):
            raise ValueError("OSS decision identity or verdict mismatch")
        if c["verdict"] in SEMANTIC_REJECTS and (c.get("coverage") is None or c.get("support") is None):
            raise ValueError("Fail-closed OSS result cannot enter gold salvage")
    queue=run/"codex_gold_review_oss_non_admissions"
    qm,gold_packets=validate_queue(queue)
    expected={rid for rid,c in decisions.items() if c["verdict"] in SEMANTIC_REJECTS}
    if (qm.get("source_plan_digest")!=plan_digest or qm.get("oss_summary_sha256")!=digest(out/"census_summary.json")
            or {p["review_id"] for p in gold_packets}!=expected or len(gold_packets)!=len(expected)):
        raise ValueError("Main gold queue differs from completed semantic rejects")
    errors={rid for rid,c in decisions.items() if c["verdict"]=="error_fail_closed"}
    terminal=json.loads((run/"quarantine/oss_fail_closed_terminal/manifest.json").read_text())
    if (terminal.get("source_plan_digest")!=plan_digest or terminal.get("count")!=len(errors)
            or {c["review_id"] for c in terminal.get("cases",[])}!=errors
            or terminal.get("retry_allowed") is not False or terminal.get("gold_salvage_allowed") is not False):
        raise ValueError("OSS terminal quarantine census mismatch")
    value={"schema":"giis_v17_turbo_main_pre_gold_v1","status":"oss_complete_gold_pending",
           "routing_revision":REVISION,"plan_digest":plan_digest,"population":len(packets),
           "semantic_rejects_for_gold":len(expected),"terminal_oss_errors":len(errors),
           "census_summary_sha256":digest(out/"census_summary.json"),
           "gold_queue_manifest_sha256":digest(queue/"queue_manifest.json"),
           "gold_review_authorized":False}
    write_bound(run/"main_pre_gold_manifest.json",value)
    return value


def heldout_authoring_census(run: Path, configuration: dict) -> tuple[dict,list[dict],list[str]]:
    run=Path(run).resolve()
    status=json.loads((run/"status.json").read_text())
    if status.get("review_routing_revision")!=REVISION:
        raise ValueError("Historical held-out routing; fresh namespace required")
    allocation=json.loads((run/"allocation.json").read_text())
    packets=read_rows(run/"review_packets.jsonl")
    expected=configuration["v17_turbo_heldout_allocation"]
    cases=allocation["cases"]
    planned={str(case["case_id"]):case for case in cases}
    successful={str(p["review_id"]):p for p in packets}
    terminal_dir=run/"quarantine/terminal_authoring_failures"
    terminal={path.stem:json.loads(path.read_text()) for path in terminal_dir.glob("*.json")}
    terminal_ids=sorted(terminal)
    if (status.get("review_routing_revision")!=REVISION or status.get("allocation_digest")!=allocation.get("allocation_digest")
            or len(cases)!=expected["case_count"] or status.get("total")!=len(cases) or len(planned)!=len(cases)
            or len(successful)!=len(packets) or len(packets)!=status.get("completed")
            or len(terminal)!=status.get("failed") or terminal_ids!=status.get("terminal_authoring_case_ids")
            or set(successful)&set(terminal) or set(successful)|set(terminal)!=set(planned)
            or any(p.get("lane")!=planned[rid]["candidate_type"] for rid,p in successful.items())
            or any(value.get("case_id")!=rid or value.get("status")!="terminal_quarantine"
                   or value.get("retry_allowed") is not False or value.get("repair_allowed") is not False
                   or value.get("semantic_review_allowed") is not False for rid,value in terminal.items())):
        raise ValueError("Held-out allocation is neither qualified nor terminally quarantined")
    expected_lanes=Counter(case["candidate_type"] for case in cases)
    if any(expected_lanes[lane]!=expected[lane+"_cases"] for lane in ("single_source","multi_source","table","comparison")):
        raise ValueError("Held-out frozen lane allocation drift")
    return status,packets,terminal_ids

def joint_checkpoint(main_run: Path, heldout_run: Path) -> dict:
    main_run=Path(main_run).resolve();heldout_run=Path(heldout_run).resolve()
    if main_run==heldout_run: raise ValueError("Main and held-out namespaces must differ")
    main=main_oss_ready(main_run)
    config=json.loads((main_run/"plan.json").read_text())["configuration"]
    status,source_packets,terminal_ids=heldout_authoring_census(heldout_run,config)
    held_q=heldout_run/"codex_gold_review_all_lanes"
    hmanifest,hpackets=validate_queue(held_q)
    if (len(hpackets)!=len(source_packets)
            or {p["review_id"] for p in hpackets}!={p["review_id"] for p in source_packets}
            or hmanifest.get("selection_digest")!=status.get("selection_digest")
            or hmanifest.get("allocation_digest")!=status.get("allocation_digest")
            or hmanifest.get("source_packet_sha256")!=digest(heldout_run/"review_packets.jsonl")
            or hmanifest.get("terminal_authoring_case_ids")!=terminal_ids):
        raise ValueError("Held-out gold queue differs from qualified authoring census")
    selection=json.loads((main_run.parents[0]/"splits/heldout_manifest.json").read_text())
    if status["selection_digest"]!=selection["selection_digest"]:
        raise ValueError("Held-out selection drift")
    plan=json.loads((main_run/"plan.json").read_text())
    training=set(plan["partitions"]["train"])|set(plan["partitions"]["validation"])
    held=set(selection["heldout_document_ids"])
    if training&held or any(not set(p["expected_document_ids"])<=held for p in hpackets):
        raise ValueError("Held-out source leakage")
    value={"schema":"giis_v17_turbo_joint_gold_checkpoint_v1","status":"awaiting_one_operator_gold_authorization",
           "routing_revision":REVISION,"main_run":str(main_run),"heldout_run":str(heldout_run),
           "main_plan_digest":main["plan_digest"],"main_pre_gold_manifest_sha256":digest(main_run/"main_pre_gold_manifest.json"),
           "main_gold_queue_manifest_sha256":main["gold_queue_manifest_sha256"],
           "heldout_gold_queue_manifest_sha256":digest(held_q/"queue_manifest.json"),
           "main_semantic_rejects":main["semantic_rejects_for_gold"],"heldout_candidates":len(hpackets),"heldout_terminal_authoring_failures":len(terminal_ids),
           "heldout_canonical_full_count_satisfied":not terminal_ids,
           "gold_review_authorized":False,"training_release":False}
    write_bound(main_run/"joint_gold_authorization_checkpoint.json",value)
    return value


def authorize_joint(main_run: Path, heldout_run: Path, statement: str) -> dict:
    if not statement.strip(): raise ValueError("Explicit operator statement required")
    checkpoint=joint_checkpoint(main_run,heldout_run)
    main_run=Path(main_run).resolve();heldout_run=Path(heldout_run).resolve()
    main_queue=main_run/"codex_gold_review_oss_non_admissions"
    held_queue=heldout_run/"codex_gold_review_all_lanes"
    for directory,expected in [(main_queue,checkpoint["main_gold_queue_manifest_sha256"]),(held_queue,checkpoint["heldout_gold_queue_manifest_sha256"])]:
        manifest,_=validate_queue(directory)
        if digest(directory/"queue_manifest.json")!=expected: raise ValueError("Joint authorization queue drift")
    authorize(main_queue,statement)
    authorize(held_queue,statement)
    value={"schema":"giis_v17_turbo_joint_gold_authorization_v1","authorized":True,
           "operator_statement":statement,"checkpoint_sha256":digest(main_run/"joint_gold_authorization_checkpoint.json"),
           "main_authorization_sha256":digest(main_queue/"authorization.json"),
           "heldout_authorization_sha256":digest(held_queue/"authorization.json")}
    write_bound(main_run/"joint_gold_authorization.json",value)
    return value


def cli():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["pre-gold","checkpoint","authorize"])
    parser.add_argument("--main",required=True,type=Path)
    parser.add_argument("--heldout",type=Path)
    parser.add_argument("--operator-statement",default="")
    args=parser.parse_args()
    if args.action=="pre-gold": result=main_oss_ready(args.main)
    elif args.action=="checkpoint":
        if args.heldout is None: parser.error("--heldout required")
        result=joint_checkpoint(args.main,args.heldout)
    else:
        if args.heldout is None: parser.error("--heldout required")
        result=authorize_joint(args.main,args.heldout,args.operator_statement)
    print(json.dumps(result,indent=2))
if __name__=="__main__":cli()
