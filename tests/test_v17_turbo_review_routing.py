import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from giis_prod import v17_turbo_review_routing as routing
from giis_prod import v17_turbo_review_packets as builder
from giis_prod import v17_turbo_oss_census as census
from giis_prod import v17_turbo_semantic_finalize as finalizer
from giis_prod import v17_turbo_retry as retry


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))

def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

def packet(rid="a", lane="single_source", partition="train"):
    return {"review_id":rid,"candidate_sha256":"h"+rid,"lane":lane,"partition":partition,
            "plan_digest":"plan","question":"Q","answer":"A","evidence":"E","teacher_declared_obligations":[]}

class RoutingTests(unittest.TestCase):
    def test_gold_requires_bound_authorization_and_complete_exact_census(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);p=packet()
            routing.write_queue(q,[p],{"scope":"training_validation_oss_non_admissions"})
            gold={"review_id":"a","candidate_sha256":"ha","lane":"single_source","reviewer":"codex_gold","verdict":"complete"}
            jsonl(q/"reviews.jsonl",[gold])
            with self.assertRaises(FileNotFoundError):routing.bind_gold(q)
            routing.authorize(q,"User authorized this exact gold review")
            self.assertEqual(routing.bind_gold(q)["reviewed"],1)
            oss={**p,"verdict":"unsupported","coverage":"complete","support":"unsupported"}
            self.assertEqual(routing.admission(p,oss=oss,gold=gold,gold_queue=q),"gold_salvaged_after_oss_non_admission")
            gold["candidate_sha256"]="foreign"
            with self.assertRaises(ValueError):routing.admission(p,oss=oss,gold=gold,gold_queue=q)

    def test_heldout_never_admitted_from_oss_and_queue_is_immutable(self):
        p=packet(partition="heldout")
        with self.assertRaisesRegex(ValueError,"gold complete"):
            routing.admission(p,oss={**p,"verdict":"complete"})
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);routing.write_queue(q,[p],{"scope":"heldout_all_generated"})
            with self.assertRaises(ValueError):routing.write_queue(q,[{**p,"answer":"changed"}],{"scope":"heldout_all_generated"})
            gold={"review_id":"a","candidate_sha256":"ha","lane":"single_source","reviewer":"codex_gold","verdict":"complete"}
            jsonl(q/"reviews.jsonl",[gold]);routing.authorize(q,"Approved heldout gold review")
            self.assertEqual(routing.admission(p,gold=gold,gold_queue=q),"gold_direct_complete")

    def test_main_machine_admission_requires_both_axes_and_identity(self):
        p=packet()
        oss={**p,"verdict":"complete","coverage":"complete","support":"supported"}
        self.assertEqual(routing.admission(p,oss=oss),"oss_direct_complete")
        for bad in [{**oss,"support":"unsupported"},{**oss,"partition":"validation"}]:
            with self.assertRaises(ValueError):routing.admission(p,oss=bad)

    def test_packet_builder_includes_all_lanes_and_inherits_creative_split(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/"artifacts/v17_turbo/new";run.mkdir(parents=True)
            def row(rid, kind, split):
                return {"example_id":rid,"plan_digest":"plan","partition":split,"dataset_kind":kind,
                        "document_ids":["d1","d2"],"bundle_id":"b",
                        "student_messages":[{"role":"user","content":"Evidence:\nEvidence body\n\nQuestion: Q"}],
                        "candidate":{"question":"Q","answer":"A","coverage_obligations":[]}}
            save(run/"passing_candidates/s.json",row("s","core_single_source","train"))
            save(run/"passing_candidates/m.json",row("m","multi_source_extra","validation"))
            creative=[{"case_id":lane,"source_example_id":"m","plan_digest":"plan","document_ids":["d1","d2"],"bundle_id":"b","answer_style":lane,"deterministic_review":{"passed":True},"question":"Q","candidate":{"answer":"A"},"required_coverage_obligations":[]} for lane in ["table","comparison"]]
            qp=run/"structured_qualification/manual_semantic_queue.jsonl";jsonl(qp,creative)
            save(run/"structured_qualification/manifest.json",{"plan_digest":"plan","status":"complete","case_count":2,"deterministic_passed":2,"manual_semantic_queue":{"path":str(qp),"row_count":2,"sha256":routing.digest(qp)}})
            save(run/"plan.json",{"plan_digest":"plan","bindings":{},"configuration":{"v17_turbo_review":{"routing_revision":routing.REVISION},"v17_turbo_structured_qualification":{"planned_total_candidates":2}}})
            save(run/"progress.json",{"plan_digest":"plan","python_passing_candidates":2})
            with patch.object(builder,"ROOT",root),patch.object(builder,"OUTPUT_ROOT",root/"artifacts/v17_turbo"):
                result=builder.prepare(run)
                self.assertEqual(result["candidate_count"],4)
                packets=routing.read_rows(run/"oss120b_all_lane_semantic_census/census_packets_blinded.jsonl")
                self.assertEqual({p["lane"] for p in packets},routing.LANES)
                self.assertTrue(all(p["partition"]=="validation" for p in packets if p["lane"] in {"table","comparison"}))

    def test_only_completed_semantic_rejects_enter_gold_errors_are_terminal(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td);out=run/"oss120b_all_lane_semantic_census";out.mkdir()
            packets=[packet("s"),packet("m","multi_source"),packet("t","table"),packet("c","comparison")]
            pp=out/"census_packets_blinded.jsonl";jsonl(pp,packets)
            save(run/"plan.json",{"plan_digest":"plan","configuration":{"v17_turbo_review":{"routing_revision":routing.REVISION}}})
            save(out/"census_manifest.json",{"source_plan_digest":"plan","packet_sha256":routing.digest(pp),"candidate_count":4})
            axes={}
            for p in packets[:3]:
                for axis in ["coverage","support"]:
                    decision="unsupported" if p["review_id"]=="t" and axis=="support" else "minor_omission" if p["review_id"]=="m" and axis=="coverage" else "complete" if axis=="coverage" else "supported"
                    axes[(p["review_id"],axis)]={"status":"ok","candidate_sha256":p["candidate_sha256"],"lane":p["lane"],"review":{axis:decision}}
            with patch.object(census,"latest",return_value=axes):summary=census.score(run,pp,out)
            self.assertEqual(summary["unresolved_errors"],1)
            queue=run/"codex_gold_review_oss_non_admissions"
            _,rows=routing.validate_queue(queue)
            self.assertEqual({p["review_id"] for p in rows},{"m","t"})
            self.assertTrue(all("verdict" not in p for p in rows))
            routing.authorize(queue,"Approved blinded reassessment of all non-admissions")
            jsonl(queue/"reviews.jsonl",[{"review_id":p["review_id"],"candidate_sha256":p["candidate_sha256"],"lane":p["lane"],"reviewer":"codex_gold","verdict":"complete"} for p in rows])
            completed=finalizer.bind_main(run)
            self.assertEqual(completed["population"],4)
            self.assertEqual(Counter(r["admission_path"] for r in completed["cases"]),{"oss_direct_complete":1,"gold_salvaged_after_oss_non_admission":2,None:1})
            self.assertTrue((run/"semantic_workflow_completion.json").is_file())
            terminal=next(r for r in completed["cases"] if r["review_id"]=="c")
            self.assertEqual(terminal["disposition"],"terminal_quarantine")
            self.assertFalse(terminal["repair_allowed"])
            self.assertTrue((run/"quarantine/oss_fail_closed_terminal/manifest.json").is_file())

    def test_author_failure_retried_once_and_never_reopened(self):
        with tempfile.TemporaryDirectory() as td:
            identity={"partition":"train","bundle_id":"b","request_number":1}
            calls=[]
            def failed(): calls.append(1); raise ValueError("malformed teacher JSON")
            with self.assertRaises(retry.TerminalGenerationFailure):retry.author_request(td,identity,failed)
            self.assertEqual(len(calls),2)
            with self.assertRaises(retry.TerminalGenerationFailure):retry.author_request(td,identity,failed)
            with self.assertRaises(retry.TerminalGenerationFailure):retry.author_request(td,{**identity,"request_number":2},failed)
            self.assertEqual(len(calls),2)

    def test_oss_failure_gets_only_one_retry_across_resumes(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"axes.jsonl";p=packet();calls=[]
            def failed(task):
                calls.append(1)
                record={"review_id":"a","candidate_sha256":"ha","axis":"coverage","status":"error"}
                with path.open("a") as f:f.write(json.dumps(record)+"\n")
                return record
            with patch.object(census.protocol,"RESULTS",path),patch.object(census.protocol,"review_axis",side_effect=failed):
                census.review_with_retries((p,"coverage"))
                census.review_with_retries((p,"coverage"))
            self.assertEqual(len(calls),2)
            with self.assertRaisesRegex(ValueError,"terminal"):
                routing.admission(p,oss={**p,"verdict":"error_fail_closed"},gold={"verdict":"complete"},gold_queue=Path(td))

    def test_gold_fail_closed_needs_disposition_authorization_and_has_no_retry(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);p=packet(partition="heldout")
            routing.write_queue(q,[p],{"scope":"heldout_all_generated"})
            routing.authorize(q,"Approved gold assessment")
            g={"review_id":"a","candidate_sha256":"ha","lane":"single_source","reviewer":"codex_gold","verdict":"error_fail_closed"}
            jsonl(q/"reviews.jsonl",[g])
            with self.assertRaisesRegex(ValueError,"awaits operator"):routing.bind_gold(q)
            disposition=routing.authorize_gold_failures(q,"Approved terminal archival","archive")
            self.assertFalse(disposition["retry_allowed"])
            self.assertFalse(disposition["repair_allowed"])
            self.assertEqual(routing.bind_gold(q)["counts"],{"error_fail_closed":1})

    def test_heldout_full_gold_queue_and_legacy_refusal(self):
        config={"v17_turbo_heldout_allocation":{"case_count":4,**{lane+"_cases":1 for lane in routing.LANES}}}
        with tempfile.TemporaryDirectory() as td:
            run=Path(td);status={"total":4,"completed":4,"failed":0,"selection_digest":"sel","allocation_digest":"alloc","review_routing_revision":routing.REVISION,"review_policy_fingerprint":"fingerprint","terminal_authoring_case_ids":[]}
            save(run/"status.json",status)
            save(run/"allocation.json",{"allocation_digest":"alloc","cases":[{"case_id":lane,"candidate_type":lane} for lane in sorted(routing.LANES)]})
            rows=[{**packet(lane,lane,"heldout"),"assigned_reviewer":"Codex gold"} for lane in sorted(routing.LANES)]
            jsonl(run/"review_packets.jsonl",rows)
            result=finalizer.prepare_heldout(run,config)
            self.assertEqual(result["packet_count"],4)
            queue=run/"codex_gold_review_all_lanes"
            routing.authorize(queue,"Approved full heldout gold review")
            jsonl(queue/"reviews.jsonl",[{"review_id":p["review_id"],"candidate_sha256":p["candidate_sha256"],"lane":p["lane"],"reviewer":"codex_gold","verdict":"complete"} for p in rows])
            completed=finalizer.bind_heldout(run)
            self.assertEqual(completed["population"],4)
            self.assertTrue(all(r["admission_path"]=="gold_direct_complete" for r in completed["cases"]))
            status.pop("review_routing_revision");save(run/"status.json",status)
            with self.assertRaisesRegex(ValueError,"Historical"):finalizer.prepare_heldout(run,config)

if __name__=="__main__":unittest.main()
