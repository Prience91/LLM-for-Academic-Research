import json
import tempfile
import unittest
from pathlib import Path
from giis_prod.v17_turbo_pre_gold import main_oss_ready, joint_checkpoint, authorize_joint
from giis_prod.v17_turbo_review_routing import REVISION, digest, write_bound, write_queue


def jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))

class JointPreGoldTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)/"artifacts/v17_turbo"
        self.main=self.base/"main";self.held=self.base/"held"
        self.main.mkdir(parents=True);self.held.mkdir()
        self.reject={"review_id":"r1","candidate_sha256":"a"*64,"lane":"single_source","partition":"train","plan_digest":"plan","document_ids":["train_doc"]}
        self.error={"review_id":"r2","candidate_sha256":"b"*64,"lane":"multi_source","partition":"validation","plan_digest":"plan","document_ids":["valid_doc"]}
        self.packets=[self.reject,self.error]
        plan={"plan_digest":"plan","configuration":{"v17_turbo_review":{"routing_revision":REVISION},"v17_turbo_heldout_allocation":{"case_count":1,"single_source_cases":1,"multi_source_cases":0,"table_cases":0,"comparison_cases":0}},"partitions":{"train":["train_doc"],"validation":["valid_doc"]}}
        write_bound(self.main/"plan.json",plan)
        out=self.main/"oss120b_all_lane_semantic_census"
        jsonl(out/"census_packets_blinded.jsonl",self.packets)
        source_sha=digest(out/"census_packets_blinded.jsonl")
        write_bound(out/"census_manifest.json",{"routing_revision":REVISION,"source_plan_digest":"plan","candidate_count":2,"packet_sha256":source_sha})
        cases=[dict(self.reject,verdict="unsupported",coverage="complete",support="unsupported"),dict(self.error,verdict="error_fail_closed",coverage=None,support=None)]
        write_bound(out/"census_summary.json",{"routing_revision":REVISION,"source_plan_digest":"plan","packet_sha256":source_sha,"total":2,"gold_selection":"completed_oss_semantic_rejects","cases":cases})
        self.gold=self.main/"codex_gold_review_oss_non_admissions"
        write_queue(self.gold,[self.reject],{"source_plan_digest":"plan","oss_summary_sha256":digest(out/"census_summary.json")})
        write_bound(self.main/"quarantine/oss_fail_closed_terminal/manifest.json",{"source_plan_digest":"plan","count":1,"cases":[cases[1]],"retry_allowed":False,"gold_salvage_allowed":False})
        (self.base/"splits").mkdir()
        write_bound(self.base/"splits/heldout_manifest.json",{"heldout_document_ids":["test_doc"],"selection_digest":"selection"})
        hp={"review_id":"h1","candidate_sha256":"c"*64,"lane":"single_source","expected_document_ids":["test_doc"]}
        jsonl(self.held/"review_packets.jsonl",[hp])
        write_bound(self.held/"allocation.json",{"allocation_digest":"allocation","cases":[{"case_id":"h1","candidate_type":"single_source"}]})
        write_bound(self.held/"status.json",{"review_routing_revision":REVISION,"total":1,"completed":1,"failed":0,"terminal_authoring_case_ids":[],"selection_digest":"selection","allocation_digest":"allocation"})
        self.held_gold=self.held/"codex_gold_review_all_lanes"
        write_queue(self.held_gold,[hp],{"selection_digest":"selection","allocation_digest":"allocation","source_packet_sha256":digest(self.held/"review_packets.jsonl"),"terminal_authoring_case_ids":[]})

    def test_one_checkpoint_keeps_both_gold_queues_unauthorized(self):
        checkpoint=joint_checkpoint(self.main,self.held)
        self.assertEqual(checkpoint["main_semantic_rejects"],1)
        self.assertEqual(checkpoint["heldout_candidates"],1)
        self.assertFalse((self.gold/"authorization.json").exists())
        self.assertFalse((self.held_gold/"authorization.json").exists())
        auth=authorize_joint(self.main,self.held,"Operator approves gold review of both frozen queues")
        self.assertTrue(auth["authorized"])
        self.assertTrue((self.gold/"authorization.json").exists())
        self.assertTrue((self.held_gold/"authorization.json").exists())

    def test_exhausted_oss_error_cannot_enter_gold_queue(self):
        (self.gold/"review_packets_blinded.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in self.packets))
        with self.assertRaises(ValueError):main_oss_ready(self.main)


    def test_terminal_heldout_case_does_not_block_other_gold_packets(self):
        plan=json.loads((self.main/"plan.json").read_text())
        plan["configuration"]["v17_turbo_heldout_allocation"]["case_count"]=2
        plan["configuration"]["v17_turbo_heldout_allocation"]["table_cases"]=1
        (self.main/"plan.json").write_text(json.dumps(plan))
        allocation=json.loads((self.held/"allocation.json").read_text())
        allocation["cases"].append({"case_id":"h2","candidate_type":"table"})
        (self.held/"allocation.json").write_text(json.dumps(allocation))
        terminal=self.held/"quarantine/terminal_authoring_failures/h2.json"
        write_bound(terminal,{"case_id":"h2","status":"terminal_quarantine","retry_allowed":False,"repair_allowed":False,"semantic_review_allowed":False})
        status=json.loads((self.held/"status.json").read_text())
        status.update(total=2,failed=1,terminal_authoring_case_ids=["h2"])
        (self.held/"status.json").write_text(json.dumps(status))
        queue=json.loads((self.held_gold/"queue_manifest.json").read_text())
        queue["terminal_authoring_case_ids"]=["h2"]
        (self.held_gold/"queue_manifest.json").write_text(json.dumps(queue))
        checkpoint=joint_checkpoint(self.main,self.held)
        self.assertEqual(checkpoint["heldout_candidates"],1)
        self.assertEqual(checkpoint["heldout_terminal_authoring_failures"],1)
        self.assertFalse(checkpoint["heldout_canonical_full_count_satisfied"])

    def test_missing_heldout_case_blocks_joint_checkpoint(self):
        status=json.loads((self.held/"status.json").read_text());status["failed"]=1
        (self.held/"status.json").write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError,"neither qualified nor terminally quarantined"):joint_checkpoint(self.main,self.held)
