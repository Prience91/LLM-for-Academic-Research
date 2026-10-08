import inspect
import tomllib
import unittest
from pathlib import Path

from giis_prod.v17_turbo_builder import V17TurboDatasetBuilder
from giis_prod.heldout_authoring import heldout_case_allocation
from giis_prod.config import Settings
from giis_prod.v17_turbo_layer import SupportUnit, validate_structure
from giis_prod.v17_turbo_contract import (
    heldout_reviewer_for, semantic_review_plan, validate_heldout_benchmark_policy,
    validate_retrieval_index_policy,
)
from giis_prod.v17_turbo_canary import _validate_teacher_endpoint


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    def encode(self, answer, add_special_tokens=False):
        return list(range(int(answer)))


class _Settings:
    def get(self, key):
        return {
            "response.standard_max_tokens": 900,
            "response.long_form_max_tokens": 2200,
        }[key]


class V17TurboCapacityTests(unittest.TestCase):
    def builder(self):
        builder = object.__new__(V17TurboDatasetBuilder)
        builder.tokenizer = _Tokenizer()
        builder.settings = _Settings()
        return builder

    def test_capacity_boundaries_and_automatic_assignment(self):
        builder = self.builder()
        cases = [
            ("1", "standard", True),
            ("900", "standard", True),
            ("901", "long_form", True),
            ("2200", "long_form", True),
            ("2201", "standard", False),
        ]
        for answer, expected_mode, expected_valid in cases:
            with self.subTest(answer=answer):
                example = {"answer": answer, "response_mode": "standard"}
                valid, _ = builder._normalize_capacity_mode(example)
                self.assertEqual(valid, expected_valid)
                if expected_valid:
                    self.assertEqual(example["response_mode"], expected_mode)

    def test_teacher_endpoint_and_profile_port_must_match(self):
        valid = {"teacher": {"base_url": "http://127.0.0.1:8800/v1"}, "teacher_server": {"host": "127.0.0.1", "port": 8800}}
        _validate_teacher_endpoint(valid)
        invalid = {"teacher": {"base_url": "http://127.0.0.1:8800/v1"}, "teacher_server": {"host": "127.0.0.1", "port": 8200}}
        with self.assertRaisesRegex(ValueError, "endpoint/profile mismatch"):
            _validate_teacher_endpoint(invalid)

    def test_configuration_has_no_word_gate_and_uses_8192_student_context(self):
        with (ROOT / "config/settings_v17_turbo.toml").open("rb") as handle:
            config = tomllib.load(handle)
        self.assertEqual(config["v17_turbo"]["contract_revision"], 4)
        self.assertEqual(config["models"]["base"], "Qwen/Qwen3.8-27B")
        self.assertEqual(config["models"]["student"], "Qwen/Qwen3.8-27B")
        self.assertEqual(config["models"]["student_source"], "Qwen/Qwen3.8-27B")
        self.assertIn("Qwen3.5-122B-A10B-GPTQ-Int4", config["models"]["teacher_source"])
        self.assertEqual(config["paths"]["sft_dir"], "artifacts/v17_turbo/sft_release")
        self.assertIn("semantic_review_census.jsonl", config["paths"]["semantic_audit_queue"])
        self.assertIn("semantic_exclusions.json", config["paths"]["sft_exclusions"])
        self.assertNotIn("sft_challenger_16k2", config["paths"]["sft_exclusions"])
        self.assertEqual(config["training"]["max_length"], 8192)
        self.assertEqual(config["response"]["standard_max_tokens"], 900)
        self.assertEqual(config["response"]["long_form_max_tokens"], 2200)
        self.assertNotIn("minimum_long_form_words", config["response"])
        self.assertFalse(config["v17_turbo_capacity"]["word_target_enabled"])
        self.assertFalse(config["v17_turbo_capacity"]["word_ceiling_enabled"])

    def test_retrieval_index_policy_is_fresh_and_scalable(self):
        with (ROOT / "config/settings_v17_turbo.toml").open("rb") as handle:
            config = tomllib.load(handle)
        policy = config["v17_turbo_retrieval_index"]
        self.assertEqual(policy["maximum_corpus_documents"], 20000)
        self.assertFalse(policy["runtime_dependency_on_earlier_policy"])
        self.assertEqual(config["paths"]["pdf_dir"], "RAG Corpus")
        self.assertEqual(config["v17_turbo"]["corpus_documents"], 1000)
        self.assertEqual(config["v17_turbo"]["frozen_heldout_documents"], 100)
        self.assertEqual(config["v17_turbo"]["generation_documents"], 900)
        active_index = Path(config["paths"]["active_index"])
        self.assertTrue(active_index.as_posix().startswith("artifacts/v17_turbo/index/"))

    def test_heldout_allocation_matches_planned_candidate_mix_and_covers_all_papers(self):
        with (ROOT / "config/settings_v17_turbo.toml").open("rb") as handle:
            config = tomllib.load(handle)
        settings = Settings(root=ROOT, data=config)
        document_ids = [f"doc_{index:03d}" for index in range(100)]
        cases = heldout_case_allocation(document_ids, settings)
        counts = {
            lane: sum(case["candidate_type"] == lane for case in cases)
            for lane in ("single_source", "multi_source", "table", "comparison")
        }
        self.assertEqual(counts, {
            "single_source": 71, "multi_source": 25, "table": 2, "comparison": 2,
        })
        self.assertEqual(
            {document_id for case in cases for document_id in case["document_ids"]},
            set(document_ids),
        )
        self.assertTrue(all(
            len(case["document_ids"]) == (1 if case["candidate_type"] == "single_source" else 2)
            for case in cases
        ))

    def test_heldout_benchmark_author_and_reviewer_routing(self):
        with (ROOT / "config/settings_v17_turbo.toml").open("rb") as handle:
            config = tomllib.load(handle)
        policy = config["v17_turbo_heldout_benchmark"]
        frozen = validate_heldout_benchmark_policy(policy)
        self.assertEqual(
            frozen["authoring_model"],
            "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        )
        self.assertEqual(heldout_reviewer_for("single_source", policy), "Codex gold")
        for candidate_type in ("multi_source", "table", "comparison"):
            self.assertEqual(heldout_reviewer_for(candidate_type, policy), "Codex gold")
        self.assertTrue(frozen["reference_answers_hidden_from_student"])
        self.assertTrue(frozen["training_feedback_prohibited"])
        self.assertTrue(frozen["one_shot_final_evaluation"])

    def test_coverage_contract_requires_complete_mapping(self):
        units = [SupportUnit("[E1]:S001", "[E1]", "Supported fact.")]
        candidate = {
            "answer_style": "prose",
            "coverage_obligations": [{
                "obligation_id": "O1", "requirement": "State fact",
                "supports": [{"evidence_id": "[E1]", "support_unit_ids": ["[E1]:S001"]}],
            }],
            "answer_units": [{
                "answer_unit_id": "AU1", "obligation_ids": ["O1"],
                "text": "Supported fact [E1].",
            }],
        }
        self.assertTrue(validate_structure(candidate, units)["passed"])
        candidate["answer_units"][0]["obligation_ids"] = []
        self.assertIn("unmapped_obligation", validate_structure(candidate, units)["reasons"])

    def test_main_semantic_routing_is_all_lane_oss_then_rejected_gold(self):
        single = [{"example_id": "s1", "candidate": {"answer_style": "prose"}}]
        multi = [
            {"example_id": "m1", "candidate": {"answer_style": "prose"}},
            {"example_id": "t1", "candidate": {"answer_style": "table"}},
        ]
        plan = semantic_review_plan(single, multi, seed=3407)
        self.assertEqual({row["example_id"] for row in plan["all_lane_machine_census"]}, {"s1", "m1", "t1"})
        self.assertEqual(plan["gold_selection"], "completed_oss_semantic_rejects")
        self.assertTrue(plan["gold_review_requires_explicit_authorization"])
        self.assertEqual(plan["reviewer_ownership"]["all_lane_machine_census"], "GPT-OSS-120B")

    def test_turbo_modules_do_not_import_earlier_policy_modules(self):
        files = [
            "v17_turbo_layer.py", "v17_turbo_builder.py", "v17_turbo_contract.py",
            "v17_turbo_qualification.py", "v17_turbo_canary.py",
        ]
        forbidden = ("from .v16", "from .v17_builder", "from .v17_layer", "from .v17_contract", "from .v17_qualification")
        for name in files:
            text = (ROOT / "src/giis_prod" / name).read_text(encoding="utf-8")
            for fragment in forbidden:
                self.assertNotIn(fragment, text, f"{name} imports {fragment}")
        builder_text = (ROOT / "src/giis_prod/v17_turbo_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("long_form_too_short", builder_text)
        self.assertNotIn("standard_answer_too_long", builder_text)


if __name__ == "__main__":
    unittest.main()


class TerminalFailureContinuationTests(unittest.TestCase):
    def test_exhausted_qwen_request_quarantines_bundle_without_raising(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from giis_prod.v17_turbo_retry import TerminalGenerationFailure
        builder=object.__new__(V17TurboDatasetBuilder)
        builder.settings=SimpleNamespace(path=lambda key:Path("/tmp/v17_test_work"))
        builder._terminal_generation_bundles=set()
        events=[]
        builder._audit=lambda event,**fields:events.append((event,fields))
        bundle=SimpleNamespace(bundle_id="bundle_failed",document_ids=("doc_failed",))
        with patch("giis_prod.v17_turbo_builder.author_request",side_effect=TerminalGenerationFailure("exhausted")):
            result=builder._request_examples_with_retries(bundle,partition="train",request_number=1)
        self.assertEqual(result,[])
        self.assertEqual(builder._terminal_generation_bundles,{"bundle_failed"})
        self.assertEqual(events[0][0],"terminal_authoring_request_quarantined")

    def test_creative_qualification_uses_available_bundles_after_shortfall(self):
        from giis_prod.v17_turbo_contract import qualification_cases
        source={"dataset_kind":"multi_source_extra","v17_turbo_python_passed":True,
                "candidate":{"answer_style":"prose"},"document_ids":["d1","d2"],
                "bundle_id":"b1","example_id":"e1"}
        with self.assertRaises(ValueError):
            qualification_cases([source],seed=1,bundle_count=2)
        cases=qualification_cases([source],seed=1,bundle_count=2,allow_underfilled=True)
        self.assertEqual({row["answer_style"] for row in cases},{"table","comparison"})

    def test_terminal_underfloor_paper_does_not_abort_other_work(self):
        from types import SimpleNamespace, MethodType
        from unittest.mock import patch
        builder=object.__new__(V17TurboDatasetBuilder)
        builder.teacher={"seed":1,"minimum_core_examples_per_document":2,
                         "underfilled_release_floor":2,"maximum_core_examples_per_document":4,
                         "maximum_core_requests_per_document":3,
                         "maximum_zero_yield_requests_before_minimum":2,
                         "initial_candidate_batch_size":4,"refill_candidate_batch_size":2,
                         "generation_concurrency":1}
        builder._terminal_generation_bundles=set()
        builder._audit=lambda *args,**kwargs:None
        bundle=SimpleNamespace(bundle_id="b1",document_ids=("d1",),chunks=[])
        def exhausted(self,bundle,*args,**kwargs):
            self._terminal_generation_bundles.add(bundle.bundle_id)
            return []
        builder._request_examples_with_retries=MethodType(exhausted,builder)
        with patch("giis_prod.teacher.build_bundles",return_value=[bundle]):
            accepted,rejected=builder._records_for_partition("train",{"d1"},[],checkpoint_digest=None,multi_source_fraction=0)
        self.assertEqual((accepted,rejected),([],[]))
