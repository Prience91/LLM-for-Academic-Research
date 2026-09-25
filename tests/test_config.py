from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from giis_prod.config import Settings


class ConfigTests(unittest.TestCase):
    def test_typed_environment_override(self):
        with patch.dict(os.environ, {"GIIS__SERVING__MAX_MODEL_LEN": "16384"}, clear=False):
            settings = Settings.load()
        self.assertEqual(settings.get("serving.max_model_len"), 16384)

    def test_unknown_override_fails(self):
        with patch.dict(os.environ, {"GIIS__SERVING__NOT_A_SETTING": "1"}, clear=False):
            with self.assertRaises(KeyError):
                Settings.load()

    def test_teacher_dataset_targets(self):
        settings = Settings.load()
        self.assertEqual(settings.get("teacher.expected_training_documents"), 900)
        self.assertEqual(settings.get("teacher.minimum_core_examples_per_document"), 3)
        self.assertEqual(settings.get("teacher.maximum_core_examples_per_document"), 4)
        self.assertEqual(
            settings.get("teacher.maximum_zero_yield_requests_before_minimum"), 2
        )
        self.assertEqual(settings.get("teacher.initial_candidate_batch_size"), 4)
        self.assertEqual(settings.get("teacher.refill_candidate_batch_size"), 2)
        self.assertEqual(settings.get("teacher.temperature"), 0.30)
        self.assertEqual(settings.get("teacher.verification_batch_size"), 2)
        self.assertEqual(settings.get("teacher.verification_max_output_tokens"), 3000)
        self.assertEqual(settings.get("teacher.evidence_chunks_per_document"), 8)
        self.assertEqual(settings.get("teacher.minimum_substantive_chunks_per_document"), 4)
        self.assertEqual(settings.get("teacher.minimum_training_citation_coverage"), 0.40)
        self.assertEqual(settings.get("teacher.maximum_question_similarity"), 0.72)
        self.assertEqual(
            settings.get("teacher.maximum_question_embedding_similarity"), 0.88
        )
        self.assertEqual(settings.get("teacher.timed_smoke_max_documents"), 5)
        self.assertTrue(settings.get("teacher.timed_smoke_disable_multi_source"))
        self.assertEqual(settings.get("teacher.minimum_long_form_examples_per_document"), 0)
        self.assertEqual(settings.get("teacher.maximum_long_form_examples_per_document"), 2)
        self.assertTrue(settings.get("teacher.creative_answer_styles_enabled"))

    def test_mini_run_uses_1000_900_100_paper_level_plan(self):
        settings = Settings.load()
        self.assertEqual(settings.get("smoke_run_ii.expected_corpus_documents"), 1000)
        self.assertEqual(settings.get("smoke_run_ii.expected_sft_documents"), 900)
        self.assertEqual(settings.get("smoke_run_ii.heldout_documents"), 100)
        self.assertEqual(settings.get("smoke_run_ii.semantic_audit_maximum"), 400)
        self.assertEqual(settings.get("smoke_run_ii.semantic_audit_fraction"), 0.10)

    def test_active_v17_turbo_handoff_targets_fresh_qlora_release(self):
        settings = Settings.load()
        self.assertEqual(settings.get("paths.sft_dir"), "artifacts/v17_turbo/sft_release")
        self.assertEqual(
            settings.get("paths.training_selection_manifest"),
            "artifacts/v17_turbo/sft_release/training_selection.json",
        )
        self.assertEqual(
            settings.get("paths.semantic_audit_queue"),
            "artifacts/v17_turbo/sft_release/semantic_review_census.jsonl",
        )
        self.assertEqual(
            settings.get("paths.sft_exclusions"),
            "artifacts/v17_turbo/sft_release/semantic_exclusions.json",
        )
        self.assertTrue(settings.get("training.use_4bit"))
        tags = {tag.strip() for tag in settings.get("training.wandb_tags").split(",")}
        self.assertIn("qlora-nf4", tags)
        self.assertIn("v17-turbo", tags)

    def test_oom_safe_lora_retry_preserves_effective_batch_and_checkpoints(self):
        settings = Settings.load()
        train_batch = settings.get("training.train_batch_size")
        accumulation = settings.get("training.gradient_accumulation_steps")
        self.assertEqual(train_batch, 1)
        self.assertEqual(settings.get("training.eval_batch_size"), 1)
        self.assertEqual(accumulation, 16)
        self.assertEqual(train_batch * accumulation, 16)
        self.assertEqual(settings.get("training.logging_steps"), 1)
        self.assertEqual(settings.get("training.eval_steps"), 10)
        self.assertEqual(settings.get("training.save_steps"), 10)

    def test_metadata_enrichment_is_strict_and_network_enabled(self):
        settings = Settings.load()
        self.assertTrue(settings.get("metadata.enabled"))
        self.assertTrue(settings.get("metadata.allow_network"))
        self.assertFalse(settings.get("metadata.filename_is_curated_title"))
        self.assertTrue(settings.get("ingestion.require_verified_metadata"))

    def test_capacity_lanes_have_no_legacy_word_floor(self):
        settings = Settings.load()
        self.assertNotIn("minimum_long_form_words", settings.section("response"))
        self.assertFalse(settings.get("v17_turbo_capacity.word_target_enabled"))
        self.assertFalse(settings.get("v17_turbo_capacity.word_ceiling_enabled"))

    def test_deployment_citation_gate_remains_stricter_than_training(self):
        settings = Settings.load()
        self.assertEqual(settings.get("teacher.minimum_training_citation_coverage"), 0.40)
        self.assertEqual(settings.get("response.minimum_citation_coverage"), 0.80)

    def test_teacher_server_reserves_dgx_unified_memory_headroom(self):
        settings = Settings.load()
        self.assertEqual(settings.get("teacher_server.gpu_memory_utilization"), 0.78)

    def test_rag_corpus_directory_is_the_corpus_source(self):
        settings = Settings.load()
        self.assertEqual(settings.get("paths.pdf_dir"), "RAG Corpus")

    def test_training_heldout_directory_is_the_split_source(self):
        settings = Settings.load()
        self.assertEqual(settings.get("paths.training_pdf_dir"), "Training Corpus")

    def test_ingestion_is_native_text_first_with_bounded_ocr_fallback(self):
        settings = Settings.load()
        self.assertTrue(settings.get("ingestion.ocr_fallback_enabled"))
        self.assertEqual(settings.get("ingestion.chunk_overlap_tokens"), 64)
        self.assertEqual(settings.get("ingestion.native_text_minimum_characters"), 1000)
        self.assertEqual(settings.get("ingestion.native_text_minimum_chunks"), 2)
        self.assertEqual(settings.get("ingestion.maximum_rejected_document_fraction"), 0.10)

    def test_legacy_pdf_directory_is_migrated_to_canonical_source(self):
        with TemporaryDirectory() as temporary:
            config = Path(temporary) / "settings.toml"
            config.write_text('[paths]\npdf_dir = "data/pdfs"\n', encoding="utf-8")
            settings = Settings.load(config)
        self.assertEqual(settings.get("paths.pdf_dir"), "RAG Corpus")

    def test_previous_operator_directory_names_are_migrated(self):
        with TemporaryDirectory() as temporary:
            config = Path(temporary) / "settings.toml"
            config.write_text(
                '[paths]\npdf_dir = "Scientific papers"\ntraining_pdf_dir = "training set"\n',
                encoding="utf-8",
            )
            settings = Settings.load(config)
        self.assertEqual(settings.get("paths.pdf_dir"), "RAG Corpus")
        self.assertEqual(settings.get("paths.training_pdf_dir"), "Training Corpus")


if __name__ == "__main__":
    unittest.main()
