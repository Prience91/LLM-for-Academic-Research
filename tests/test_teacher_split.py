from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from giis_prod.config import Settings
from giis_prod.io import stable_digest, write_json_atomic, write_jsonl_atomic
from giis_prod.production_readiness import ProductionReadinessError
from giis_prod.teacher import (
    DatasetAuditJournal,
    EvidenceBundle,
    TeacherDatasetBuilder,
    TeacherInfrastructureError,
    _cited_document_ids,
    _claim_units,
    _is_substantive_chunk,
    _question_similarity,
    _render_grounded_segments,
    build_bundles,
    select_training_documents,
    split_document_ids,
)


class TeacherSplitTests(unittest.TestCase):
    @staticmethod
    def _selection_settings(root: Path) -> Settings:
        return Settings(
            root=root,
            data={
                "paths": {
                    "training_pdf_dir": "Papers_Training",
                    "document_registry": "artifacts/corpus/documents.jsonl",
                    "training_selection_manifest": "artifacts/sft/training_selection.json",
                    "benchmark_file": "benchmark/benchmark.jsonl",
                },
                "teacher": {"expected_training_documents": 2},
            },
        )

    def test_hand_picked_training_files_match_registry_by_hash(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dir = root / "Papers_Training"
            training_dir.mkdir()
            first = training_dir / "renamed-a.pdf"
            second = training_dir / "b.pdf"
            first.write_bytes(b"paper-a")
            second.write_bytes(b"paper-b")
            from giis_prod.io import sha256_file

            registry = [
                {
                    "document_id": "d1",
                    "sha256": sha256_file(first),
                    "source_file": "original-a.pdf",
                    "title": "A",
                    "theme": "climate",
                    "metadata_status": "verified",
                },
                {
                    "document_id": "d2",
                    "sha256": sha256_file(second),
                    "source_file": "b.pdf",
                    "title": "B",
                    "theme": "ecology",
                    "metadata_status": "provisional",
                },
            ]
            settings = self._selection_settings(root)
            write_jsonl_atomic(settings.path("paths.document_registry"), registry)

            selected, manifest = select_training_documents(settings)

            self.assertEqual({row["document_id"] for row in selected}, {"d1", "d2"})
            self.assertEqual(manifest["selected_document_count"], 2)
            self.assertTrue(settings.path("paths.training_selection_manifest").exists())

    def test_training_file_absent_from_complete_corpus_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dir = root / "Papers_Training"
            training_dir.mkdir()
            (training_dir / "a.pdf").write_bytes(b"paper-a")
            (training_dir / "not-in-corpus.pdf").write_bytes(b"paper-x")
            settings = self._selection_settings(root)
            write_jsonl_atomic(
                settings.path("paths.document_registry"),
                [
                    {
                        "document_id": "d1",
                        "sha256": hashlib.sha256(b"paper-a").hexdigest(),
                        "source_file": "a.pdf",
                        "title": "A",
                        "theme": None,
                        "metadata_status": "verified",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "not present byte-for-byte"):
                select_training_documents(settings)

    def test_benchmark_document_cannot_enter_training_selection(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dir = root / "Papers_Training"
            training_dir.mkdir()
            first = training_dir / "a.pdf"
            second = training_dir / "b.pdf"
            first.write_bytes(b"paper-a")
            second.write_bytes(b"paper-b")
            from giis_prod.io import sha256_file

            settings = self._selection_settings(root)
            write_jsonl_atomic(
                settings.path("paths.document_registry"),
                [
                    {
                        "document_id": "d1",
                        "sha256": sha256_file(first),
                        "source_file": "a.pdf",
                        "title": "A",
                        "theme": None,
                        "metadata_status": "verified",
                    },
                    {
                        "document_id": "d2",
                        "sha256": sha256_file(second),
                        "source_file": "b.pdf",
                        "title": "B",
                        "theme": None,
                        "metadata_status": "verified",
                    },
                ],
            )
            write_jsonl_atomic(
                settings.path("paths.benchmark_file"),
                [{"case_id": "held-out", "expected_document_ids": ["d2"]}],
            )
            with self.assertRaisesRegex(ValueError, "benchmark-held-out"):
                select_training_documents(settings)

    def test_document_split_has_no_leakage(self):
        documents = [
            {"document_id": f"d{index}", "theme": "climate" if index < 6 else "ecology"}
            for index in range(10)
        ]
        train, validation = split_document_ids(documents, validation_fraction=0.2, seed=42)
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, {row["document_id"] for row in documents})

    def test_teacher_evidence_filter_excludes_non_substantive_sections(self):
        base = {
            "document_id": "d1",
            "page_start": 1,
            "page_end": 1,
            "ordinal": 0,
            "chunk_id": "c1",
            "title": "Paper title",
        }
        self.assertFalse(
            _is_substantive_chunk({**base, "section": "References", "text": "Citation list"})
        )
        self.assertFalse(
            _is_substantive_chunk(
                {
                    **base,
                    "section": "C R E D i T authorship contribution statement",
                    "text": "Roles",
                }
            )
        )
        self.assertFalse(
            _is_substantive_chunk({**base, "section": "P. Läderach", "text": "Affiliation"})
        )
        self.assertFalse(
            _is_substantive_chunk(
                {**base, "section": "Paper title", "text": "Author names and affiliations"}
            )
        )
        self.assertTrue(
            _is_substantive_chunk(
                {**base, "section": "Paper title", "text": "Abstract Substantive findings"}
            )
        )
        self.assertTrue(
            _is_substantive_chunk({**base, "section": "Results", "text": "Findings"})
        )

    def test_question_similarity_detects_close_paraphrases(self):
        first = "How did the authors calculate district COVID-19 potential outreach?"
        paraphrase = "How was a district's COVID-19 potential outreach calculated by the authors?"
        distinct = "Which regions were classified as having low epidemic susceptibility?"
        self.assertGreaterEqual(_question_similarity(first, paraphrase), 0.60)
        self.assertLess(_question_similarity(first, distinct), 0.60)

    def test_multisource_citations_must_resolve_to_distinct_documents(self):
        bundle = EvidenceBundle(
            bundle_id="multi",
            document_ids=("d1", "d2"),
            chunks=[
                {"document_id": "d1"},
                {"document_id": "d1"},
                {"document_id": "d2"},
            ],
        )
        self.assertEqual(_cited_document_ids(bundle, "Claim [E1][E2]."), {"d1"})
        self.assertEqual(_cited_document_ids(bundle, "Synthesis [E1][E3]."), {"d1", "d2"})

    def test_multisource_bundles_stay_inside_allowed_partition(self):
        allowed = {"d1", "d2"}
        chunks = []
        for document_id in ["d1", "d2", "d3"]:
            for ordinal in range(6):
                chunks.append(
                    {
                        "chunk_id": f"{document_id}-{ordinal}",
                        "document_id": document_id,
                        "page_start": ordinal + 1,
                        "ordinal": ordinal,
                        "text": "text",
                        "title": "title",
                    }
                )
        bundles = build_bundles(chunks, allowed, multi_source_fraction=1.0, seed=7)
        self.assertTrue(bundles)
        self.assertTrue(all(set(bundle.document_ids) <= allowed for bundle in bundles))

    def test_multisource_examples_do_not_reduce_core_allocation(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "seed": 7,
            "multi_source_fraction": 1.0,
            "minimum_core_examples_per_document": 4,
            "maximum_core_examples_per_document": 4,
            "adaptive_stop_after_empty_requests": 2,
            "maximum_question_similarity": 1.01,
        }

        class FakeTokenizer:
            @staticmethod
            def apply_chat_template(messages, **kwargs):
                return list(range(len(messages)))

        builder.tokenizer = FakeTokenizer()
        request_number = 0

        def request_examples(bundle, accepted_questions=None, **kwargs):
            nonlocal request_number
            request_number += 1
            markers = "[E1]"
            if len(bundle.document_ids) > 1:
                first_by_document = {}
                for index, chunk in enumerate(bundle.chunks, start=1):
                    first_by_document.setdefault(chunk["document_id"], f"[E{index}]")
                markers = "".join(first_by_document.values())
            return [
                {
                    "question": f"Focused question {request_number}",
                    "answer": f"Supported answer {markers}",
                    "response_mode": "standard",
                },
                {
                    "question": f"Synthesis question {request_number}",
                    "answer": f"Supported synthesis {markers}",
                    "response_mode": "long_form",
                },
            ]

        builder._request_examples = request_examples
        builder._validate_example = lambda bundle, example: (True, "ok")
        chunks = []
        for document_id in ["d1", "d2"]:
            for ordinal in range(6):
                chunks.append(
                    {
                        "chunk_id": f"{document_id}-{ordinal}",
                        "document_id": document_id,
                        "page_start": ordinal + 1,
                        "page_end": ordinal + 1,
                        "ordinal": ordinal,
                        "text": "evidence text",
                        "title": "title",
                    }
                )

        accepted, rejected = builder._records_for_partition(
            "train", {"d1", "d2"}, chunks
        )
        core = [row for row in accepted if row["dataset_kind"] == "core_single_source"]
        extras = [row for row in accepted if row["dataset_kind"] == "multi_source_extra"]
        counts = Counter(row["document_ids"][0] for row in core)
        self.assertEqual(counts, {"d1": 4, "d2": 4})
        self.assertTrue(extras)
        self.assertTrue(all(len(row["document_ids"]) == 2 for row in extras))
        self.assertEqual(rejected, [])

    def test_adaptive_core_allocation_stops_immediately_after_minimum(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "response": {"minimum_long_form_words": 350},
                    "training": {"max_length": 8192},
                },
            )
            builder.teacher = {
                "seed": 7,
                "multi_source_fraction": 0.0,
                "minimum_core_examples_per_document": 2,
                "maximum_core_examples_per_document": 4,
                "adaptive_stop_after_empty_requests": 2,
                "maximum_core_requests_per_document": 10,
                "maximum_question_similarity": 1.01,
            }

            class FakeTokenizer:
                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return list(range(len(messages)))

            builder.tokenizer = FakeTokenizer()
            request_number = 0
            requested_batch_sizes = []

            def request_examples(bundle, accepted_questions=None, **kwargs):
                nonlocal request_number
                request_number += 1
                requested_batch_sizes.append(kwargs["requested_examples"])
                prefix = "Accepted" if request_number == 1 else "Rejected"
                return [
                    {
                        "question": f"{prefix} question {request_number}-{index}",
                        "answer": "Supported answer [E1].",
                        "response_mode": "standard",
                    }
                    for index in (1, 2)
                ]

            builder._request_examples = request_examples
            builder._validate_example = lambda bundle, example: (
                not example["question"].startswith("Rejected"),
                "synthetic_rejection",
            )
            chunks = [
                {
                    "chunk_id": f"d1-{ordinal}",
                    "document_id": "d1",
                    "page_start": ordinal + 1,
                    "page_end": ordinal + 1,
                    "ordinal": ordinal,
                    "text": "evidence text",
                    "title": "title",
                }
                for ordinal in range(6)
            ]

            accepted, rejected = builder._records_for_partition("train", {"d1"}, chunks)

            self.assertEqual(len(accepted), 2)
            self.assertEqual(len(rejected), 0)
            self.assertEqual(request_number, 1)
            self.assertEqual(requested_batch_sizes, [4])

    def test_underfilled_document_stops_after_two_zero_yield_requests(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "response": {"minimum_long_form_words": 350},
                    "training": {"max_length": 8192},
                },
            )
            builder.teacher = {
                "seed": 7,
                "multi_source_fraction": 0.0,
                "minimum_core_examples_per_document": 2,
                "maximum_core_examples_per_document": 4,
                "adaptive_stop_after_empty_requests": 2,
                "maximum_zero_yield_requests_before_minimum": 2,
                "maximum_core_requests_per_document": 100,
                "maximum_question_similarity": 1.01,
            }

            class FakeTokenizer:
                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return list(range(len(messages)))

            builder.tokenizer = FakeTokenizer()
            request_number = 0

            def request_examples(bundle, accepted_questions=None, **kwargs):
                nonlocal request_number
                request_number += 1
                return [
                    {
                        "question": f"Rejected question {request_number}-{index}",
                        "answer": "Unsupported answer [E1].",
                        "response_mode": "standard",
                    }
                    for index in (1, 2)
                ]

            builder._request_examples = request_examples
            builder._validate_example = lambda bundle, example: (
                False,
                "synthetic_rejection",
            )
            chunks = [
                {
                    "chunk_id": f"d1-{ordinal}",
                    "document_id": "d1",
                    "page_start": ordinal + 1,
                    "page_end": ordinal + 1,
                    "ordinal": ordinal,
                    "text": "evidence text",
                    "title": "title",
                }
                for ordinal in range(6)
            ]

            with self.assertRaisesRegex(RuntimeError, "minimum 2 core examples"):
                builder._records_for_partition("train", {"d1"}, chunks)

            self.assertEqual(request_number, 2)

    def test_core_generation_resumes_from_document_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "models": {"teacher": "teacher", "base": "base"},
                    "response": {"minimum_long_form_words": 700},
                    "training": {"max_length": 8192},
                },
            )

            class FakeTokenizer:
                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return list(range(len(messages)))

            def make_builder(start: int):
                builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
                builder.settings = settings
                builder.teacher = {
                    "seed": 7,
                    "multi_source_fraction": 0.0,
                    "minimum_core_examples_per_document": 4,
                    "maximum_core_examples_per_document": 4,
                    "adaptive_stop_after_empty_requests": 2,
                    "maximum_core_requests_per_document": 1,
                    "maximum_question_similarity": 1.01,
                    "temperature": 0.3,
                    "max_output_tokens": 5000,
                }
                builder.tokenizer = FakeTokenizer()

                def request_examples(bundle, accepted_questions=None, **kwargs):
                    self.assertEqual(len(accepted_questions or []), start)
                    return [
                        {
                            "question": f"Question {start + index}",
                            "answer": "Supported answer [E1]",
                            "response_mode": "standard",
                        }
                        for index in (1, 2)
                    ]

                builder._request_examples = request_examples
                builder._validate_example = lambda bundle, example: (True, "ok")
                return builder

            chunks = [
                {
                    "chunk_id": f"d1-{ordinal}",
                    "document_id": "d1",
                    "page_start": ordinal + 1,
                    "page_end": ordinal + 1,
                    "ordinal": ordinal,
                    "text": "evidence text",
                    "title": "title",
                }
                for ordinal in range(6)
            ]
            with self.assertRaisesRegex(RuntimeError, "minimum 4 core examples"):
                make_builder(0)._records_for_partition(
                    "train", {"d1"}, chunks, checkpoint_digest="selection"
                )

            checkpoint_files = list(
                (root / "artifacts" / "sft" / "checkpoints").rglob("*.json")
            )
            self.assertEqual(len(checkpoint_files), 1)

            accepted, rejected = make_builder(2)._records_for_partition(
                "train", {"d1"}, chunks, checkpoint_digest="selection"
            )
            self.assertEqual(len(accepted), 4)
            self.assertEqual(rejected, [])

    def test_v12_checkpoint_is_not_imported_into_v13_structured_evidence_policy(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "models": {"teacher": "teacher", "base": "base"},
                    "response": {"minimum_long_form_words": 350},
                    "training": {"max_length": 8192},
                },
            )
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = settings
            builder.teacher = {
                "temperature": 0.3,
                "max_output_tokens": 5000,
                "minimum_core_examples_per_document": 6,
                "maximum_core_examples_per_document": 10,
                "adaptive_stop_after_empty_requests": 2,
            }

            class FakeTokenizer:
                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return list(range(len(messages)))

            builder.tokenizer = FakeTokenizer()
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Grounded evidence.",
                    }
                ],
            )
            question = "What did the paper report?"
            answer = "The paper reported grounded evidence [E1]."
            record = {
                "example_id": "sft_" + stable_digest(bundle.bundle_id, question, answer),
                "split": "train",
                "dataset_kind": "core_single_source",
                "document_ids": ["document-1"],
                "bundle_id": bundle.bundle_id,
                "response_mode": "standard",
                "messages": builder._training_messages(bundle, question, answer),
                "token_count": 3,
            }
            legacy_fingerprint = builder._checkpoint_policy_fingerprint(
                generation_policy_version=12, minimum_long_form_words=350
            )
            legacy_path = builder._core_checkpoint_path(
                "selection",
                "train",
                "document-1",
                policy_fingerprint=legacy_fingerprint,
            )
            write_json_atomic(
                legacy_path,
                {
                    "checkpoint_version": 1,
                    "selection_digest": "selection",
                    "policy_fingerprint": legacy_fingerprint,
                    "partition": "train",
                    "document_id": "document-1",
                    "bundle_id": bundle.bundle_id,
                    "target": 10,
                    "accepted": [record],
                    "accepted_questions": [question],
                    "rejected": [],
                    "requests_completed": 1,
                },
            )

            state = builder._load_core_checkpoint(
                "selection", "train", "document-1", bundle, 6, 10
            )

            self.assertEqual(state["accepted"], [])
            self.assertFalse(
                builder._core_checkpoint_path("selection", "train", "document-1").exists()
            )

    def test_underfilled_checkpoint_gets_a_new_zero_yield_budget_on_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "models": {"teacher": "teacher", "base": "base"},
                    "response": {"minimum_long_form_words": 350},
                    "training": {"max_length": 8192},
                },
            )
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = settings
            builder.teacher = {
                "temperature": 0.3,
                "max_output_tokens": 5000,
                "minimum_core_examples_per_document": 6,
                "maximum_core_examples_per_document": 10,
                "maximum_zero_yield_requests_before_minimum": 2,
            }
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Grounded evidence.",
                    }
                ],
            )
            path = builder._core_checkpoint_path(
                "selection", "validation", "document-1"
            )
            write_json_atomic(
                path,
                {
                    "checkpoint_version": 1,
                    "selection_digest": "selection",
                    "policy_fingerprint": builder._checkpoint_policy_fingerprint(),
                    "partition": "validation",
                    "document_id": "document-1",
                    "bundle_id": "bundle-1",
                    "minimum": 6,
                    "target": 10,
                    "accepted": [],
                    "accepted_questions": [],
                    "rejected": [{"reason": "missing_evidence_markers"}],
                    "requests_completed": 2,
                    "consecutive_empty_requests": 2,
                    "adaptive_complete": False,
                },
            )

            state = builder._load_core_checkpoint(
                "selection", "validation", "document-1", bundle, 6, 10
            )

            self.assertEqual(state["consecutive_empty_requests"], 0)
            self.assertEqual(state["requests_completed"], 2)
            self.assertEqual(state["rejected"], [{"reason": "missing_evidence_markers"}])

    def test_exhausted_checkpoint_at_release_floor_is_complete_on_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "models": {"teacher": "teacher", "base": "base"},
                    "response": {"minimum_long_form_words": 350},
                    "training": {"max_length": 8192},
                },
            )
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = settings
            builder.teacher = {
                "temperature": 0.3,
                "max_output_tokens": 5000,
                "minimum_core_examples_per_document": 6,
                "maximum_core_examples_per_document": 10,
                "underfilled_release_floor": 4,
                "maximum_zero_yield_requests_before_minimum": 2,
            }
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Grounded evidence.",
                    }
                ],
            )
            accepted = []
            questions = []
            for index in range(4):
                question = f"Question {index}?"
                answer = "Grounded answer [E1]."
                messages = builder._training_messages(bundle, question, answer)
                accepted.append(
                    {
                        "example_id": "sft_"
                        + stable_digest(bundle.bundle_id, question, answer),
                        "split": "train",
                        "dataset_kind": "core_single_source",
                        "document_ids": ["document-1"],
                        "bundle_id": "bundle-1",
                        "response_mode": "standard",
                        "answer_style": "prose",
                        "messages": messages,
                        "token_count": 1,
                    }
                )
                questions.append(question)
            path = builder._core_checkpoint_path("selection", "train", "document-1")
            write_json_atomic(
                path,
                {
                    "checkpoint_version": 1,
                    "selection_digest": "selection",
                    "policy_fingerprint": builder._checkpoint_policy_fingerprint(),
                    "partition": "train",
                    "document_id": "document-1",
                    "bundle_id": "bundle-1",
                    "minimum": 6,
                    "target": 10,
                    "accepted": accepted,
                    "accepted_questions": questions,
                    "rejected": [],
                    "requests_completed": 3,
                    "consecutive_empty_requests": 2,
                    "adaptive_complete": False,
                },
            )
            builder._validate_example = lambda bundle, example: (True, "ok")

            state = builder._load_core_checkpoint(
                "selection", "train", "document-1", bundle, 6, 10
            )

            self.assertTrue(state["adaptive_complete"])
            self.assertEqual(state["consecutive_empty_requests"], 2)
            self.assertEqual(len(state["accepted"]), 4)

    def test_marker_validation_distinguishes_missing_and_out_of_range(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "response": {"minimum_long_form_words": 350},
                    "training": {"max_length": 8192},
                },
            )
            builder.teacher = {
                "minimum_training_citation_coverage": 0.80,
                "maximum_question_similarity": 1.01,
                "maximum_question_embedding_similarity": 1.01,
            }

            class FakeTokenizer:
                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return list(range(len(messages)))

            builder.tokenizer = FakeTokenizer()
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Grounded evidence.",
                    }
                ],
            )
            missing = {
                "question": "Question?",
                "answer": "Answer without a citation.",
                "response_mode": "standard",
            }
            out_of_range = {
                "question": "Question?",
                "answer": "Answer with an unavailable citation [E2].",
                "response_mode": "standard",
            }
            malformed = {
                "question": "Question?",
                "answer": (
                    "This sufficiently detailed factual sentence uses a malformed citation "
                    "[E1, E2]."
                ),
                "response_mode": "standard",
            }
            insufficient_coverage = {
                "question": "Question?",
                "answer": (
                    "This first factual sentence contains enough words but has no citation. "
                    "This second factual sentence contains enough words and has one [E1]."
                ),
                "response_mode": "standard",
            }
            secondary_citation = {
                "question": "What was reported?",
                "answer": "Smith et al. (2020) reported the finding [E1].",
                "response_mode": "standard",
            }

            self.assertEqual(
                builder._validate_example(bundle, missing),
                (False, "missing_evidence_markers"),
            )
            self.assertEqual(
                builder._validate_example(bundle, out_of_range),
                (False, "out_of_range_evidence_markers"),
            )
            self.assertEqual(
                builder._validate_example(bundle, malformed),
                (False, "malformed_evidence_markers"),
            )
            self.assertEqual(
                builder._validate_example(bundle, insufficient_coverage),
                (False, "insufficient_citation_coverage"),
            )
            self.assertEqual(
                builder._validate_example(bundle, secondary_citation),
                (False, "non_corpus_secondary_citation"),
            )

            rejected_before_semantic_verification = builder._prevalidate_for_verifier(
                bundle,
                [missing, insufficient_coverage],
                [],
            )
            self.assertEqual(rejected_before_semantic_verification, [])
            self.assertEqual(
                missing["_prevalidation_rejection_reason"], "missing_evidence_markers"
            )
            self.assertEqual(
                insufficient_coverage["_prevalidation_rejection_reason"],
                "insufficient_citation_coverage",
            )

    def test_structured_evidence_renderer_rejects_invalid_evidence_ids(self):
        with self.assertRaisesRegex(ValueError, "invalid evidence segment"):
            _render_grounded_segments(
                [
                    {
                        "text": "Unsupported mapping.",
                        "separator_before": "space",
                        "supports": [
                            {"evidence_id": 2, "supporting_quote": "Grounded evidence."}
                        ],
                    }
                ],
                [{"text": "Grounded evidence."}],
            )

    def test_structured_evidence_renderer_validates_each_quote_against_its_chunk(self):
        chunks = [
            {"text": "Alpha increased by 4.2% in the treated group."},
            {"text": "Beta remained stable in the control group."},
        ]
        with self.assertRaisesRegex(ValueError, "invalid evidence segment"):
            _render_grounded_segments(
                [
                    {
                        "text": "Alpha increased by 4.2%.",
                        "separator_before": "space",
                        "supports": [
                            {
                                "evidence_id": 2,
                                "supporting_quote": "Alpha increased by 4.2%",
                            }
                        ],
                    }
                ],
                chunks,
            )

    def test_structured_evidence_renderer_preserves_scientific_prose_and_layout(self):
        chunks = [
            {
                "text": (
                    "Smith et al. reported 4.2% in the U.S. cohort. "
                    "Fig. 2 shows the fitted response."
                )
            }
        ]
        answer = _render_grounded_segments(
            [
                {
                    "text": "Smith et al. reported 4.2% in the U.S. cohort.",
                    "separator_before": "space",
                    "supports": [
                        {
                            "evidence_id": 1,
                            "supporting_quote": "Smith et al.  reported 4.2% in the U.S. cohort.",
                        }
                    ],
                },
                {
                    "text": "- Fig. 2 shows the fitted response.",
                    "separator_before": "newline",
                    "supports": [
                        {
                            "evidence_id": 1,
                            "supporting_quote": "Fig. 2 shows the fitted response.",
                        }
                    ],
                },
            ],
            chunks,
        )
        self.assertEqual(
            answer,
            "Smith et al. reported 4.2% in the U.S. cohort [E1].\n"
            "- Fig. 2 shows the fitted response [E1].",
        )

    def test_grounding_verifier_restores_immutable_question_and_mode(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "models": {"teacher": "teacher"},
                    "response": {"minimum_long_form_words": 350},
                },
            )
            builder.teacher = {
                "max_output_tokens": 5000,
                "maximum_transient_retries_per_request": 0,
            }
            payload = {
                "verdicts": [
                    {
                        "candidate_id": "candidate-1",
                        "fully_answers_question": True,
                        "speculative": False,
                        "relies_on_secondary_source": False,
                        "claim_verdicts": [
                            {
                                "claim_id": "claim_1",
                                "entailed": True,
                                "reason": "entailed",
                                "supports": [
                                    {
                                        "evidence_id": 1,
                                        "supporting_quote": "Corrected grounded evidence.",
                                    }
                                ],
                            },
                            {
                                "claim_id": "claim_2",
                                "entailed": True,
                                "reason": "entailed",
                                "supports": [
                                    {
                                        "evidence_id": 1,
                                        "supporting_quote": "A second supported claim.",
                                    }
                                ],
                            },
                        ],
                    }
                ]
            }
            response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload))
                    )
                ]
            )
            captured: dict[str, object] = {}

            def create(**kwargs):
                captured.update(kwargs)
                return response

            builder.client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=create)
                )
            )
            builder.timing = {"generation_seconds": 0.0, "verification_seconds": 0.0}
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Corrected grounded evidence. A second supported claim.",
                    }
                ],
            )
            original = {
                "candidate_id": "candidate-1",
                "question": "Original question?",
                "answer": "Corrected grounded answer [E1]. A second claim [E1]!",
                "response_mode": "standard",
            }

            reviewed = builder._verify_grounding(bundle, [original])

            self.assertEqual(reviewed[0]["question"], "Original question?")
            self.assertEqual(reviewed[0]["response_mode"], "standard")
            self.assertEqual(
                reviewed[0]["answer"],
                "Corrected grounded answer [E1]. A second claim [E1]!",
            )
            self.assertTrue(reviewed[0]["_grounding_supported"])
            verifier_prompt = captured["messages"][1]["content"]
            self.assertIn("Do not rewrite any question", verifier_prompt)
            self.assertIn("cross-domain adaptation", verifier_prompt)
            self.assertNotIn("repair these synthetic examples", verifier_prompt)
            self.assertEqual(captured["reasoning_effort"], "low")

    def test_grounding_verifier_reassembles_reordered_rows_by_candidate_id(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "models": {"teacher": "teacher"},
                    "response": {"minimum_long_form_words": 350},
                },
            )
            builder.teacher = {"max_output_tokens": 5000}
            builder.timing = {"generation_seconds": 0.0, "verification_seconds": 0.0}
            payload = {
                "verdicts": [
                    {
                        "candidate_id": "candidate-b",
                        "fully_answers_question": True,
                        "speculative": False,
                        "relies_on_secondary_source": False,
                        "claim_verdicts": [
                            {
                                "claim_id": "claim_1",
                                "entailed": True,
                                "reason": "entailed",
                                "supports": [
                                    {"evidence_id": 1, "supporting_quote": "Beta evidence."}
                                ],
                            }
                        ],
                    },
                    {
                        "candidate_id": "candidate-a",
                        "fully_answers_question": True,
                        "speculative": False,
                        "relies_on_secondary_source": False,
                        "claim_verdicts": [
                            {
                                "claim_id": "claim_1",
                                "entailed": True,
                                "reason": "entailed",
                                "supports": [
                                    {"evidence_id": 1, "supporting_quote": "Alpha evidence."}
                                ],
                            }
                        ],
                    },
                ]
            }
            builder.client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kwargs: SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content=json.dumps(payload))
                                )
                            ]
                        )
                    )
                )
            )
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Alpha evidence. Beta evidence.",
                    }
                ],
            )
            originals = [
                {
                    "candidate_id": "candidate-a",
                    "question": "Original A?",
                    "answer": "Alpha answer [E1].",
                    "response_mode": "standard",
                },
                {
                    "candidate_id": "candidate-b",
                    "question": "Original B?",
                    "answer": "Beta answer [E1].",
                    "response_mode": "standard",
                },
            ]

            reviewed = builder._verify_grounding(bundle, originals)

            self.assertEqual(
                [(row["candidate_id"], row["question"], row["answer"]) for row in reviewed],
                [
                    ("candidate-a", "Original A?", "Alpha answer [E1]."),
                    ("candidate-b", "Original B?", "Beta answer [E1]."),
                ],
            )

    def test_grounding_verifier_rejects_duplicate_candidate_ids(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "models": {"teacher": "teacher"},
                    "response": {"minimum_long_form_words": 350},
                },
            )
            builder.teacher = {
                "max_output_tokens": 5000,
                "maximum_transient_retries_per_request": 0,
            }
            builder.timing = {"generation_seconds": 0.0, "verification_seconds": 0.0}
            duplicate = {
                "candidate_id": "candidate-a",
                "fully_answers_question": False,
                "speculative": False,
                "relies_on_secondary_source": False,
                "claim_verdicts": [
                    {
                        "claim_id": "claim_1",
                        "entailed": False,
                        "reason": "unsupported",
                        "supports": [],
                    }
                ],
            }
            builder.client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kwargs: SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=json.dumps({"verdicts": [duplicate, duplicate]})
                                    )
                                )
                            ]
                        )
                    )
                )
            )
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Evidence.",
                    }
                ],
            )
            originals = [
                {
                    "candidate_id": "candidate-a",
                    "question": "A?",
                    "answer": "A [E1].",
                    "response_mode": "standard",
                },
                {
                    "candidate_id": "candidate-b",
                    "question": "B?",
                    "answer": "B [E1].",
                    "response_mode": "standard",
                },
            ]

            with self.assertRaisesRegex(
                TeacherInfrastructureError, "duplicated a candidate identity"
            ):
                builder._verify_grounding(bundle, originals)

    def test_transient_failures_retry_without_turning_empty_quality_into_a_retry(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "maximum_transient_retries_per_request": 2,
            "transient_retry_backoff_seconds": 0.0,
        }
        bundle = EvidenceBundle("bundle-1", ("document-1",), [{"text": "Evidence."}])
        attempts: list[int] = []

        def flaky_request(*args, **kwargs):
            attempts.append(kwargs["requested_examples"])
            if len(attempts) < 3:
                raise TimeoutError("teacher timed out")
            return [{"question": "Q?"}]

        builder._request_examples = flaky_request
        result = builder._request_examples_with_retries(bundle, requested_examples=8)
        self.assertEqual(result, [{"question": "Q?"}])
        self.assertEqual(attempts, [8, 8, 8])

        quality_calls = 0

        def empty_quality_request(*args, **kwargs):
            nonlocal quality_calls
            quality_calls += 1
            return []

        builder._request_examples = empty_quality_request
        self.assertEqual(builder._request_examples_with_retries(bundle), [])
        self.assertEqual(quality_calls, 1)

    def test_exhausted_schema_failures_abort_as_infrastructure(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "maximum_transient_retries_per_request": 2,
            "transient_retry_backoff_seconds": 0.0,
        }
        bundle = EvidenceBundle("bundle-1", ("document-1",), [{"text": "Evidence."}])
        attempts = 0

        def invalid_schema(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise json.JSONDecodeError("invalid", "{", 1)

        builder._request_examples = invalid_schema
        with self.assertRaises(TeacherInfrastructureError):
            builder._request_examples_with_retries(bundle)
        self.assertEqual(attempts, 3)

    def test_exhausted_verifier_failure_does_not_regenerate_candidates(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "maximum_transient_retries_per_request": 2,
            "transient_retry_backoff_seconds": 0.0,
        }
        bundle = EvidenceBundle("bundle-1", ("document-1",), [{"text": "Evidence."}])
        attempts = 0

        def failed_verifier(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise TeacherInfrastructureError("verifier retry budget exhausted")

        builder._request_examples = failed_verifier
        with self.assertRaisesRegex(TeacherInfrastructureError, "verifier retry budget"):
            builder._request_examples_with_retries(bundle)
        self.assertEqual(attempts, 1)

    def test_verifier_timeout_retries_the_same_compact_batch(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "verification_batch_size": 2,
            "maximum_transient_retries_per_request": 2,
            "transient_retry_backoff_seconds": 0.0,
        }
        bundle = EvidenceBundle("bundle-1", ("document-1",), [{"text": "Evidence."}])
        examples = [
            {
                "candidate_id": f"candidate-{index}",
                "question": f"Question {index}?",
                "answer": f"Answer {index} [E1].",
                "response_mode": "standard",
            }
            for index in (1, 2)
        ]
        attempts: list[list[str]] = []

        def verify(batch_bundle, batch, **kwargs):
            attempts.append([row["candidate_id"] for row in batch])
            if len(attempts) == 1:
                raise TimeoutError("verifier timed out")
            return batch

        builder._verify_grounding_batch = verify
        reviewed = builder._verify_grounding(bundle, examples)

        self.assertEqual(len(reviewed), 2)
        self.assertEqual(
            attempts,
            [["candidate-1", "candidate-2"], ["candidate-1", "candidate-2"]],
        )

    def test_empty_singleton_verifier_output_is_quarantined_and_run_continues(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "maximum_verification_attempts": 2,
            "maximum_consecutive_verification_quarantines": 3,
            "transient_retry_backoff_seconds": 0.0,
        }
        bundle = EvidenceBundle("bundle-1", ("document-1",), [{"text": "Evidence."}])
        examples = [
            {
                "candidate_id": "candidate-1",
                "question": "Question?",
                "answer": "Answer [E1].",
                "response_mode": "standard",
            }
        ]
        attempts = 0

        def empty_verifier(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise ValueError("Teacher returned empty grounding-verification output")

        builder._verify_grounding_batch = empty_verifier
        reviewed = builder._verify_grounding(bundle, examples)

        self.assertEqual(attempts, 2)
        self.assertFalse(reviewed[0]["_grounding_supported"])
        self.assertEqual(
            reviewed[0]["_grounding_rejection_reason"],
            "semantic_verification_unavailable",
        )

    def test_repeated_verifier_quarantines_open_circuit_breaker(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "maximum_verification_attempts": 1,
            "maximum_consecutive_verification_quarantines": 1,
            "transient_retry_backoff_seconds": 0.0,
        }
        builder._consecutive_verification_quarantines = 1
        bundle = EvidenceBundle("bundle-1", ("document-1",), [{"text": "Evidence."}])
        examples = [
            {
                "candidate_id": "candidate-1",
                "question": "Question?",
                "answer": "Answer [E1].",
                "response_mode": "standard",
            }
        ]
        builder._verify_grounding_batch = lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Teacher returned empty grounding-verification output")
        )

        with self.assertRaisesRegex(TeacherInfrastructureError, "circuit breaker"):
            builder._verify_grounding(bundle, examples)

    def test_request_prompt_can_disable_long_form_generation(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "models": {"teacher": "teacher"},
                    "response": {
                        "minimum_long_form_words": 350,
                        "target_long_form_words": 850,
                        "maximum_long_form_words": 1100,
                    },
                },
            )
            builder.teacher = {"temperature": 0.3, "max_output_tokens": 5000}
            captured: dict[str, object] = {}
            payload = {
                "examples": [
                    {
                        "question": f"Question {index}?",
                        "answer": f"Answer {index} [E1].",
                        "response_mode": "standard",
                    }
                    for index in range(1, 8)
                ]
            }

            def create(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=json.dumps(payload))
                        )
                    ]
                )

            builder.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            builder.timing = {
                "generation_seconds": 0.0,
                "verification_seconds": 0.0,
            }
            builder._prevalidate_for_verifier = (
                lambda bundle, examples, accepted_questions: examples
            )
            builder._verify_grounding = lambda bundle, examples, **kwargs: examples
            builder._validate_example = lambda bundle, example: (True, "ok")
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Grounded evidence.",
                    }
                ],
            )

            builder._request_examples(
                bundle, allow_long_form=False, requested_examples=7
            )

            prompt = captured["messages"][1]["content"]
            self.assertIn("do not produce a long_form example", prompt)
            self.assertIn("Create exactly 7 high-quality training examples", prompt)
            schema = captured["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["properties"]["examples"]["minItems"], 7)
            self.assertEqual(schema["properties"]["examples"]["maxItems"], 7)

    def test_build_production_lock_precedes_release_and_selection(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "teacher": {"expected_training_documents": 2500},
                },
            )
            with (
                patch("giis_prod.teacher.begin_dataset_release") as begin_release,
                patch("giis_prod.teacher.select_training_documents") as select_documents,
            ):
                with self.assertRaises(ProductionReadinessError):
                    builder.build()
            begin_release.assert_not_called()
            select_documents.assert_not_called()

    def test_smoke_build_marks_release_building_before_selection(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "teacher": {"expected_training_documents": 2500},
                },
            )
            builder._checkpoint_policy_fingerprint = lambda: "policy-v15"
            events: list[str] = []

            def begin_release(*args, **kwargs):
                events.append("begin")

            def stop_at_selection(*args, **kwargs):
                events.append("selection")
                raise RuntimeError("stop-after-selection")

            with (
                patch("giis_prod.teacher.begin_dataset_release", side_effect=begin_release),
                patch(
                    "giis_prod.teacher.select_training_documents",
                    side_effect=stop_at_selection,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-after-selection"):
                    builder.build(expected_documents=5)
            self.assertEqual(events, ["begin", "selection"])

    def test_verifier_reuses_candidates_in_two_sub_batches_after_invalid_json(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "models": {"teacher": "teacher"},
                    "response": {
                        "minimum_long_form_words": 350,
                    },
                },
            )
            builder.teacher = {"max_output_tokens": 5000}
            builder.timing = {
                "generation_seconds": 0.0,
                "verification_seconds": 0.0,
            }
            examples = [
                {
                    "candidate_id": f"candidate-{index}",
                    "question": f"Original question {index}?",
                    "answer": f"Draft answer {index} [E1].",
                    "response_mode": "standard",
                }
                for index in (1, 2)
            ]
            responses = [
                "{\"verdicts\":[",
                json.dumps(
                    {
                        "verdicts": [
                            {
                                "candidate_id": "candidate-1",
                                "fully_answers_question": True,
                                "speculative": False,
                                "relies_on_secondary_source": False,
                                "claim_verdicts": [
                                    {
                                        "claim_id": "claim_1",
                                        "entailed": True,
                                        "reason": "entailed",
                                        "supports": [
                                            {
                                                "evidence_id": 1,
                                                "supporting_quote": "supported result",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "verdicts": [
                            {
                                "candidate_id": "candidate-2",
                                "fully_answers_question": True,
                                "speculative": False,
                                "relies_on_secondary_source": False,
                                "claim_verdicts": [
                                    {
                                        "claim_id": "claim_1",
                                        "entailed": True,
                                        "reason": "entailed",
                                        "supports": [
                                            {
                                                "evidence_id": 1,
                                                "supporting_quote": "supported result",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ),
            ]
            requested_sizes = []

            def create(**kwargs):
                schema = kwargs["response_format"]["json_schema"]["schema"]
                requested_sizes.append(
                    schema["properties"]["verdicts"]["minItems"]
                )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=responses.pop(0))
                        )
                    ]
                )

            builder.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "The study reported a supported result.",
                    }
                ],
            )

            result = builder._verify_grounding(bundle, examples)

            self.assertEqual(requested_sizes, [2, 1, 1])
            self.assertEqual(
                [row["question"] for row in result],
                ["Original question 1?", "Original question 2?"],
            )
            self.assertTrue(all(row["_grounding_supported"] for row in result))

    def test_granite_batch_encoding_counts_input_ids_not_mapping_keys(self):
        encoded = {
            "input_ids": [[101, 102, 103, 104]],
            "attention_mask": [[1, 1, 1, 1]],
        }
        self.assertEqual(TeacherDatasetBuilder._token_count(encoded), 4)
        self.assertEqual(TeacherDatasetBuilder._token_count([101, 102, 103]), 3)

    def test_only_explicit_small_smoke_run_disables_multisource_extras(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "multi_source_fraction": 0.35,
            "timed_smoke_max_documents": 5,
            "timed_smoke_disable_multi_source": True,
        }
        self.assertEqual(builder._multi_source_fraction_for_run(5), (0.0, True))
        self.assertEqual(builder._multi_source_fraction_for_run(6), (0.35, False))
        self.assertEqual(builder._multi_source_fraction_for_run(None), (0.35, False))

    def test_prevalidation_uses_embedding_similarity_for_conceptual_duplicates(self):
        import numpy as np

        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "response": {"minimum_long_form_words": 350},
                    "training": {"max_length": 8192},
                },
            )
            builder.teacher = {
                "minimum_training_citation_coverage": 0.40,
                "maximum_question_similarity": 1.01,
                "maximum_question_embedding_similarity": 0.88,
            }

            class FakeTokenizer:
                @staticmethod
                def apply_chat_template(messages, **kwargs):
                    return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

            class FakeEncoder:
                @staticmethod
                def encode(questions, **kwargs):
                    vectors = {
                        "How did forest cover affect suitability?": [1.0, 0.0],
                        "What was the suitability response in wooded areas?": [0.95, 0.05],
                        "Which temperature data were used?": [0.0, 1.0],
                    }
                    return np.asarray([vectors[question] for question in questions])

            builder.tokenizer = FakeTokenizer()
            builder.question_encoder = FakeEncoder()
            bundle = EvidenceBundle(
                bundle_id="bundle-1",
                document_ids=("document-1",),
                chunks=[
                    {
                        "document_id": "document-1",
                        "page_start": 1,
                        "page_end": 1,
                        "section": "Results",
                        "title": "Paper",
                        "text": "Grounded evidence.",
                    }
                ],
            )
            examples = [
                {
                    "question": "What was the suitability response in wooded areas?",
                    "answer": (
                        "The paper reported a measurable suitability response in wooded areas "
                        "[E1]."
                    ),
                    "response_mode": "standard",
                },
                {
                    "question": "Which temperature data were used?",
                    "answer": (
                        "The paper explicitly identified the temperature data used for analysis "
                        "[E1]."
                    ),
                    "response_mode": "standard",
                },
            ]

            viable = builder._prevalidate_for_verifier(
                bundle,
                examples,
                ["How did forest cover affect suitability?"],
            )

            self.assertEqual([row["question"] for row in viable], [examples[1]["question"]])
            self.assertEqual(
                examples[0]["_prevalidation_rejection_reason"],
                "semantic_duplicate_question",
            )

    def test_claim_units_preserve_text_and_extract_declared_evidence(self):
        claims = _claim_units(
            "First result was 4.2% [E1].\nSecond result remained stable [E2][E3]."
        )
        self.assertEqual([row["claim_id"] for row in claims], ["claim_1", "claim_2"])
        self.assertEqual(claims[0]["cited_evidence_ids"], [1])
        self.assertEqual(claims[1]["cited_evidence_ids"], [2, 3])
        self.assertIn("4.2%", claims[0]["text"])

    def test_audit_journal_is_immediately_readable(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit" / "run.jsonl"
            journal = DatasetAuditJournal(path, "run-1")
            journal.append("candidate_final_decision", candidate_id="candidate-1", reason="ok")

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["run_id"], "run-1")
            self.assertEqual(rows[0]["event"], "candidate_final_decision")
            self.assertEqual(rows[0]["candidate_id"], "candidate-1")

    def test_build_journals_keyboard_interrupt_before_reraising(self):
        with TemporaryDirectory() as temporary:
            builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
            builder.settings = Settings(
                root=Path(temporary),
                data={
                    "paths": {"sft_dir": "artifacts/sft"},
                    "teacher": {"expected_training_documents": 2500},
                },
            )
            builder._checkpoint_policy_fingerprint = lambda: "f" * 64

            def interrupt(*args, **kwargs):
                raise KeyboardInterrupt

            builder._build_release = interrupt
            with self.assertRaises(KeyboardInterrupt):
                builder.build(expected_documents=5)

            audit_files = list((Path(temporary) / "artifacts" / "sft" / "audit").glob("*.jsonl"))
            self.assertEqual(len(audit_files), 1)
            events = [
                json.loads(line)["event"]
                for line in audit_files[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events, ["run_started", "run_failed"])

    def test_semantic_verifier_rejects_quote_from_an_uncited_chunk(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.settings = Settings(
            root=Path("."),
            data={"models": {"teacher": "teacher"}},
        )
        builder.teacher = {"verification_batch_size": 2, "verification_max_output_tokens": 3000}
        builder.timing = {"generation_seconds": 0.0, "verification_seconds": 0.0}
        payload = {
            "verdicts": [
                {
                    "candidate_id": "candidate-1",
                    "fully_answers_question": True,
                    "speculative": False,
                    "relies_on_secondary_source": False,
                    "claim_verdicts": [
                        {
                            "claim_id": "claim_1",
                            "entailed": True,
                            "reason": "entailed",
                            "supports": [
                                {
                                    "evidence_id": 2,
                                    "supporting_quote": "The actual supporting result.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        builder.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
                        ]
                    )
                )
            )
        )
        bundle = EvidenceBundle(
            bundle_id="bundle-1",
            document_ids=("document-1",),
            chunks=[
                {
                    "document_id": "document-1",
                    "page_start": 1,
                    "page_end": 1,
                    "section": "Results",
                    "title": "Paper",
                    "text": "An unrelated result.",
                },
                {
                    "document_id": "document-1",
                    "page_start": 2,
                    "page_end": 2,
                    "section": "Results",
                    "title": "Paper",
                    "text": "The actual supporting result.",
                },
            ],
        )
        original = {
            "candidate_id": "candidate-1",
            "question": "What was found?",
            "answer": "The study reported the result [E1].",
            "response_mode": "standard",
        }

        reviewed = builder._verify_grounding(bundle, [original])

        self.assertFalse(reviewed[0]["_grounding_supported"])
        self.assertEqual(
            reviewed[0]["_grounding_rejection_reason"], "invalid_verifier_provenance"
        )


if __name__ == "__main__":
    unittest.main()
