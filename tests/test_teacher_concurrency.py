from __future__ import annotations

import threading
import time
import unittest

from giis_prod.teacher import TeacherDatasetBuilder


class TeacherGenerationConcurrencyTests(unittest.TestCase):
    def test_two_requests_overlap_but_release_order_stays_stable(self):
        builder = TeacherDatasetBuilder.__new__(TeacherDatasetBuilder)
        builder.teacher = {
            "seed": 3407,
            "multi_source_fraction": 0.0,
            "evidence_chunks_per_document": 1,
            "minimum_substantive_chunks_per_document": 1,
            "minimum_core_examples_per_document": 1,
            "maximum_core_examples_per_document": 1,
            "minimum_long_form_examples_per_document": 0,
            "maximum_long_form_examples_per_document": 0,
            "maximum_core_requests_per_document": 1,
            "maximum_zero_yield_requests_before_minimum": 1,
            "initial_candidate_batch_size": 1,
            "refill_candidate_batch_size": 1,
            "maximum_question_similarity": 0.72,
            "generation_concurrency": 2,
        }
        builder.timing = {"generation_seconds": 0.0, "verification_seconds": 0.0}
        builder._audit = lambda *args, **kwargs: None
        builder._validate_example = lambda bundle, example: (True, "ok")
        builder._training_messages = lambda bundle, question, answer: [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        builder._chat_token_count = lambda messages: 1

        barrier = threading.Barrier(2, timeout=2)

        def request(bundle, **kwargs):
            barrier.wait()
            if bundle.document_ids[0] == "doc_a":
                time.sleep(0.05)
            return [
                {
                    "candidate_id": f"candidate_{bundle.document_ids[0]}",
                    "question": f"Question for {bundle.document_ids[0]}?",
                    "answer": "Grounded answer [E1].",
                    "response_mode": "standard",
                    "answer_style": "prose",
                }
            ]

        builder._request_examples_with_retries = request
        chunks = [
            {
                "document_id": document_id,
                "chunk_id": f"chunk_{document_id}",
                "page_start": 1,
                "page_end": 1,
                "ordinal": 0,
                "section": "Results",
                "title": document_id,
                "text": "Substantive evidence text " * 20,
            }
            for document_id in ("doc_a", "doc_b")
        ]

        accepted, rejected = builder._records_for_partition(
            "train",
            {"doc_a", "doc_b"},
            chunks,
            multi_source_fraction=0.0,
        )

        self.assertEqual(rejected, [])
        self.assertEqual(
            [row["document_ids"] for row in accepted], [["doc_a"], ["doc_b"]]
        )


if __name__ == "__main__":
    unittest.main()
