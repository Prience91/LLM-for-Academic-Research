from collections import OrderedDict
from types import SimpleNamespace

from giis_prod.retrieval import HybridRetriever
from giis_prod.retrieval import QueryRewriter


def candidate(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "document_id": "doc-1",
            "title": "Source",
            "authors": ["Author"],
            "year": 2026,
            "page_start": 1,
            "page_end": 1,
            "token_count": 20,
        },
        "dense_similarity": 0.9,
        "lexical_score": 0.8,
        "fused_score": 0.03,
        "reranker_score": 5.0,
    }


class Planner:
    def rewrite(self, query, history):
        return query

    def plan_obligations(self, query):
        return ["requirement one", "requirement two", "requirement three", "requirement four"]

    def expand_obligations(self, query, explicit, evidence):
        return explicit, True

    def validate_coverage(self, query, obligations, evidence):
        return [
            {
                "obligation_index": index,
                "obligation": obligation,
                "covered": True,
                "evidence_ids": [evidence[index - 1].evidence_id],
                "validation": "semantic",
            }
            for index, obligation in enumerate(obligations, 1)
        ]


def test_obligation_slots_override_legacy_three_chunk_document_ceiling():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.configuration = {
        "enable_retrieval_cache": False,
        "enable_obligation_aware_retrieval": True,
        "max_chunks_per_document": 3,
        "obligation_max_chunks_per_document": 8,
        "final_chunks": 12,
        "maximum_context_tokens": 6144,
        "enable_context_deduplication": True,
        "context_deduplication_similarity": 0.85,
        "min_reranker_score": 0.0,
        "enable_source_diversity": True,
        "source_diversity_penalty": 0.08,
    }
    retriever.rewriter = Planner()
    retriever._cache = OrderedDict()
    ranked = {
        "question": [candidate("base", "overall framing for the research question")],
        "requirement one": [candidate("one", "methodological evidence for the first requirement")],
        "requirement two": [candidate("two", "substantive findings for the second requirement")],
        "requirement three": [candidate("three", "limitations relevant to the third requirement")],
        "requirement four": [candidate("four", "implications relevant to the fourth requirement")],
    }
    retriever._rank_candidates = lambda query, expanded=False: list(ranked.get(query, []))

    result = retriever.retrieve("question")

    assert not result.abstained
    assert result.coverage_complete is True
    assert len(result.obligations) == 4
    assert len(result.evidence) >= 4
    assert {item.document_id for item in result.evidence} == {"doc-1"}
    assert {"one", "two", "three", "four"}.issubset(
        {item.chunk_id for item in result.evidence}
    )


def test_explicit_clause_guard_preserves_three_part_question():
    obligations = QueryRewriter._explicit_obligations(
        "How did the study investigate perceptions, what did it find about worldviews, "
        "and what communication implications followed?"
    )
    assert len(obligations) == 3
    assert "communication implications" in obligations[2]


def test_explicit_clause_guard_attaches_leading_context_without_inventing_obligation():
    obligations = QueryRewriter._explicit_obligations(
        "Under the paper's framework, what conditions must a practice satisfy, "
        "and what constraints limit adoption?"
    )
    assert len(obligations) == 2
    assert all(item.startswith("Under the paper's framework,") for item in obligations)
