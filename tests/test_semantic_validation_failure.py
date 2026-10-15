import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import giis_prod.generation as generation
from giis_prod.retrieval import HybridRetriever, QueryRewriter


def candidate(key):
    return {"chunk_id": key, "text": "Synthetic evidence " + key,
            "metadata": {"document_id": "doc-1", "title": "Synthetic source",
                         "authors": ["Author"], "year": 2026,
                         "page_start": 1, "page_end": 1, "token_count": 20},
            "dense_similarity": 0.9, "fused_score": 0.03, "reranker_score": 5.0}


def semantic_rows(obligations, evidence):
    return [{"obligation_index": i, "obligation": o, "covered": True,
             "evidence_ids": [evidence[0].evidence_id], "validation": "semantic"}
            for i, o in enumerate(obligations, 1)]


def make_retriever(review, extra=False):
    r = HybridRetriever.__new__(HybridRetriever)
    r.configuration = {"enable_retrieval_cache": False,
                       "enable_obligation_aware_retrieval": True,
                       "max_chunks_per_document": 3, "obligation_max_chunks_per_document": 8,
                       "final_chunks": 12, "maximum_context_tokens": 6144,
                       "enable_context_deduplication": False,
                       "min_reranker_score": 0.0, "enable_source_diversity": False}
    r._cache = OrderedDict()
    r.rewriter = SimpleNamespace(rewrite=lambda q,h:q, plan_obligations=lambda q:["requirement"],
        expand_obligations=lambda q,o,e:(o,True), validate_coverage=review)
    r._rank_candidates = lambda q,expanded=False: [candidate("extra" if extra and q=="question" else "one")]
    return r


def test_failed_validator_cannot_become_complete_through_fallback():
    result = make_retriever(lambda q,o,e:[]).retrieve("question")
    assert result.coverage_complete is False
    assert result.obligation_coverage[0]["covered"] is False
    assert result.obligation_coverage[0]["status"] == "validation_unavailable"
    assert result.obligation_coverage[0]["evidence_ids"] == []


def test_failed_final_review_does_not_keep_initial_approval():
    calls = []
    def review(q,o,e):
        calls.append(len(e))
        return semantic_rows(o,e) if len(calls)==1 else []
    result = make_retriever(review, extra=True).retrieve("question")
    assert calls == [1,2]
    assert result.coverage_complete is False
    assert result.obligation_coverage[0]["validation"] == "unavailable"


def test_successful_semantic_review_remains_complete():
    result = make_retriever(lambda q,o,e:semantic_rows(o,e)).retrieve("question")
    assert result.coverage_complete is True
    assert result.obligation_coverage[0]["validation"] == "semantic"


def test_validator_timeout_is_reported_explicitly():
    r = QueryRewriter.__new__(QueryRewriter)
    def timeout(*args, **kwargs): raise TimeoutError("synthetic timeout")
    r._complete = timeout
    rows = r.validate_coverage("q",["requirement"],[SimpleNamespace(evidence_id="E1",text="synthetic")])
    assert rows[0]["status"] == "validation_unavailable"
    assert rows[0]["covered"] is False


@pytest.mark.parametrize("payload",[
    "invalid JSON",
    {"coverage":[]},
    {"coverage":[{"obligation_index":1,"covered":"false","evidence_ids":["E1"]}]},
    {"coverage":[{"obligation_index":1,"covered":True,"evidence_ids":[]}]},
    {"coverage":[{"obligation_index":1,"covered":True,"evidence_ids":["foreign"]}]},
    {"coverage":[{"obligation_index":True,"covered":True,"evidence_ids":["E1"]}]},
])
def test_malformed_semantic_output_never_approves(payload):
    r = QueryRewriter.__new__(QueryRewriter)
    r._complete = lambda *a,**k: payload if isinstance(payload,str) else json.dumps(payload)
    rows = r.validate_coverage("q",["requirement"],[SimpleNamespace(evidence_id="E1",text="synthetic")])
    assert rows[0]["status"] == "validation_unavailable"
    assert rows[0]["covered"] is False


def test_valid_uncovered_decision_is_not_a_reviewer_failure():
    r = QueryRewriter.__new__(QueryRewriter)
    r._complete = lambda *a,**k: json.dumps({"coverage":[
        {"obligation_index":1,"covered":False,"evidence_ids":[]}]})
    rows = r.validate_coverage("q",["requirement"],[SimpleNamespace(evidence_id="E1",text="synthetic")])
    assert rows[0]["validation"] == "semantic"
    assert rows[0]["status"] == "uncovered"


def make_generator(monkeypatch, coverage):
    result = SimpleNamespace(abstained=False, reason=None, evidence=[object()],
        rewritten_query="question", obligations=["requirement"], obligation_coverage=coverage,
        coverage_complete=False, retrieval_passes=2, atomic_planning_succeeded=True)
    g = generation.GroundedGenerator.__new__(generation.GroundedGenerator)
    g.retriever = SimpleNamespace(retrieve=lambda *a:result)
    g.retrieval = {"minimum_distinct_sources_for_long_form":1}
    g.response = {}
    monkeypatch.setattr(generation,"evidence_is_sufficient",lambda *a:True)
    monkeypatch.setattr(generation,"render_reference_panel",lambda *a:"references")
    return g


def test_generation_stops_before_model_call_when_review_unavailable(monkeypatch):
    g = make_generator(monkeypatch,[{"obligation_index":1,"covered":False,
        "evidence_ids":[],"validation":"unavailable"}])
    def forbidden(*a,**k): raise AssertionError("generation must not be called")
    g._generate = forbidden
    result = g.answer("question",requested_mode=generation.ResponseMode.STANDARD)
    assert result.abstained is True
    assert result.abstention_reason == "semantic_validation_unavailable"
    assert result.coverage_complete is False


def test_valid_review_with_evidence_gap_can_generate_limitation(monkeypatch):
    g = make_generator(monkeypatch,[{"obligation_index":1,"covered":False,
        "evidence_ids":[],"validation":"semantic"}])
    g._generate = lambda *a,**k:generation._CompletionResult(text="Evidence is insufficient.",finish_reason="stop")
    g._response_rejection_reasons = lambda *a:[]
    monkeypatch.setattr(generation,"render_inline_citations",lambda text,evidence:text)
    result = g.answer("question",requested_mode=generation.ResponseMode.STANDARD)
    assert result.abstained is False


def test_prompt_does_not_label_unverified_candidates_covered():
    prompt = generation._coverage_prompt(["requirement"],[{"obligation_index":1,
        "covered":True,"evidence_ids":["E1"],"validation":"targeted_retrieval"}])
    assert "validation_unavailable" in prompt
    assert "[covered" not in prompt
