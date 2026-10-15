# Obligation–evidence semantic validation failure handling

## Authority and scope

This amendment aligns retrieval and generation with the semantic failure contract in [UI_POLICY.md](UI_POLICY.md). It corrects failure handling in the original academic-research workflow. It does not change SFT dataset generation, retrieval thresholds, answer capacities or model identities, and it does not make the entire development UI production-conformant.

The configured local model assesses whether the supplied passages sufficiently support each obligation, then serves answer generation through separate calls. This is self-validation, not an independent review. The Codex-to-MCP spin-off delegates that role to its calling Codex client; that separate deployment is not the execution target of this amendment.

## Required behavior

1. Candidate selection, similarity and reserved slots are provisional retrieval evidence, never semantic approval.
2. The semantic response must contain exactly one decision per obligation. Indices must be unique valid integers, coverage decisions must be JSON booleans, and passage identifiers must belong to the supplied packet. A positive decision requires at least one supplied evidence ID.
3. Timeouts, failed requests, malformed JSON, incomplete/duplicate rows, invalid decisions and foreign IDs produce `status=validation_unavailable`, `validation=unavailable`, `covered=false`, and no approved evidence IDs.
4. Retrieval fallback must preserve that unavailable state. It must not convert a selected candidate into `covered=true`.
5. `coverage_complete=true` requires valid semantic decisions for every obligation, explicit support IDs for every positive decision, and successful atomic planning. Retrieval-only mappings cannot meet this condition.
6. A changed final selected bundle must be semantically reviewed again. If that review fails, an earlier positive completeness claim must not survive.
7. Bounded retrieval rescue addresses evidence gaps found by a working semantic reviewer. A reviewer outage is not proof that more retrieval will solve the problem.
8. If any required semantic decision is unavailable, answer generation must abstain before calling the answer model, using `semantic_validation_unavailable`. A valid semantic decision that evidence is insufficient remains an ordinary evidence gap; it can be handled through the normal explicit-limitation policy.

The generator also labels unverified mappings as unavailable rather than describing them as covered in the model prompt.

## Validation record: 15 October 2026

The source correction was applied in Mini-run on Spark. Existing source and policy versions were backed up inside that project. The GitHub publication uses files whose SHA256 hashes match the corrected, tested Mini-run versions.

The focused suite passed **26 tests**, using synthetic evidence and no model calls or corpus-document reads:

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/test_obligation_retrieval.py \
  tests/test_semantic_validation_failure.py \
  tests/test_generation_citation_policy.py
```

Tests cover timeout and malformed-result handling, fallback completeness, final-review failure, the pre-generation abstention gate, valid semantic approvals and valid uncovered decisions. The GitHub workflow runs this same focused suite without loading model/index artifacts.

No inference runtime was started or restarted by this amendment. A future process loads the corrected source at startup; an existing process that already imported the old module must reload it. Historical runs have not been retrospectively reclassified, and these synthetic tests do not measure a model's semantic judgment accuracy.
