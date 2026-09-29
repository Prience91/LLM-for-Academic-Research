# GIIS Mini run guidebook

This directory is an independent V17-Turbo restart over the 1,000-paper climate-change canary. The reusable policy, code, configuration, tests, and operational documentation were copied from Smoke run II. No prior model, dataset, vector database, parsed corpus, benchmark output, log, cache, or generated artifact was copied.

## Canonical authority

1. `V17_TURBO_POLICY.md` defines the end-to-end generation, review, held-out, retrieval-binding, and release rules.
2. `PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md` defines raw intake, incremental parsing, metadata reconciliation, canonical corpus freezing, embedding, and hybrid-index activation.
3. `V17_TURBO_PROTOCOL.md` defines stage order and execution gates.
4. `UI_POLICY.md` defines decoder interaction, token capacity, obligation completion, visible coverage, citation, security, and release behavior.
5. `config/settings.toml` is the active Mini-run configuration; `config/settings_v17_turbo.toml` is its named frozen-plan copy.
6. `src/giis_prod/` is the implementation. V17-Turbo modules do not depend on V15, V16, or V17 policy modules.
7. `giis_pipeline.py` is the operator entry point.

## Fresh-state boundaries

- RAG intake: `RAG Corpus` — currently 1,250 files with 1,195 unique hashes; exactly 1,000 verified unique papers will be frozen as the canary, including held-out papers.
- Training source: `Training Corpus` — created only after the held-out split and containing exactly the 900 non-held-out papers.
- Held out: 100 papers, frozen before generation.
- Generation pool: 900 papers; 810 train-generation and 90 validation-generation.
- Fresh outputs: `artifacts/v17_turbo/`, `benchmark/`, and `models/` are created by the Mini run only.
- Shared Hugging Face snapshots may be read from the central cache; model outputs are never inherited from Smoke run II.

> **Canonical routing amendment (2026-09-29):** [V17_TURBO_UNIVERSAL_OSS_FIRST_AMENDMENT.md](V17_TURBO_UNIVERSAL_OSS_FIRST_AMENDMENT.md) supersedes the earlier lane-split semantic-routing rule for newly frozen plans. The active hash-bound held-out run remains governed by its frozen fingerprint until completion.

## Stage order

1. Verify the overprovisioned `RAG Corpus` against the original and 250-paper expansion manifests, reconcile duplicates and quarantine, then freeze exactly 1,000 verified SHA-256-distinct papers as the canary. Treat an exact twelve-hexadecimal-character plus double-underscore filename prefix as collision metadata during title lookup, preserve the full source filename, and audit every normalization.
2. Parse with Docling, apply OCR fallback where required, and run cached DOI/title reconciliation through Crossref and OpenAlex. Preserve provider provenance and quarantine unresolved or ambiguous identities without weakening the match threshold.
3. Freeze a verified release of at least 1,000 SHA-256-distinct documents and require its deterministic audit to pass. Keep unresolved intake outside the release for later adjudication, then build the content-addressed Granite/Chroma and SQLite FTS5 indexes.
4. Freeze the deterministic 100-paper held-out split, then materialize and hash-verify exactly 900 non-held-out papers into `Training Corpus`. Held-out papers remain in RAG.
5. Author held-out obligation/reference packets with Qwen 122B; keep references hidden from the student. Route every held-out lane through OSS first, then send only OSS non-admissions to permission-gated gold review.
6. Generate V17-Turbo data with Qwen 122B, using obligation-first evidence assembly, deterministic Python verification, bounded retries, capacity lanes, and resumable per-paper checkpoints.
7. Under the 2026-09-29 universal OSS-first amendment, route every candidate in every lane through GPT-OSS-120B first. Admit OSS-complete-and-supported candidates directly; freeze every OSS non-admission in a blinded, permission-gated gold-review queue. Gold-complete candidates may be salvaged; hold minor omissions for authorized repair and full requalification; discard material incompleteness and unsupported content.
8. Materialize only admitted records, bind the actual Qwen 27B student tokenizer, enforce the 8,192-token sample ceiling, and run the worst-case one-step QLoRA memory canary.
9. Train Qwen 27B in direct-answer mode, select checkpoints using the validation split, merge the adapter, and preserve the 100-paper held-out set for one-shot final evaluation.
10. Serve the merged model behind the hybrid RAG path and Gradio UI. Retrieval must assemble evidence by obligation and validate coverage before generation.

## Operator commands

```bash
cd "/home/codeaccounting/Documents/Mini run"
source /home/codeaccounting/Documents/LLM_env/bin/activate
python3 giis_pipeline.py status
python3 giis_pipeline.py corpus --expected-documents 1000
python3 giis_pipeline.py audit-corpus
python3 giis_pipeline.py plan-heldout --expected-corpus-documents 1000 --heldout-documents 100
python3 giis_pipeline.py materialize-training
python3 giis_pipeline.py author-heldout --output artifacts/v17_turbo/heldout_authoring
```

Dataset generation, reviewer handoff, training, merge, held-out evaluation, and deployment remain gated by the preceding frozen manifests. Follow `V17_TURBO_PROTOCOL.md` rather than skipping stages.

## Policy map

| Concern | Governing implementation |
|---|---|
| Parsing and metadata | `PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md`, `ingestion.py`, `metadata.py`, `corpus_audit.py` |
| Embeddings and vector/lexical indexes | `PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md`, `indexing.py`, `v17_turbo_contract.py` |
| Hybrid and obligation-aware retrieval | `retrieval.py`, `citations.py`, `evaluation.py` |
| V17-Turbo generation | `heldout_authoring.py`, `v17_turbo_layer.py`, `v17_turbo_builder.py`, `v17_turbo_canary.py` |
| Semantic routing | `semantic_audit.py`, `v17_turbo_contract.py`, `v17_turbo_qualification.py` |
| SFT and packaging | `training_materialization.py`, `training.py`, `packaging.py` |
| Serving and UI | `UI_POLICY.md`, `server.py`, `ui.py`, `generation.py`, `citations.py`, `giis_pipeline.py` |
| Release and regression tests | `dataset_release.py`, `production_readiness.py`, `tests/` |
| Historical held-out comparison | `HELDOUT_MODEL_COMPARISON_NOTE.md` |

## Excluded Smoke-run state

The bootstrap intentionally excludes `artifacts/`, `models/`, prior paper directories, `benchmark/`, `wandb/`, caches, backups, legacy scripts, run-specific comparison scripts, old databases, old split/import manifests, and superseded policy documents.
