# LLM for Academic Research 
LLM for Academic Research is an evidence-grounded research pipeline for scientific literature. The repository combines verified document ingestion, hybrid obligation-aware retrieval, V17-Turbo synthetic-data generation, QLoRA supervised fine-tuning, held-out evaluation, and a coverage-aware user interface.

The repository contains framework code and policy documentation only. Papers, parsed corpora, vector databases, generated datasets, model weights, adapters, checkpoints, credentials, and run artifacts are intentionally excluded.

## Architecture

```mermaid
flowchart LR
    A[Scientific PDFs] --> B[Docling parsing]
    B --> C[Crossref and OpenAlex reconciliation]
    C --> D[Verified document registry]
    D --> E[Granite embeddings and Chroma]
    D --> F[SQLite FTS5 lexical index]
    E --> G[Obligation-aware hybrid retrieval]
    F --> G
    G --> H[Grounded answer generation]
    D --> I[V17-Turbo candidate generation]
    I --> J[Deterministic qualification]
    J --> K[Train and validation: all-lane OSS review]
    J --> P[Held-out: all-lane gold review]
    K --> O[Authorized gold review of semantic rejects]
    K --> L
    P --> L
    O --> L[Immutable SFT release]
    L --> M[Qwen QLoRA training]
    M --> H
    H --> N[Coverage-aware UI]
```

## Framework components

1. **Parsing, metadata, embedding, and indexing**
   - Docling native extraction with bounded OCR fallback.
   - Source-bound overrides, then Crossref DOI/title lookup and OpenAlex DOI/title fallback.
   - Budget deferrals stay backlogged while parsing continues; oldest deferred work gets renewed allowance first.
   - English unresolved metadata goes to Qwen-122B vision adjudication; referrals and non-English sources await authorized gold review.
   - SHA-256-bound document and chunk registries.
   - Granite embeddings in Chroma plus a matching SQLite FTS5 lexical index.

2. **Obligation-evidence retrieval**
   - Dense and lexical candidate fusion, reranking, source diversity, and context deduplication.
   - Obligation-first retrieval and explicit coverage validation before decoding.
   - Page, section, document, and citation provenance retained through the answer path.

3. **V17-Turbo SFT generation**
   - Single-source, multi-source, table, and comparison lanes.
   - Evidence-backed obligation ledgers and answer-unit mappings.
   - Capacity lanes that preserve answer completeness without legacy word ceilings.
   - Every training/validation lane receives GPT-OSS-120B review; confident semantic rejects enter blinded, permission-gated gold review.
   - Every generated held-out reference receives direct gold review across every lane.
   - Qwen authoring and OSS failed axes get one fail-closed retry; exhausted failures are terminal quarantine, not gold salvaging.
   - One launch runs main generation, OSS review, and held-out authoring before a joint gold-authorization checkpoint.

4. **Fine-tuning and evaluation**
   - QLoRA/NF4 fine-tuning for the Qwen student with memory canaries and immutable releases.
   - Validation-loss checkpointing and frozen held-out evaluation.
   - Exact tokenizer alignment between materialization, training, and serving.

5. **Research UI**
   - Standard and long-form capacity modes.
   - Visible evidence coverage, citations, source identities, and incomplete-coverage states.
   - Fail-closed behavior when the active model or index does not match its release binding.

## Repository layout

```text
config/                 Portable example configuration
guidebook/              Canonical policy and protocol documents
scripts/                Ingestion and DGX Spark orchestration tools
src/giis_prod/          Pipeline implementation
tests/                  Deterministic policy and implementation tests
giis_pipeline.py        Consolidated command-line entry point
```

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
cp config/settings.example.toml config/settings_v17_turbo.toml
cp .env.example .env
```

Model serving and QLoRA training require the optional runtime dependencies and compatible accelerator drivers:

```bash
python -m pip install -e '.[serve,train-4bit]'
```

Supply credentials only through environment variables. In particular, use `GIIS__METADATA__OPENALEX_API_KEY` for OpenAlex, `TEACHER_API_KEY` for the teacher endpoint, and `VLLM_API_KEY` for the inference endpoint. Real `.env` files are ignored.

## Typical workflow

```bash
# Inspect available commands
python giis_pipeline.py --help

# Run the test suite
pytest
```

For the current deployment commands and execution-binding refresh procedure, see [the deployment guide](guidebook/DEPLOYMENT_AND_SYNC.md). Linux is required for the DGX orchestration and file-locking paths.

Copy `config/settings.example.toml` before a run and adapt corpus locations, model identifiers, scale controls, and deployment ports. Run-specific selection manifests and frozen release bindings are generated locally and must remain associated with their corresponding corpus and artifacts.

## Canonical policies

- [Parsing, metadata, embedding, and hybrid-index policy](guidebook/PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md)
- [Full-corpus metadata adjudication and ingestion policy](guidebook/METADATA_INGESTION_POLICY.md)
- [V17-Turbo policy](guidebook/V17_TURBO_POLICY.md)
- [V17-Turbo execution protocol](guidebook/V17_TURBO_PROTOCOL.md)
- [V17-Turbo split-specific semantic routing and retry contract](guidebook/V17_TURBO_UNIVERSAL_OSS_FIRST_AMENDMENT.md)
- [Reviewer benchmark and hybrid semantic-review rationale](guidebook/REVIEWER_BENCHMARK_AND_HYBRID_SEMANTIC_REVIEW.md)
- [UI policy](guidebook/UI_POLICY.md)

The policy documents define the canonical behavior. Configuration and executables must implement those controls without weakening release gates.

## Data and model governance

This repository does not grant redistribution rights for source papers or third-party model weights. Operators are responsible for the licenses and access terms governing their corpora, models, APIs, and generated releases.
