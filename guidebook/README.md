# GIIS framework guidebook

The repository contains reusable code and policy; corpus files, model weights, indexes, review outputs, and credentials remain deployment artifacts.

## Canonical authority

- [Metadata ingestion](METADATA_INGESTION_POLICY.md): source-bound overrides, Crossref first, OpenAlex fallback, non-blocking quota backlogs, English Qwen vision adjudication, authorized gold referrals, and promotion before parsing.
- [Parsing and hybrid indexing](PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md): extraction, chunk provenance, accepted-only Granite/Chroma and SQLite FTS5 construction, audits and index bindings.
- [V17-Turbo policy](V17_TURBO_POLICY.md), [protocol](V17_TURBO_PROTOCOL.md), and [routing/retry contract](V17_TURBO_UNIVERSAL_OSS_FIRST_AMENDMENT.md): training and validation use OSS across all lanes, then authorized gold review of confident semantic rejects. Every generated held-out reference receives direct gold review.
- [Held-out handoff](V17_TURBO_HELDOUT_HANDOFF.md): main OSS census, held-out authoring, and joint gold-authorization pause.
- [UI policy](UI_POLICY.md): obligation completion, evidence coverage and release-bound decoding.
- [Semantic validation failure handling](SEMANTIC_VALIDATION_FAILURE_HANDLING.md): unavailable-review states, no retrieval-only approval, final-bundle validation and pre-generation abstention.
- [Deployment and synchronization](DEPLOYMENT_AND_SYNC.md): portable configuration, execution fingerprints and launch commands.

Exhausted authoring/OSS fail-closed errors are terminal quarantine without gold salvaging. Gold fail-closed outcomes have no retry and require operator authorization for archival or quarantine. No gold review, dataset release, training or index activation is authorized merely by synchronizing this repository.

The representative 1,000-paper plan has 810 training, 90 validation and 100 held-out papers. Held-out papers may remain in RAG but never enter SFT or training-time validation. Every new run freezes its own source identities, allocation, evidence, code/configuration hashes and output namespace.
