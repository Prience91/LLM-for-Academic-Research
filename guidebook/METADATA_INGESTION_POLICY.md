# Canonical Full_RAG parsing, metadata adjudication, embedding and indexing

Status: canonical Full_RAG workflow. Updated 2026-10-08. Mini-run remains independent. Existing active indexes and running adjudication jobs are preserved when policy/code is updated.

## Intake and provider qualification

Inventory PDFs in Full_RAG/Full_Corpus by source SHA-256; process each unique content hash once and retain duplicate provenance. Never use filenames as authoritative bibliographic identity. A verified source title supersedes a tentative filename or embedded PDF title.

Check approved source-bound overrides first. Otherwise resolve in this order: CrossRef DOI -> CrossRef title -> OpenAlex DOI -> OpenAlex title. A complete, well-supported CrossRef match is sufficient; OpenAlex is requested only when CrossRef leaves identity or required metadata unresolved. Automatic provider acceptance retains its established checks: provider title/authors/year, matching DOI when applicable, strict title-search agreement and ambiguity margin, and provider title corroborated in the first three source PDF pages. Cached provider results remain usable. A missing DOI is not itself a rejection.

## Budget deferrals are not exhausted metadata cases

OpenAlex budget-floor and 429 outcomes create infrastructure deferrals, not document failures. Record them in openalex_deferred_backlog.jsonl and continue other work. Free DOI requests can proceed at a title-search budget floor unless a provider-wide 429 blocks them. Respect the recorded reset/Retry-After. Process yesterday's/oldest deferred work before fresh provider requests when allowance returns; a long scan revisits stashed work if the circuit reopens. Continuous mode resumes provider cycles automatically; once is diagnostic. Invalid project keys or recurring provider errors remain deferred for operator correction. The installed project-specific key is loaded from the protected OpenAlex environment file, never stored in policy or logs.

A quota-deferred document does NOT advance to Qwen as though all provider lookup opportunities were exhausted.

## English Qwen adjudication after unresolved provider lookup

If provider lookup returns unresolved/incomplete/ambiguous identity after the available sequence, call Qwen3.5-122B-A10B-GPTQ-Int4 through the configured vision endpoint. The endpoint must report the configured served-model identity. Use source text from the opening 12 pages (compacted to 33,000 characters) plus one freshly rendered source page image, with temperature 0, non-thinking mode, and 1,200 output tokens. The canonical prompt is config/metadata_adjudication_prompt.txt; it is the protocol tested in the10- and20-document pilots.

Local qualification is English-only. Clear Nepali/Devanagari text is routed directly to gold review; other non-English/uncertain languages identified by Qwen are also routed to gold. Language detection is a Devanagari pre-screen plus the model language verdict, not a universal pre-call language classifier. Qwen qualification requires a naturally completed structured response with actual title, authors/editors or responsible corporate issuer, publication year, publisher/issuing body, document type, and no unresolved required fields. Missing DOI remains acceptable. Preserve source hashes, request/evidence hashes, responses and reviewer identity.

Confident local qualifications may proceed under the user's accepted resource/quality trade-off. They are labelled local_model_metadata_adjudication, never gold or human verification. Metadata confidence 0 for this method means uncalibrated, not guaranteed correctness; the verified flag records policy qualification. The 30-document pilot found 4 materially erroneous metadata records among 25 locally admitted documents. Reject-only gold review does not detect errors in local admissions. Do not claim100% metadata correctness from qualification status.

Unresolved metadata, malformed/truncated responses, non-English versions, model-identity mismatch and document-local execution failures go to the gold-review queue. Cached referrals are not repeatedly sent to Qwen. The existing 1,421-document source-bound worker may continue and its results are imported without regenerating reviews. Its server memory guard remains separate and active.

## Gold review and final disposition

Gold review does not run automatically. Queued cases carry awaiting_user_authorization. After explicit user authorization, a human/Codex reviewer adjudicates against the retained source PDF and evidence, independently of a mere local verdict. Approved gold decisions are recorded in Parsed_Corpus/metadata/adjudication/gold_review_ledger.jsonl with source SHA-256, source_file, status gold_qualified or gold_rejected, metadata for qualifications, reviewer, reviewed_at and authorization_reference. The importer refuses qualification from this canonical gold ledger without those authorization fields.

Gold qualifications enter the same source-bound promotion bridge as local qualifications. Approved gold rejection adds a persistent manual_human_review_only guard: no automatic retry or promotion; raw sources/evidence remain retained. A subsequent model qualification cannot override a gold rejection. Existing approved gold qualifications take precedence over lower-authority local records. Already completed gold reviews are reused; this integration is not authorization to start new gold review.

## Promotion into parsing

Before each canonical parsing cycle, run python -m giis_prod.metadata_adjudication promote --settings Pipeline/config/settings_full_rag.toml. This imports configured completed provider/gold ledgers, current append-only Qwen results and canonical decisions. Validate the actual PDF SHA-256 and source containment; reject stale/mismatched identities and incomplete required metadata. Merge approved results into verified_overrides.jsonl and the configured selected-parsing manifest. Record ready_for_parsing.jsonl, consolidated gold_review_queue.jsonl and promotion_report.json. Gold rejections take precedence. Keep explicit DOI clearing so a cited or unverified DOI cannot survive through a null adjudicated DOI.

The import_ledgers list contains current migration inputs plus the canonical gold ledger; future projects can use their own paths without depending on earlier smoke-run policies. The live 1,421-document worker and existing gold ledgers retain their separate identities. Policy updates do not themselves start parsing or activate another index.

## Parsing, chunking and hybrid indexing

Only qualified metadata enters substantive Docling parsing. Native-text conversion is attempted first; insufficient extraction uses OCR fallback. CUDA accelerates supported parser operations while CPU orchestration/PDF extraction remain necessary. Preserve section/page provenance and document identity. Use the Granite embedding tokenizer, 512-token maximum chunks and 64-token overlap for split oversized chunks; omit fragments shorter than 120 characters. Keep parser failures/no-usable-chunk outcomes in their distinct resumable/quarantine statuses.

Materialization includes passages only from accepted documents. Metadata-review cases and the 184 historical retroactive deferrals contribute no indexed passages. The historical confirmed baseline and previous index remain auditable.

Embed accepted passages with ibm-granite/granite-embedding-30m-english on CUDA into a content-addressed Chroma collection. Build the SQLite FTS5 lexical database from exactly the same frozen passage registry. Check parent-document acceptance, inherited metadata, unique/nonempty passage IDs, passage/document counts and dense/lexical content-version agreement. For the isolated release, publish its active service binding only after the release audit and index-count checks pass. The current staged-release audit/publisher still encode that release's historical 1,444/184 baseline and paths; this metadata integration does not generalize that separate release machinery.

## Execution references

- production_ingestion.py: inventory, resumable parsing journal, accepted-only materialization and hybrid index construction.
- src/giis_prod/metadata.py: provider ordering, provider acceptance, quota/cache logic and accepted override methods.
- src/giis_prod/metadata_adjudication.py: reusable Qwen fallback, referral persistence, authorization-bound gold imports and promotion bridge.
- src/giis_prod/ingestion.py: provider-first builder, adjudication fallback and Docling/OCR chunking.
- config/settings_full_rag.toml: paths, provider settings and metadata_adjudication endpoint/prompt/import configuration.
- run_full_rag_pipeline.sh and run_forward_qualified_release.sh: promote completed adjudications before future parsing cycles.
