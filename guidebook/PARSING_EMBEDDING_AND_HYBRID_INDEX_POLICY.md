# GIIS parsing, metadata, embedding, and hybrid-index policy

Status: canonical component policy for the Mini-run V17-Turbo pipeline.

This policy governs the path from raw scientific PDFs to the immutable dense and lexical indexes used by dataset generation, held-out evaluation, and deployed RAG. It is independently executable within the Mini-run workspace. V17-Turbo consumes its frozen outputs but may not weaken its identity, quarantine, audit, or index-binding gates.

## 1. Intake and corpus target

`RAG Corpus` is the sole raw-paper intake. It may be overprovisioned to absorb duplicates, corrupt files, unresolved metadata, and quarantine. Every intake wave has an immutable selection manifest containing source path, destination filename, SHA-256, byte size, selection order, seed, and manifest digest.

The initial planning target was 1,000 accepted SHA-256-distinct documents. Raw file count is not corpus count. The active initial index release contains all 1,007 presently verified documents; its exact count and hashes are frozen in the release manifests. Later verified additions create a new content-addressed release up to the supported 20,000-paper scale. The current intake contains 1,250 files and 1,195 unique hashes: 1,000 original files with 945 unique hashes plus 250 genuinely unseen additions.

Collision-safe filenames use an exact twelve-hexadecimal-character plus double-underscore prefix. Ingestion strips only that prefix for title and metadata lookup, preserves the complete filename as `source_file`, and records `corpus_filename_hash_prefix_removed`.

## 2. Manifest-bound incremental parsing

A new intake wave is parsed from its manifest rather than by rescanning previously completed papers. Its registry, chunks, review queue, failures, journal, and status remain isolated until qualification. Each paper is checkpointed, and resume requires the same selection-manifest digest. The established registry is immutable during the incremental pass.

Parsing uses Docling with native extraction first and OCR fallback when native text has fewer than 1,000 characters or fewer than two usable chunks. Malformed PDF date strings are read as raw values; a conversion error cannot abort an otherwise parseable paper.

## 3. Bibliographic reconciliation

Bibliographic identity must be verified before embedding. The resolver uses this order:

1. source-bound verified override, when one already exists;
2. DOI extracted from trusted PDF metadata or the title pages, checked against Crossref and OpenAlex;
3. normalized-title searches against Crossref and OpenAlex;
4. independent-provider corroboration and document-local title-page evidence; and
5. Codex adjudication for residual ambiguous or incomplete cases.

Provider responses are cached and retain provider identity, lookup key, retrieval time, response, match score, provenance, and warnings. The normal title-similarity threshold is 0.93, the ambiguity margin is 0.03, and a verified distinctive fragment requires at least six tokens and 40 characters. Thresholds may not be lowered merely to meet the target count.

A verified record requires a title, publication year, and authors or a defensible issuing organization. Once verified, this canonical bibliographic title is authoritative. The source filename is retained as a tentative provenance label and may trigger a non-blocking audit warning, but it cannot replace the verified title or exclude the record. Generic-title matches, conflicting DOI/title combinations, implausible years, incomplete author fields, and ambiguous candidates remain quarantined.

### OpenAlex authentication and quota handling

OpenAlex uses a project-specific key supplied only through `GIIS__METADATA__OPENALEX_API_KEY`. The key must never be committed, written into TOML, placed in manifests, or emitted in logs or artifacts. Cached anonymous results remain reusable after authenticated operation begins.

The client reads quota headers and honors `Retry-After`. A 429 response opens a provider circuit breaker for the advertised interval; repeated requests are not issued while the circuit is open. Provider exhaustion is an infrastructure state, not evidence that a paper lacks metadata. Crossref work may continue while OpenAlex is deferred, but unresolved records cannot be treated as corroborated.

## 4. Codex adjudication and verified overrides

Codex is the gold metadata reviewer for cases deterministic reconciliation cannot settle. Each packet must contain the PDF title-page evidence, source filename and SHA-256, extracted DOI, Crossref and OpenAlex candidates, title scores, years, authors or issuing organization, conflicts, and provider provenance.

Codex may issue a source-bound verified override only when the identity is supported by the PDF and authoritative bibliographic evidence. It must not infer missing values from topic similarity or approve the nearest search result. The override binds the source SHA-256 or exact source filename and records title, authors or issuing organization, year, DOI when present, document type, provenance, reviewer, and decision time. Unresolved papers remain quarantined or are replaced.

## 5. Merge and canonical freeze

The baseline and every qualified increment are merged by SHA-256, never by filename alone. A document enters the candidate registry only when it is verified, has usable chunks, and has complete required provenance. Duplicate hashes, review-required records, rejected records, parsing failures, and documents without usable chunks are excluded.

The merge produces a deterministic accepted-pool manifest. Each active index release freezes its exact verified selection, document registry, and chunk registry before audit or indexing. The initial active release freezes all 1,007 presently verified unique documents; unresolved and failed records remain outside it. No document may be silently added to an existing content-addressed release.

## 6. Chunk construction

Chunks use the Granite embedding tokenizer with a maximum of 512 tokens, 64-token overlap, and a minimum of 120 characters. Oversized units are split without losing forward progress. Every chunk binds its document ID, title, authors, year, DOI, publication fields, section, pages, ordinal, token count, and source provenance. No evidence text is clipped during later candidate assembly.

## 7. Pre-index corpus audit

Before embedding, the deterministic audit requires:

- the exact release-manifest count of SHA-256-distinct documents (1,007 for the initial active release);
- verified metadata with title, authors or issuing organization, year, and provenance;
- canonical titles taken from verified metadata, with filename differences retained only as warnings;
- normalized DOI values;
- at least one usable chunk for every accepted document;
- unique document IDs, source identities, hashes, and chunk IDs; and
- chunk metadata identical to its parent document.

Any blocking defect prevents both dense and lexical index creation.

## 8. Dense embedding index

Dense vectors use `ibm-granite/granite-embedding-30m-english`. Embeddings are normalized and stored in a V17-Turbo-owned persistent Chroma collection using cosine distance. The collection name derives from the content version of the complete frozen chunk sequence. Default ingestion batches contain 128 chunks.

An existing collection is reusable only when its count is either zero or exactly equal to the frozen chunk count. A partial collection fails closed. After ingestion, the stored count must exactly equal the chunk-file count.

## 9. Lexical index

The same frozen chunks populate a versioned SQLite FTS5 database. Lexical fields are title, section, and evidence text, while the complete chunk payload and document identity remain available for reconstruction. Dense and lexical indexes always derive from the same ordered chunk file and corpus version.

## 10. Binding and activation

`active_index.json` records corpus version, Chroma collection, Chroma path, lexical database, embedding model, chunk count, and activation time. `index_binding.json` additionally binds document and chunk registry hashes, the immutable Chroma payload/tree digest, lexical database digest, configuration and implementation identities, and the absence of runtime dependence on an earlier project.

Retrieval, held-out evaluation, generation, and deployment may begin only after both manifests validate. Any change to documents, metadata, chunks, embedding model, collection, paths, or artifact digests requires a new content version and a validated rebuild. Previous indexes remain immutable and rollback-safe.

## 11. Failure and recovery rules

Parsing and reconciliation are resumable by manifest digest. Embedding is resumable only through an intact, count-consistent immutable collection; a partial or mismatched collection is rebuilt under a new content-addressed identity. Quarantined documents never enter an active index. Provider failure never converts an unresolved record into an accepted one.

For expansion toward 20,000 papers, each intake wave repeats the same manifest, incremental parsing, reconciliation, merge, audit, content-version, dual-index, and binding sequence. Scale changes counts, not verification standards.

## Current operational checkpoint

The initial active release contains 1,007 verified documents and 55,002 chunks. Its pre-index audit passed with zero blocking errors. The remaining unresolved or failed source records are deferred for later adjudication and final corpus audit; they are excluded from this release and cannot contaminate either index. The release is content-addressed, so later repairs produce a new version without altering this frozen build.
