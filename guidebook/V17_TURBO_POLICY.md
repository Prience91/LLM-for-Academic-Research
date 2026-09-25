# GIIS V17-Turbo policy


The raw-corpus-to-index boundary is governed by `PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md`; V17-Turbo consumes only its frozen, validated index binding.
Status: canonical self-contained synthetic-dataset policy architecture. It governs the complete
Qwen-122B generation, deterministic qualification, and GPT-OSS-120B single-source semantic-verification
workflow. Dataset generation still requires a frozen plan; training release and training remain
separately authorized operations.

Root: `/home/codeaccounting/Documents/Mini run`

## Authority and independence

`V17_TURBO_POLICY.md`, `V17_TURBO_PROTOCOL.md`, `config/settings_v17_turbo.toml`, and the
`v17_turbo_*` modules contain the complete operative architecture. Earlier policy documents are
historical evidence only. They are not normative dependencies and may not supply a missing rule,
threshold, prompt, disposition, or authorization.

V17-Turbo has its own run namespace beneath `artifacts/v17_turbo`, its own partition and import
manifests, its own content-addressed plan, and its own builder, coverage layer, arithmetic contract,
qualification runner, and canary runner. It may reuse generic pipeline utilities and immutable corpus
content, but it imports no earlier versioned policy module. A candidate from another fingerprint may
enter a V17-Turbo release only through an explicit cross-policy qualification process.

## Teacher and student identities

Qwen3.5-122B-A10B-GPTQ-Int4 is the generation and held-out-reference teacher.
Qwen3.8-27B is the SFT student and the value of `models.base`; its tokenizer defines assembled sample
length, the 8,192-token capacity check, completion masks, training, packaging, and student evaluation.
The teacher and student identities must remain separate in configuration and release provenance. Every
final release is retokenized with the exact student snapshot through the training chat-template path.
A tokenizer change creates a new immutable child release and requires all capacity checks to pass again.

## Objective and stage boundary

V17-Turbo produces evidence-grounded supervised fine-tuning candidates that answer every material
requirement created by their question without unsupported expansion. It preserves a compact,
auditable evidence-to-obligation and obligation-to-answer mapping. Answer length follows supported
substance: the teacher stops when all supported obligations are complete, without padding and without
compressing a complete answer merely to satisfy a response label.

The workflow is:

1. isolate and audit the corpus;
2. freeze held-out, train-generation, and validation-generation partitions;
3. construct boundary-complete evidence bundles;
4. freeze a content-addressed plan;
5. generate single-source prose candidates;
6. run deterministic Python verification and bounded refills;
7. generate paired multi-source prose candidates;
8. qualify table and comparison variants;
9. run independent semantic review as the final quality judgment;
10. quarantine, hold, discard, or mark candidates eligible for a separately authorized release; and
11. maintain a separately frozen, one-shot held-out benchmark whose references are authored by
    Qwen-122B and reviewed according to candidate type.

Python verifies structure and mechanical grounding. It never establishes scientific entailment or
semantic completeness. Semantic review is final and does not rewrite or regenerate a candidate.

## Corpus and split

The representative canary freezes exactly 1,000 accepted SHA-256-distinct papers. The raw RAG intake
may be overprovisioned to absorb duplicates, quarantine, or irreparable metadata. The current intake
contains 1,250 files and 1,195 unique hashes before reconciliation. Surplus accepted papers remain
outside the frozen canary. One hundred papers, exactly 10% of the frozen canary, are held out and
unavailable to generation or training selection. The remaining 900 consist of
810 train-generation and 90 validation-generation papers. Corpus registry, chunks, metadata audit, split,
source hashes, model identity, implementation, configuration, benchmark, and output namespace bind
into the plan digest.

`RAG Corpus` is the sole raw-paper intake directory and may contain the verified overprovisioning pool.
The frozen 1,000-paper selection drawn from it includes the 100 held-out papers required for final
retrieval evaluation. `Training Corpus` is the only
generation/training source and is materialized with exactly the 900 non-held-out papers after the
split is frozen. It is
a verified subset of the RAG corpus, not an identical mirror. `Papers_Imported` remains a dormant
source vault for controlled future expansion and is never ingested directly.

A copied paper may carry the deterministic collision-protection prefix `hhhhhhhhhhhh__`, where every
`h` is a hexadecimal digit. That exact prefix is storage identity rather than title text: ingestion
strips it before filename-based title selection and external metadata queries, while preserving the
complete filename in `source_file`. Similar-looking prefixes that do not match this exact contract are
left intact. The normalization event is recorded as `corpus_filename_hash_prefix_removed`.

## Pre-embedding metadata reconciliation

Bibliographic identity is resolved before embedding. Ingestion performs cached DOI-first and title-based
lookups against Crossref and OpenAlex, records the provider response, match score, provenance, and
lookup warnings, and accepts only a verified, unambiguous match with complete required metadata. A
provider timeout or rate limit is an infrastructure outcome rather than evidence that the paper lacks
metadata; the lookup remains retryable from cache-aware reconciliation. OpenAlex uses a project-specific
key supplied only through `GIIS__METADATA__OPENALEX_API_KEY`; the key is never written to configuration,
artifacts, logs, or manifests. A 429 response opens a provider circuit breaker for the advertised
`Retry-After` interval instead of generating repeated requests. Conflicting, incomplete, or ambiguous
results remain in the metadata review queue. They may advance only through corroborating
provider evidence or a source-bound verified override. Thresholds are never relaxed merely to satisfy
the corpus count.

Embedding and both index builders remain blocked until the quarantine ceiling, accepted-document floor,
SHA-256 uniqueness requirement, and corpus audit all pass. Raw provider responses and reconciliation
decisions are retained so the resulting document identity can be reproduced.

## Retrieval-index binding

The 1,000-paper canary requires a newly built V17-Turbo-owned hybrid retrieval index over every
accepted chunk from the frozen corpus. Exact document and chunk counts, corpus-version identifiers,
and content digests are frozen only after parsing, metadata audit, chunking, embedding, and lexical
index construction complete. The architecture admits content-addressed corpus expansions through
20,000 papers without changing this policy. Dense retrieval uses Chroma embeddings from
`ibm-granite/granite-embedding-30m-english`; lexical retrieval uses the paired SQLite FTS5 index.
The vectors may be materialized from a prior index only when the document and chunk registries are
byte-identical and the corpus-version, chunk-count, collection, embedding-model, and index audits all
match. The materialized copy must live beneath `artifacts/v17_turbo/index` and must not depend at
runtime on an earlier policy directory.

`active_index.json` selects the private dense and lexical artifacts. `index_binding.json` binds their
lineage, the document and chunk hashes, the immutable Chroma vector-payload digest, the lexical database digest,
and the absence of a runtime dependency. Both manifests are part of the frozen plan fingerprint. Any
corpus, embedding, collection, path, or artifact-digest drift invalidates the binding and requires a
validated rebuild or rematerialization before retrieval, benchmark evaluation, or deployment. Each
expansion receives a new corpus version, registry hashes, exact document and chunk counts, dense and
lexical artifact digests, and plan fingerprint. The prior index remains immutable and rollback-safe.
An expansion may append verified documents or rebuild the index, but it may not reuse an old binding
for changed corpus content.

### Obligation-aware passage assembly

The hybrid dense, lexical, and reranking stack remains the source-discovery foundation. After query
rewriting, deployment deterministically preserves every explicit question clause. A preliminary hybrid
retrieval then supplies corpus passages from which the planner derives three to ten atomic,
independently searchable evidence obligations. Broad requests for methods, findings, mechanisms,
factors, barriers, limitations, comparisons, or implications must be split into their substantive
evidence-backed components. It must use only the question, permitted dialogue history, and preliminary
corpus passages; benchmark references, hidden rubrics, training answers, and reviewer feedback are
prohibited.

Each obligation receives its own hybrid retrieval and a reserved evidence position before global rank,
source diversity, or redundancy filtering fills the remaining context. Ordinary retrieval retains the
three-chunk per-document ceiling, while obligation-aware assembly may retain up to ten substantively
distinct chunks from one document within the unchanged twelve-chunk and 6,144-token context limits.
This prevents source-diversity policy from suppressing different answer components located far apart in
the same paper.

Before answer generation, a semantic coverage validator maps every question-derived obligation to exact
evidence IDs. One bounded expanded retrieval pass may rescue an uncovered obligation. The complete map,
including any unresolved requirement, is passed to the generator. The generator must address every
covered obligation and explicitly report an evidence limitation for an uncovered obligation rather than
supplying remembered or outside knowledge. Evaluation records the obligations, evidence map, aggregate
and per-case coverage, retrieval-pass count, and whether atomic planning succeeded. Planner failure may
fall back to the preserved explicit clauses for safe retrieval, but that fallback is never reported as
successful atomic coverage. If an answer stops at its completion-token ceiling, it is rejected and receives
one repair at the long-form capacity. A repair that also stops at its ceiling fails closed.

## Locked held-out benchmark

Held-out authoring and review are part of the V17-Turbo workflow itself. They are executed through the
canonical `heldout-pipeline` state machine rather than administered as an external follow-up. The state
machine performs Qwen authoring, inline Python qualification, immutable reviewer-packet construction,
the guarded Qwen-to-OSS handoff, the full OSS single-source census, Codex-gold routing for every
multi-source and creative case, disposition binding, and final benchmark freeze. Every stage is
checkpointed and resumes only against exact selection, allocation, candidate, packet, configuration,
implementation, and model identities.

The canonical gate requires all 100 planned references in the exact 71 single-source, 25 multi-source,
two-table, and two-comparison allocation. An explicitly authorized run-local partial pre-review
materialization is an exception artifact only. It does not amend this policy, establish a new minimum,
or satisfy the canonical full-count benchmark gate.

The 100 held-out papers form a sealed, one-shot final benchmark. They never supply SFT candidates,
training-time validation candidates, repair evidence, checkpoint-selection evidence, prompt-tuning
evidence, or policy-tuning feedback. The 90 validation-generation papers provide the development
signal used during SFT; no development subset may be carved from the held-out papers. If a held-out
result influences the model, retrieval configuration, prompt, threshold, or policy, that benchmark is
spent and a newly sampled untouched final set is required.

Qwen3.5-122B-A10B-GPTQ-Int4 is the canonical held-out reference generator. From frozen held-out
evidence it creates the question, candidate type, atomic coverage obligations, exact supporting
passages and source identities, semantic guardrails, and a concise evidence-backed reference answer.
It must not see a student answer while authoring or revising a benchmark case. References and
obligation rubrics remain hidden from the evaluated student; scoring measures obligation satisfaction
and evidential support rather than similarity to Qwen wording.

Held-out review is routed by candidate type. GPT-OSS-120B performs the complete dual-axis semantic
review of every single-source case. Codex gold review is mandatory for every multi-source, table, and
comparison case. Only a `complete` and fully supported case may enter the frozen benchmark. A minor
omission is held for repair and full requalification; material incompleteness, unsupported content, a
review error, missing axis, source-boundary violation, or routing mismatch fails closed. Multi-source
and creative cases may use only held-out documents and may never mix a held-out paper with a training
or validation-generation paper.

The frozen benchmark manifest binds document IDs and hashes, questions, candidate types, evidence,
obligations, references, reviewer identities and verdicts, case order, and all artifact digests. The
evaluation model receives only the question and permitted retrieved evidence. Final held-out results
may be reported but cannot be used for tuning.

The September 2026 eight-case set has supplied retrieval-policy feedback and is therefore spent as an
unbiased final benchmark. A rerun of those same cases is permitted only as a paired diagnostic regression
test and must be labelled accordingly. It cannot re-establish the final benchmark gate; that requires a
newly sampled and sealed held-out set.

Every single-source bundle contains at most eight substantive chunks and requires at least four.
Support units use stable IDs, contain complete sentence boundaries, retain source-document ownership,
and exclude unmatched trailing fragments. Paired-paper evidence binds both paper identities before
generation and is never silently truncated.

## Candidate schema and coverage

Every primary candidate contains:

- a question fully answerable from the supplied evidence;
- a compact list of material coverage obligations;
- for each obligation, its evidence marker and stable support-unit IDs;
- ordered answer units mapped to obligation IDs;
- a response mode; and
- an answer style.

The final answer is materialized once by joining the ordered answer units. Every obligation maps to at
least one answer unit. Every answer unit cites evidence supporting each obligation assigned to it.
Obligation IDs and answer-unit IDs are unique. Support IDs must exist and belong to the declared
evidence marker.

The teacher should prefer a coherent evidence-backed theme and generally three to six substantive
obligations when the evidence supports them. Obligation count is diagnostic, never a numerical pass
condition. The teacher must narrow a question when the evidence cannot support every requested part.

## Candidate allocation and retry protocol

Each of the 900 generation papers receives an initial request for four single-source candidates. Python
retains at most four and prefers at least three. If fewer than three pass, the engine may issue two
refill requests of two candidates each, supplying accepted questions as novelty exclusions. After the
bounded request budget, two fully valid candidates may be released with an explicit underfilled event.
Fewer than two fails the paper closed.

Candidate-quality failure does not receive an individual retry. Transport, timeout, server,
empty-response, and structured-transport failures receive at most two infrastructure retries and do
not consume quality refills. Two consecutive zero-yield requests before the preferred minimum stop
work on that paper. Successful paper checkpoints remain resumable under the exact same plan digest.

## Token-capacity lanes

Capacity is measured from the materialized answer with the frozen teacher tokenizer:

| Actual answer length | Assigned lane | Deterministic action |
| --- | --- | --- |
| 1-900 tokens | `standard` | retain the standard label or normalize to it |
| 901-2,200 tokens | `long_form` | automatically promote to `long_form` |
| More than 2,200 tokens | none | reject as `answer_exceeds_long_form_capacity` |

The teacher-supplied label is advisory. Python assigns the final lane from actual answer-token count
before quota accounting and audit output. There is no minimum, target, or maximum word count. A short
complete answer is valid; a longer answer is valid only when its additional content is relevant,
supported, and required by the question.

The teacher API and verified server profile both use `127.0.0.1:8800`; planning fails if their host or port differs. The server context remains 16,384 tokens. Rendered teacher input is limited to 16,000 tokens and the
structured teacher completion allowance remains 8,000 tokens per request. The complete assembled
student training sample is limited to 8,192 tokens. Evidence is compacted by candidate-specific
selection before generation; it is never clipped during assembly. Any assembled sample beyond 8,192
fails deterministic verification.

Before any SFT training at 8,192, a one-optimizer-step worst-case QLoRA canary must use the longest
eligible samples and demonstrate adequate unified-memory headroom. Passing generation alone does not
authorize training.

## Deterministic Python verification

Python performs, in order:

1. strict schema parsing and one-time answer materialization;
2. non-empty question and answer checks;
3. tokenizer-based response-mode normalization and the 2,200-token answer ceiling;
4. lane-specific answer-style enforcement;
5. exact evidence-marker syntax and range checks;
6. rejection of non-corpus secondary citations presented as direct evidence;
7. at least 40% training citation coverage;
8. complete student-chat tokenization and the 8,192-token ceiling;
9. lexical and embedding duplicate-question checks at 0.72 and 0.88;
10. unique and valid obligation, answer-unit, evidence, and support-unit mappings;
11. complete obligation-to-answer mapping with supporting citations per answer unit; and
12. multi-source participation and synthesis rules when applicable.

Every request and candidate records raw output, decision, reason, request number, elapsed time, token
telemetry, server finish reason, source identity, plan digest, and checkpoint state. A Python pass means
only that deterministic checks passed.

## Multi-source and structured allocation

Multi-source volume equals 35% of 3,600 planned core drafts under the two-candidate bundle
contract: 1,260 candidates from 630 paired-paper bundles. Each candidate assigns at least one material
obligation to each paper, cites both substantively, and performs genuine comparison or synthesis.
It preserves differences in method, scale, population, timeframe, and outcome and does not convert
juxtaposition or association into an unsupported causal claim.

After the primary Python pass, the engine selects one passing multi-source prose source from each of
101 distinct bundles. Holding question, evidence, sources, and obligations fixed, it creates one
Markdown table and one point-by-point comparison from each: 101 tables and 101 comparisons.
These 202 variants do not satisfy core minima or multi-source allocation. Each passes independent
schema, formatting, source-use, citation, context, and safety checks. Failure affects only the variant
and never triggers regeneration.

## Continuous execution and integrity

Two workers form a continuously replenished queue. A newly available slot immediately receives the
next eligible paper. Candidate validation and checkpoint mutation remain serialized and final output
uses frozen document order. A pause marker stops new dispatch, and a run lock prevents duplicate
execution. Resume requires exact configuration, source, implementation, and plan identity.

No threshold, prompt, model setting, split, evidence rule, or disposition changes inside a frozen run.
Any amendment requires a new configuration fingerprint and output namespace.

## Canonical model orchestration and automatic review handoff

Every V17-Turbo execution uses a new output namespace. Core checkpoints, raw journals, materialized
passing candidates, creative checkpoints, semantic packets, results, and status records are local to
that namespace. Immutable corpus and model files may be shared; generated candidates and checkpoints
may not. The materialized passing-candidate directory is an exact view of the current accepted set:
resume residue is removed before downstream selection. A creative checkpoint is reusable only when
its plan digest, case ID, format, parent example ID, bundle ID, and document IDs all match.

Qwen3.5-122B-A10B-GPTQ-Int4 is the canonical V17-Turbo teacher. It serves only on the reserved
generation endpoint `127.0.0.1:8800`, with the frozen 16,384-token context and two-sequence profile.
GPT-OSS-120B is the canonical single-source semantic reviewer. It serves only on the reserved review
endpoint `127.0.0.1:8600`, with the frozen component-mapped dual-axis protocol. The two model servers
must never be resident concurrently.

The automatic handoff is part of the policy:

1. complete and checkpoint all single-source, multi-source, table, and comparison generation;
2. verify the active plan digest, final passing-file count, 202-case structured-qualification manifest, and provenance
   of every materialized passing candidate;
3. freeze blinded review packets for every Python-passing single-source candidate, including packet
   count and SHA-256 digest;
4. stop Qwen-122B, require port 8800 to close, and require at least 100 GiB available system memory;
5. require review port 8600 to be unoccupied, then start GPT-OSS-120B and verify its health;
6. review every single-source packet on both coverage and support axes, retrying malformed axis output
   only within the frozen reviewer retry allowance; and
7. freeze the census summary, per-axis records, dispositions, errors, packet digest, and plan digest,
   then release the reviewer server.

A port collision, model-profile mismatch, foreign plan digest, candidate-count mismatch, incomplete
creative manifest, foreign passing file, failed server transition, malformed review, or unresolved
review error fails closed. It cannot silently reduce the census or admit a candidate. Multi-source,
table, and comparison candidates remain routed to gold review and are never admitted by the OSS
single-source census.

## Independent semantic review

Semantic review receives the complete question, numbered evidence, support units, declared compact
ledger, and materialized answer. The reviewer independently reconstructs the material obligations and
must not assume that the teacher ledger is exhaustive or correct.

Review judges factual support, citation-to-claim entailment, explicit and implied completeness, scope,
population, method, units, comparison dimensions, uncertainty, source ownership, unsupported
expansion, causal overstatement, guardrails, and every substantive table cell or comparison point.
There is no numerical omission tolerance.

Verdicts are:

- `complete`: all material requirements are supported and answered;
- `minor_omission`: only a secondary detail is missing; hold for possible repair and fresh review;
- `materially_incomplete`: a requested component or necessary qualification is absent; discard; and
- `unsupported`: a material claim exceeds, contradicts, or misattributes evidence; discard.

Unreviewed is a workflow state, not a semantic verdict. Semantic rejection does not trigger teacher
regeneration.

## Reviewer routing and dispositions

Every Python-passing single-source candidate is reviewed by the frozen GPT-OSS-120B component-mapped
dual-axis protocol. No single-source candidate receives gold review under this policy. GPT-OSS accepts
become eligible for later operator release consideration; rejects and fail-closed errors enter
quarantine. The generation programme may overprovision the source corpus by 40% to absorb conservative
single-source rejection without weakening the gate.

Every Python-passing multi-source, table, and comparison candidate receives gold semantic review.
These lanes bypass machine admission because hidden contamination is costlier than visible quarantine
and their cross-source or structured reasoning is more difficult to gate reliably. Gold `complete`
results become eligible; `minor_omission` results remain held; `materially_incomplete` and
`unsupported` results are discarded.

Eligibility is not a training release. Release requires a separate immutable selected-row manifest,
policy and source fingerprints, hashes, exclusions, split counts, audit records, and explicit operator
authorization.

The canonical release uses dataset-release schema version 2. Its manifest binds the immutable train,
validation, rejected, consolidated, and admission files; the V17-Turbo policy fingerprint; reviewer
authority; release guards; and a frozen training-selection manifest describing only papers represented
by admitted examples. Zero-yield generation papers are recorded explicitly and are not counted as
represented training documents. The SFT, training-selection, and materialization paths must resolve to
the same release directory. Activating this dataset pathway does not authorize or start training.

## Production locks

Before scaling beyond a representative canary, record passing evidence for:

- multi-source checkpoint resume under interruption;
- minimum passing multi-source yield and quarantine behavior;
- observed answer-token lane distribution and automatic promotion;
- the 8,192-token worst-case QLoRA memory canary;
- representative theme and document-quality coverage;
- reviewer-server stability and fail-closed behavior;
- frozen benchmark thresholds;
- storage, runtime, and recovery preflight; and
- cumulative rejection, quarantine, and overprovisioning accounting.

A missing or false gate blocks larger generation. Flags may not be enabled merely to bypass a lock.
