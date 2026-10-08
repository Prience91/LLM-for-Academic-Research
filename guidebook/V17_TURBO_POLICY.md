# GIIS V17-Turbo policy


The raw-corpus-to-index boundary is governed by `PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md`; V17-Turbo consumes only its frozen, validated index binding.
Status: canonical self-contained synthetic-dataset policy architecture. It governs the complete
Qwen-122B generation, deterministic qualification, and split-specific semantic-verification
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
9. run OSS-120B coverage and support review on every training and validation lane, retrying failed axes once;
10. freeze completed OSS semantic rejects in a blinded gold queue, then author and deterministically qualify all held-out lanes into a second blinded gold queue;
11. pause at one hash-bound checkpoint for explicit authorization of both gold reviews; and
12. after authorized gold dispositions, quarantine, hold, discard, or mark candidates eligible for separately authorized release or benchmark freeze.

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
a complete gold queue for all lanes, the explicit gold-authorization checkpoint, gold review of every
generated reference, disposition binding, and final benchmark freeze. Every stage is
checkpointed and resumes only against exact selection, allocation, candidate, packet, configuration,
implementation, and model identities.

The canonical gate requires all 100 planned references in the exact 71 single-source, 25 multi-source,
two-table, and two-comparison allocation. An explicitly authorized run-local partial pre-review
materialization is an exception artifact only. Terminally exhausted authoring slots are now
recorded as an explicit shortfall while every fully generated, qualified reference proceeds to the
gold queue. This preserves review continuity but does not satisfy the canonical 100-case final
benchmark gate or establish a new minimum.

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

Held-out review is gold review for every single-source, multi-source, table, and comparison candidate.
The automatic pipeline prepares a content-addressed full-population gold queue and stops at
`awaiting_codex_gold_review_authorization`. Gold review requires explicit operator authorization.
OSS review is not a held-out prerequisite or admission path. Only a deterministically qualified,
gold-complete, fully supported reference may enter the benchmark. Minor omissions remain held for
authorized repair and deterministic plus gold requalification; material incompleteness, unsupported
content, missing review, or routing mismatch fails closed. Multi-source and creative cases use only
held-out documents. The main all-lane OSS census and its blinded semantic-rejection queue must be
complete and hash-bound before held-out authoring begins. Main gold decisions are not a prerequisite.
The automatic single-launch path then freezes the full held-out gold queue and pauses once for
operator authorization of both queues; it never starts gold assessment automatically.

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
Fewer than two from ordinary quality-refill exhaustion fails that paper closed. If an authoring
request itself exhausts its one retry, that request is terminally quarantined and the paper may
contribute any already qualified candidates even below the usual floor; other papers continue.

Candidate-quality failure does not receive an individual retry. Transport, timeout, server,
empty-response, truncated completion, and structured-transport failures receive exactly one retry and do
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

Each new execution owns its candidates, checkpoints, reviewer packets, decisions, and manifests under
its own output namespace. Shared immutable corpus and model files are allowed. Candidate identity,
source ownership, and plan fingerprints must match before reuse. Frozen historical runs cannot be
resumed under the new `split_review_v2` routing; use a fresh plan and namespace.

Qwen-122B serves generation at `127.0.0.1:8800`. For main training and validation, automatic handoff
freezes packets for every Python-passing single-source, multi-source, table, and comparison candidate,
stops Qwen, verifies port closure and 100 GiB available memory, starts OSS-120B at
`127.0.0.1:8600`, and runs both semantic axes for every packet. Servers remain mutually exclusive.
Only completed OSS semantic rejects enter a blinded content-addressed gold queue. Exhausted OSS execution failures are terminal quarantine entries with no repair or gold salvage.
The pipeline then stops for explicit gold-review authorization. OSS and gold outcomes are preserved
separately. Missing packets, foreign identities, incomplete axes, or server uncertainty fail closed.

Held-out authoring instead freezes a gold queue containing every generated reference across all lanes,
releases Qwen, and stops for explicit gold authorization. It does not start OSS. The canonical wrappers
require an explicit fresh output namespace. Review and finalization verify hashes and authorization.

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

Training and validation: every Python-passing candidate in all four lanes receives GPT-OSS-120B
component-mapped dual-axis review first. Complete coverage and supported claims admit through
`oss_direct_complete`. Only completed OSS semantic rejects (minor omission, material incompleteness, or unsupported content) enter the blinded gold queue. Fail-closed review errors receive one OSS retry; exhausted errors are terminal quarantine entries with no repair or gold salvage. Authorized gold-complete decisions salvage through
`gold_salvaged_after_oss_non_admission`; minor omissions are held for authorized repair; material
incompleteness and unsupported content are discarded. Repaired main candidates repeat deterministic,
OSS, and required gold qualification with a new hash and preserved parent identity.

Held-out: every generated candidate receives authorized gold review. Eligible references require all
deterministic gates and a gold-complete decision through `gold_direct_complete`. Repaired references
repeat deterministic and gold qualification. OSS does not determine held-out eligibility.

Gold queues bind the original question, evidence, obligations, answer, candidate hash, lane, and split.
Machine verdicts remain hidden during gold decisions. Every queued case needs a final disposition;
unreviewed candidates block completion. An authoring failure without an assessable candidate is recorded
separately and does not satisfy the full allocation. Semantic rejection never triggers automatic regeneration.

Eligibility is separate from release authorization. Schema-v2 training releases require a hash-bound
`main_semantic_review_manifest.json`, matching source plan, and per-row candidate hash and admission
path. Benchmark manifests require a hash-bound `heldout_semantic_review_manifest.json`, complete gold
census, and exact reference/question identity. No legacy manifest silently satisfies the revised gates.
The student tokenizer, 8,192-token sample limit, training-selection provenance, and explicit training
release authorization remain required.

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

## Fail-closed execution contract

Fail-closed execution outcomes are distinct from confident semantic rejects. Qwen-122B retries a failed
or incomplete authoring request once: at most two total attempts. Only fully generated, deterministic-
passing candidates reach OSS or gold. The standard quality-refill allocation for papers remains separate;
it cannot reopen an exhausted authoring failure. SDK retries are disabled. Attempt counters and terminal
markers persist across resumes. Exhausted Qwen failures are quarantined/archived with no repair or
subsequent semantic review.

OSS-120B makes one initial coverage/support pass and one retry pass for failed axes only. Successful
axes are retained. At most two attempts per axis are permitted across resumes. Gold salvage requires
both axes to have completed successfully with a semantic rejection (`minor_omission`,
`materially_incomplete`, or `unsupported`). An exhausted OSS fail-closed outcome is terminal quarantine,
with no repair, gold salvage, or additional retry.

Gold fail-closed outcomes receive no retry. They await explicit operator authorization for quarantine
or archival; that authorization binds the candidate census and hashes. They remain ineligible and cannot
be treated as confident scientific rejections or repaired candidates. This rule is distinct from
explicitly authorized surgical repair of a completed gold `minor_omission` verdict.

## Single-launch pre-gold orchestration

For a fresh frozen main plan and a fresh held-out namespace,
`scripts/run_v17_turbo_pre_gold_pipeline.sh <main_run> <heldout_run>` owns the normal-path
server lifecycle and runs main train/validation generation through all lanes, bounded authoring
retries, deterministic qualification, OSS-120B dual-axis review and its bounded failed-axis retry,
then held-out Qwen authoring and deterministic qualification in every lane. The main OSS census
and terminal failure quarantine are validated before held-out authoring. Both blinded gold queues
are frozen into `joint_gold_authorization_checkpoint.json`; this is the first planned operator pause.
One explicit authorization statement can be bound to both exact queue hashes with
`python3 -m giis_prod.v17_turbo_pre_gold authorize`. Neither gold review is executed by the
launcher. Exhausted per-request Qwen failures and exhausted per-axis OSS errors are terminally quarantined
without stopping other qualified candidates. Creative qualification uses the surviving paired-paper
bundles and records any planned-case deficit. Held-out terminal authoring slots likewise leave an
explicit deficit while surviving references reach the joint gold checkpoint. Unresolved server,
corpus, plan, or identity failures still stop the run for diagnosis. An underfilled held-out queue
may receive gold review, but cannot satisfy the canonical 100-case final benchmark gate.
