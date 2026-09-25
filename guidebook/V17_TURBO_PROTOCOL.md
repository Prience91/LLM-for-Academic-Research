# GIIS V17-Turbo implementation protocol

Status: canonical self-contained executable protocol covering Qwen-122B generation through the
automatic GPT-OSS-120B single-source semantic census. Training release remains a separate,
operator-authorized stage.

Root: `/home/codeaccounting/Documents/Mini run`

## Governing files

The operative files are:

- `V17_TURBO_POLICY.md` and this protocol;
- `PARSING_EMBEDDING_AND_HYBRID_INDEX_POLICY.md` for the complete raw-corpus-to-index boundary;
- `config/settings_v17_turbo.toml`;
- `config/v17_turbo_canary_partitions.json`;
- `src/giis_prod/v17_turbo_layer.py`;
- `src/giis_prod/v17_turbo_builder.py`;
- `src/giis_prod/v17_turbo_contract.py`;
- `src/giis_prod/v17_turbo_qualification.py`; and
- `src/giis_prod/v17_turbo_canary.py`.

Earlier policy files cannot provide missing behavior. Generic tokenizer, citation, style, I/O, teacher
transport, and checkpoint utilities remain shared implementation primitives rather than policy
authorities.

## Stage 1 - prepare and verify inputs

1. Permit an overprovisioned raw intake and deterministically freeze exactly 1,000 distinct
   verified PDFs from the accepted pool. The initial pool contains 1,007 verified records, leaving seven
   content-addressed surplus replacements outside the canary. Verified bibliographic titles are authoritative;
   filenames remain tentative provenance labels. Strip only an exact twelve-hexadecimal-character plus
   double-underscore collision prefix for lookup, retain the full filename as `source_file`, and treat later
   filename/title differences as non-blocking warnings.
2. Reconcile metadata before embedding. Run cache-aware DOI and title lookups through Crossref and
   OpenAlex; preserve raw responses, match scores, provenance, conflicts, and rate-limit outcomes.
   Admit only complete, unambiguous verified matches or source-bound verified overrides. Keep all other
   records quarantined and retry provider infrastructure failures without lowering match thresholds.
   Supply an OpenAlex project key only through `GIIS__METADATA__OPENALEX_API_KEY`; never persist it.
   Honor quota headers and `Retry-After`, and resume from cached responses after the circuit breaker clears.
3. Require at least 1,000 verified SHA-256-distinct documents and pass the deterministic audit of the
   selected release before index activation. Unresolved intake remains excluded for later adjudication and
   cannot block or contaminate the frozen verified release.
4. Freeze 100 held-out documents, 810 train-generation documents, and 90 validation-generation
   documents in the V17-Turbo partition manifest.
5. Verify that held-out identities are absent from all evidence and generation inputs.
6. Materialize V17-Turbo-owned registry, chunk, audit, and held-out records beneath
   `artifacts/v17_turbo`.
7. Build or verify active dense and lexical indexes before any retrieval-dependent stage. For the
   1,000-paper canary, require newly built V17-Turbo-owned Chroma and SQLite FTS5 artifacts beneath
   `artifacts/v17_turbo/index`; byte-identical V17 vectors may be copied only after registry, corpus
   version, chunk count, collection, and embedding-model compatibility are proven.
8. Validate `active_index.json` and `index_binding.json`, including registry hashes, the complete
   Chroma tree digest, lexical database digest, private Turbo paths, and no runtime dependency on an
   earlier policy directory. Bind both manifests into every subsequent plan.
9. For any expansion up to 20,000 papers, create a new content-addressed corpus version and exact
   index binding. Verify all document/chunk counts and hashes after append or rebuild; never mutate or
   relabel the prior binding. Re-freeze every retrieval-dependent plan against the expanded index.

## Stage 1A - author, review, and seal the held-out benchmark

This is an intrinsic V17-Turbo stage, not a separate benchmark utility. The canonical operator entry
point is `python3 -u giis_pipeline.py --config config/settings_v17_turbo.toml heldout-pipeline`.
That command owns one resumable state machine from Qwen authoring through deterministic qualification,
review-packet freezing, the guarded Qwen-to-OSS server handoff, the complete OSS single-source census,
Codex-gold queue materialization, gold dispositions, and final benchmark freeze. `author-heldout` is a
low-level checkpoint/resume command and does not by itself complete the V17-Turbo held-out stage.

1. Keep all 100 held-out documents outside SFT and training-time validation inputs. Author 100
   cases using the candidate-mix projection: 71 single-source, 25 multi-source prose, two tables,
   and two comparisons. Cover every held-out document at least once.
2. Use Qwen3.5-122B-A10B-GPTQ-Int4 to author each held-out question, candidate-type declaration,
   atomic obligation rubric, exact evidence mapping, semantic guardrails, and reference answer. The
   authoring request never includes a student answer.
3. Route every single-source case to GPT-OSS-120B component-mapped dual-axis review. Route every
   multi-source, table, and comparison case to Codex gold review. A routing mismatch fails closed.
4. Require `complete` coverage and supported claims before admission. Hold minor omissions for repair
   and full requalification; reject material, unsupported, malformed, or unresolved cases.
5. For multi-source and creative cases, require every source to belong to the held-out partition.
6. Freeze questions separately from hidden references. Bind cases, evidence, obligations, reference
   answers, reviewer identities and verdicts, source hashes, case order, and file hashes in the
   benchmark manifest.
7. Prohibit held-out results from checkpoint selection, hyperparameter tuning, prompt changes,
   retrieval changes, threshold changes, or policy amendments. Once results influence a change, the
   benchmark is spent and must be replaced for final evaluation.
8. Require all 100 planned cases and the exact 71/25/2/2 allocation before canonical materialization.
   A run-local operator exception may authorize a smaller pre-review artifact, but it is recorded in
   that artifact, cannot amend V17-Turbo, cannot silently alter future counts, and cannot satisfy the
   canonical full-count gate.
9. Persist stage markers for `authoring`, `deterministic_qualified`, `handoff_to_oss`,
   `oss_review_complete`, `awaiting_codex_gold_review`, `gold_review_complete`, and `frozen`.
   Every resume verifies the held-out selection, allocation, packet, candidate, and implementation
   digests before dispatching new work.

## Stage 1B - assemble obligation-complete retrieval context

1. Run the established standalone-query rewrite and the unchanged hybrid dense, lexical, and reranking
   source-discovery pass.
2. Deterministically preserve every explicit interrogative clause, then retrieve a small preliminary
   evidence set for those clauses.
3. From the question, permitted history, and preliminary corpus passages only, derive three to ten
   atomic evidence-backed obligations. Split broad methods, findings, mechanisms, factors, barriers,
   limitations, comparisons, and implications into distinct substantive requirements. Never expose
   benchmark references, hidden rubrics, reference answers, or prior gold-review findings to this step.
4. Retrieve and rerank passages separately for every atomic obligation. Reserve one qualifying context position
   per obligation before global rank, diversity, and deduplication fill unused positions.
5. Permit up to ten substantively distinct passages from one document during obligation-aware assembly,
   while enforcing the existing twelve-passage and 6,144-token aggregate context limits.
6. Map every obligation to exact selected evidence IDs using the semantic coverage validator. If any
   obligation is uncovered and capacity remains, run one expanded targeted rescue retrieval and validate
   the resulting bundle once more.
7. Pass the obligation map to generation. Require coverage of every supported item and an explicit evidence
   limitation for any unresolved item; prohibit completion from model memory.
8. Persist planned obligations, mapped evidence IDs, coverage status, and retrieval-pass count in every
   evaluation record. Also persist whether atomic planning succeeded, and aggregate atomic-planning,
   obligation-level, and case-level coverage in the run summary. A fallback to explicit question clauses
   is safe retrieval behavior, but it does not count as successful atomic planning or complete atomic coverage.
9. A rerun of a benchmark that motivated this policy is diagnostic only. Preserve its original artifacts,
   write the rerun to a new namespace, and require a fresh sealed held-out set for a new final claim.
10. Treat a generation `finish_reason` of `length` as a failed response gate. Run one repair using the
    long-form completion capacity even when the original request used the standard lane; if the repair is
    also truncated or fails another response gate, reject it rather than admitting a partial answer.

## Stage 2 - freeze the plan

Run the V17-Turbo planner only after every bound source exists. The plan hashes the policy, protocol,
configuration, manifests, corpus records, generic primitives, V17-Turbo modules, benchmark, model
identity, split, and output namespace. Planning must fail if the output directory already exists.

Frozen canary controls are:

| Control | Value |
| --- | ---: |
| Corpus / held out / generation | 1,000 / 100 / 900 |
| Train-generation / validation-generation | 810 / 90 |
| Initial candidates per paper | 4 |
| Preferred minimum / retained maximum | 3 / 4 |
| Audited underfilled floor | 2 |
| Refill candidates / maximum refills | 2 / 2 |
| Infrastructure retries per request | 2 |
| Concurrent teacher requests | 2 |
| Multi-source drafts | 1,260 from 630 bundles |
| Structured variants | 101 tables + 101 comparisons |
| Teacher endpoint / profile port | `127.0.0.1:8800` / `8800` |
| Teacher context / rendered input | 16,384 / 16,000 tokens |
| Teacher completion allowance | 8,000 tokens per request |
| Standard / long-form answer capacity | 900 / 2,200 tokens |
| Complete student-sample ceiling | 8,192 tokens |
| Timeout | 2,400 seconds |
| Temperature / seed / thinking | 0.30 / 3407 / disabled |

## Stage 3 - generate and verify core candidates

For each generation paper, build no more than eight substantive chunks and require at least four.
Render stable, complete-sentence support units. Request four candidates and require each to identify
material obligations before composing mapped answer units.

During prevalidation, materialize each answer once, count answer tokens with the frozen tokenizer, and
overwrite the advisory response label:

- 1-900 tokens becomes `standard`;
- 901-2,200 tokens becomes `long_form`; and
- more than 2,200 tokens is rejected.

Do not calculate or enforce answer word counts. Do not reject a complete answer for being below a
word target. Then apply citation, evidence range, secondary attribution, 40% citation coverage,
8,192-token assembled-sample, duplicate-question, obligation mapping, and answer-unit support checks.

Retain at most four candidates. If fewer than three pass, request at most two two-candidate refills.
After the bounded budget, release two passing candidates with an explicit underfilled event; fewer than
two fails closed. Preserve completed checkpoints and resume only against the exact plan digest.

## Stage 4 - generate multi-source candidates

Allocate 630 deterministic paired-paper bundles across partitions and request two candidates per bundle.
Require a material obligation from each source, substantive citations to both, genuine synthesis or
comparison, preserved methodological boundaries, and no unsupported causal conversion. Multi-source
results do not replace single-source minima.

## Stage 5 - qualify creative formats

Select one passing multi-source prose candidate from each of 101 distinct bundles. Hold its question,
evidence, obligations, and sources fixed while generating one table and one point-by-point comparison.
Verify each of the 202 variants independently. Do not regenerate a failing variant and do not invalidate
its passing prose source.

## Stage 6 - freeze semantic packets

Build review packets only after primary generation and the 202-case structured-qualification manifest
are complete. Before packet creation, require the final passing-candidate file count to equal the
current progress count and require every file to carry the active plan digest. Materialized passing
files outside the current accepted set must have been removed; a resumed wave cannot remain visible
to qualification or review.

Build review packets only from Python-passing candidates. Bind candidate ID, candidate hash, plan
digest, question, complete numbered evidence, support units, declared obligations, answer units,
materialized answer, partition, lane, request path, and source identities. Packet creation must fail on
an identity mismatch.

## Stage 6A - perform the automatic model handoff

1. Serve Qwen3.5-122B-A10B-GPTQ-Int4 only at `127.0.0.1:8800` during generation.
2. After generation and packet freezing, stop the Qwen server and verify that port 8800 is closed.
3. Wait for at least 100 GiB available system memory; abort if memory does not recover.
4. Require `127.0.0.1:8600` to be free, then start GPT-OSS-120B there.
5. Verify the OSS health endpoint before dispatching any review request.
6. Keep the Qwen and OSS servers mutually exclusive. Any collision or uncertain transition fails
   closed and preserves all generation and review checkpoints for an exact resume.

## Stage 7 - apply final semantic routing

Send every single-source packet to the frozen GPT-OSS-120B component-mapped dual-axis reviewer. The
coverage axis reconstructs atomic requirements and measures actual answer coverage; the support axis
attempts to falsify every material claim against the closed evidence. Retry malformed axis output only
within the frozen reviewer allowance. Do not send any single-source packet to gold review. A candidate
is machine-complete only when both axes finish successfully, coverage is complete, and support is
supported. Machine-complete candidates become eligible for later release consideration; every minor
omission, material incompleteness, unsupported result, malformed result, transport exhaustion, and
missing axis fails closed into quarantine.

Send every multi-source, table, and comparison packet to gold semantic review. Review the required
meaning independently of the teacher ledger. Apply `complete`, `minor_omission`,
`materially_incomplete`, or `unsupported`; route them respectively to eligible, hold, discard, and
discard.

Semantic review is the final quality stage. It never edits the candidate and never automatically asks
the teacher for a replacement.

## Stage 7A - bind the student tokenizer

Keep Qwen-122B as teacher and bind Qwen3.8-27B as the student `models.base`. Retokenize every admitted
chat record with the exact student snapshot, direct-answer chat template, and completion-mask function.
Publish a new immutable child release containing per-record total and assistant-completion token counts.
Require non-empty completion masks and reject any sample above `training.max_length=8192`. Preserve the
parent release and bind its manifest hash in the child.

## Stage 8 - validate training capacity before release

From the eligible pool, identify the longest tokenized complete student samples. Before release or
training, run one guarded QLoRA optimizer step at `training.max_length=8192` using the configured
microbatch, accumulation, checkpointing, attention, and allocator controls. Record peak unified memory,
step completion, loss, configuration hash, sample IDs, and command. An OOM or inadequate headroom
blocks training and requires a new fingerprinted capacity decision.

## Stage 9 - disposition and release boundary

Retain raw outputs, all deterministic decisions, semantic outputs, quarantines, holds, discards,
finish reasons, latency, token counts, checkpoints, source hashes, and plan identity. A pause marker
stops new dispatch; a lock prevents a second runner.

No V17-Turbo runner emits a training release. A separate operator-authorized finalizer must select
eligible rows and create immutable train and validation files plus a manifest containing all counts,
hashes, policy identity, exclusions, and audit references.

The finalizer publishes schema-v2 dataset_manifest.json, training_selection.json, and
training_materialization.json beside the immutable JSONL files. The selection count is the number of
distinct papers actually represented by admitted train or validation examples; planned papers with no
admitted candidate remain explicit zero-yield entries. Canonical dataset paths must converge on this
single release directory, and validate_dataset_release must pass before any model framework loads.
Dataset-path activation leaves training_release_enabled=false until training is separately authorized.

## Execution commands

Create a new plan:

```bash
cd "/home/codeaccounting/Documents/Mini run"
/home/codeaccounting/Documents/LLM_env/bin/python3 -m giis_prod.v17_turbo_canary plan   --out artifacts/v17_turbo/<run_name>
```

Run or resume that exact plan:

```bash
cd "/home/codeaccounting/Documents/Mini run"
/home/codeaccounting/Documents/LLM_env/bin/python3 -m giis_prod.v17_turbo_canary run   --out artifacts/v17_turbo/<run_name>
```

Do not launch either command while another model transition or bound run is active without the
operator's explicit scheduling decision.
