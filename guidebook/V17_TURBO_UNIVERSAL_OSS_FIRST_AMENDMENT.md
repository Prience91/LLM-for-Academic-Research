# V17-Turbo split-specific semantic-routing amendment

Status: canonical forward amendment, revised 2026-09-29. Consolidated into canonical policy, configuration, and execution paths on 2026-09-29; new plans bind the revised operative fingerprint. The existing filename is retained for reference compatibility; the universal OSS-first routing it formerly described is superseded by this split-specific contract.

## Routing by split

| Split | First semantic reviewer, every lane | Subsequent gold review |
| --- | --- | --- |
| Training-generation | GPT-OSS-120B | Every completed OSS semantic reject, after explicit operator authorization |
| Validation-generation | GPT-OSS-120B | Every completed OSS semantic reject, after explicit operator authorization |
| Held-out references | Codex gold | Every generated candidate in every lane requires gold review before benchmark eligibility |

Lanes include single-source prose, multi-source prose, Markdown tables, and comparisons. Deterministic qualification precedes semantic review. A generated candidate that fails deterministic qualification remains quarantined and ineligible regardless of its gold assessment; no generated held-out candidate is silently dropped from the authoring or review census. Exhausted authoring failures with no assessable candidate are recorded separately and cannot be counted as reviewed references.

This amendment consolidates split-specific semantic-review routing and the fail-closed retry and terminal-disposition contract. Candidate allocation, evidence and obligation construction, capacity lanes, release authorization, held-out isolation, and other V17-Turbo controls retain their established requirements.

## Training and validation: OSS first

Every Python-passing training or validation candidate in every lane receives the frozen GPT-OSS-120B component-mapped dual-axis semantic review. The packet binds the original question, obligations, evidence, source identities, answer, candidate hash, plan digest, lane, and split. Coverage and support axes must both finish successfully. The complete OSS decision, per-axis outputs, errors, hashes, reviewer identity, and plan digest are committed immutably before downstream disposition.

A candidate becomes eligible through `oss_direct_complete` only when both axes complete successfully and the decision is complete and supported. OSS does not edit, repair, or regenerate candidates.

Only completed OSS semantic rejects enter the content-addressed gold-review queue: minor omission, material incompleteness, and unsupported content after both axes succeed. Missing/malformed output, transport/server failure, or truncated review is a fail-closed execution outcome, retried once by OSS. Exhausted execution failures are terminal quarantine entries and never enter gold salvage. Gold receives the original materials with OSS verdict, rationale, and axis outputs hidden until the independent gold decision is committed. Candidate hashes remain visible for reliable joins afterward.

The automatic Qwen-to-OSS handoff completes the training/validation OSS census and freezes the gold queue. Gold review requires explicit operator authorization. Gold-complete semantic rejects may enter the eligible pool through `gold_salvaged_after_oss_non_admission`. A gold decision never overwrites the OSS record; both decisions remain in release provenance.

## Held-out references: gold review of the full population

Qwen-122B authors held-out questions, obligations, evidence mappings, and reference answers under the frozen allocation and source-isolation rules. After deterministic qualification, the workflow freezes a gold queue covering every generated candidate in every held-out lane, including all single-source references. OSS acceptance or rejection does not determine held-out gold selection or benchmark admission. The canonical held-out pathway has no required OSS review or automatic Qwen-to-OSS transition.

The automatic terminal state is `awaiting_codex_gold_review_authorization`. The workflow may prepare and hash the complete gold queue but may not initiate gold assessment without explicit operator authorization. Gold independently assesses the original question, obligations, evidence, sources, and answer; any historical machine verdicts remain hidden during the decision.

Only a deterministically qualified, gold-complete, fully supported reference may enter the benchmark through `gold_direct_complete`. Every generated held-out candidate receives a recorded gold disposition; unreviewed candidates block benchmark finalization. Exhausted authoring slots remain terminally quarantined while successfully generated cases proceed
to gold review; the shortfall does not satisfy the unchanged full-allocation final benchmark gate.

The held-out workflow begins after the main training/validation generation and complete hash-bound OSS census. Main gold review waits while Qwen authors and qualifies all held-out lanes; both blinded gold queues then share one authorization checkpoint. It uses a fresh output namespace. References remain hidden from students and held-out papers remain isolated from SFT, training-time validation, and tuning.

## Gold dispositions and repair

- `complete`: eligible after all applicable deterministic gates pass;
- `minor_omission`: held for explicitly authorized surgical repair and full requalification;
- `materially_incomplete`: discarded;
- `unsupported`: discarded; and
- gold fail-closed error: no retry; quarantine or archive only after explicit operator authorization, with no repair.

A repaired training/validation candidate repeats deterministic, OSS, and required gold review. A repaired held-out candidate repeats deterministic and gold review. Repair creates a new candidate hash and retains its parent identity. Semantic review remains the final quality judgment.

## Release and audit requirements

Training/validation manifests report OSS totals, direct admissions, semantic rejects, fail-closed errors, gold-queue digest, gold-reviewed count, salvages, holds, discards, unresolved errors, and final disposition counts by lane and split. Every eligible row uses `oss_direct_complete` or `gold_salvaged_after_oss_non_admission`. Finalization fails closed on missing OSS records, gold selection outside the frozen semantic-rejection queue, or omitted dispositions.

Held-out manifests report every authored candidate, deterministic outcomes, gold-queue size and digest, gold-reviewed count, eligible cases, holds, discards, unresolved errors, and authoring failures by lane. Every eligible reference uses `gold_direct_complete`. Finalization fails closed if any eligible reference lacks a matching gold-complete record, any generated candidate is omitted from the census, or the full frozen allocation remains unsatisfied without an explicit run-local exception. Held-out references do not require an OSS record.

## Consolidation and frozen-run protection

This routing contract is consolidated into `V17_TURBO_POLICY.md`, `V17_TURBO_PROTOCOL.md`, `V17_TURBO_HELDOUT_HANDOFF.md`, configuration, reviewer contracts, executables, tests, and content-addressed bindings. This produces a new policy fingerprint under the name V17-Turbo. A documentation amendment alone does not establish executable compliance.

Frozen historical runs retain their original fingerprints and explicit run-local exceptions. The completed 2026-09-29 held-out gold review is recorded in its own review manifests; it is not retroactively relabeled as having executed a future consolidated fingerprint.

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
