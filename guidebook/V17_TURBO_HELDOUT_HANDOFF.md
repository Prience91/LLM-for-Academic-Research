# V17-Turbo held-out handoff contract

The canonical `heldout-pipeline` requires completion of the main training/validation all-lane OSS
census and a hash-bound queue containing only completed OSS semantic rejects. It verifies source plan,
census and queue hashes, routing revision, and terminal OSS failures before authoring. Main gold
dispositions are deferred until after held-out authoring and the joint operator checkpoint.

Qwen-122B authors the exact 100-case allocation: 71 single-source, 25 multi-source, two tables, and
two comparisons. Deterministic qualification freezes the original materials and candidate identities.
Every generated candidate in every lane is queued for Codex gold. No held-out OSS server transition
or machine admission occurs. Qwen is released after queue creation and the automatic terminal state
is `awaiting_codex_gold_review_authorization`. The single-launch wrapper binds this queue
with the main reject queue and pauses once for authorization of both reviews.

Explicit operator authorization is bound to the exact queue manifest and packet hash before gold
review. Gold receives the original question, obligations, evidence, sources, and answer. Any historical
machine verdict remains hidden. Only deterministically qualified gold-complete cases are eligible for
benchmark freeze. Missing decisions, foreign hashes, source contamination, incomplete allocation, or
unresolved review errors fail closed. Minor omissions require authorized repair and deterministic plus
gold requalification.

Exhausted Qwen held-out slots are terminal quarantine entries, never retried or repaired. All
successfully generated cases still enter the gold queue; its manifest records the exact lane deficit.
This underfilled review is valid as a diagnostic census but cannot satisfy the canonical 100-case
final benchmark gate.
Use an explicit fresh namespace. Old output namespaces and frozen fingerprints remain historical and
cannot resume under `split_review_v2`. Every implementation/configuration binding is checked before
queue preparation; changes require a new fingerprint.

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
