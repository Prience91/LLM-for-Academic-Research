# V17-Turbo universal OSS-first semantic-routing amendment

Status: canonical forward amendment, effective 2026-09-29 for every V17-Turbo plan frozen after this amendment is consolidated into the operative policy fingerprint.

This amendment changes semantic-review routing only. Generation, deterministic qualification, bounded retries, evidence and obligation construction, capacity lanes, repair rules, release authorization, held-out isolation, and all other V17-Turbo controls remain unchanged.

## Universal first-pass review

Every Python-passing candidate in every lane and split must first receive the frozen GPT-OSS-120B component-mapped dual-axis semantic review. This includes:

- single-source prose;
- multi-source prose;
- table candidates;
- comparison candidates;
- training-generation candidates;
- validation-generation candidates; and
- held-out benchmark references.

The OSS review packet binds the original question, obligations, evidence, source identities, answer, candidate hash, plan digest, lane, and split. Coverage and support axes must both finish successfully. The complete OSS decision, per-axis outputs, errors, candidate hash, packet digest, reviewer identity, and plan digest are committed immutably before downstream disposition.

An OSS candidate is admitted directly to the eligible pool only when both axes complete successfully and the decision is complete and supported. OSS does not edit, repair, or regenerate candidates.

## Gold review of OSS non-admissions

Every OSS non-admission is quarantined into a content-addressed gold-review queue. This population includes:

- minor omission;
- material incompleteness;
- unsupported content;
- missing or malformed axis output;
- exhausted reviewer retries;
- transport or server failure; and
- any other fail-closed OSS outcome.

Gold review receives the original question, obligations, evidence, sources, and answer. The OSS verdict and rationale remain hidden until the independent gold decision is committed, preventing anchoring. Candidate identity and hashes remain visible so that decisions can be joined afterward without ambiguity.

Gold review remains an explicitly operator-authorized stage. The automatic Qwen-to-OSS handoff may complete the universal OSS census and freeze the gold queue, but it may not initiate gold review without authorization.

Gold dispositions are final for the reviewed candidate:

- gold `complete`: salvage into the eligible pool;
- gold `minor_omission`: hold for explicitly authorized surgical repair followed by full deterministic, OSS, and gold requalification;
- gold `materially_incomplete`: discard;
- gold `unsupported`: discard; and
- unresolved gold error: remain quarantined.

A gold decision never overwrites the OSS record. Both decisions and the final disposition remain in release provenance.

## Main and held-out workflow

For the main dataset, OSS reviews all single-source, multi-source, table, and comparison candidates before any gold work. After the universal census, only the OSS non-admission queue is eligible for gold review.

The held-out workflow mirrors the same order. Qwen authors and deterministic qualification freezes the exact benchmark population; OSS reviews every held-out lane; the workflow freezes the rejected/non-admitted gold queue; and gold review starts only after explicit operator authorization. Only cases ultimately marked complete may enter the frozen benchmark.

## Release and audit requirements

The semantic-review manifest must report, by lane and split:

- total OSS-reviewed candidates;
- OSS direct admissions;
- OSS semantic rejects;
- OSS fail-closed errors;
- gold-queue size and digest;
- gold-reviewed count;
- gold salvages, holds, discards, and unresolved errors; and
- final eligible, quarantined, held, and discarded counts.

Every release row must identify its admission path as either `oss_direct_complete` or `gold_salvaged_after_oss_non_admission`. Release finalization fails closed if any candidate lacks an OSS record, if a gold-reviewed row was absent from the frozen OSS non-admission queue, or if a queued candidate is silently omitted from the disposition census.

## Transition and frozen-run protection

The currently active held-out run is bound to the earlier policy and protocol hashes. Its files must not be edited in place while review is active. It completes under its frozen routing and remains an auditable historical run.

Before the next V17-Turbo plan is frozen, this amendment must be consolidated into `V17_TURBO_POLICY.md`, `V17_TURBO_PROTOCOL.md`, `V17_TURBO_HELDOUT_HANDOFF.md`, configuration, reviewer contracts, executables, tests, and content-addressed bindings. That consolidation creates a new policy fingerprint while retaining the name V17-Turbo. No new run may claim compliance with this amendment until those policy and executable surfaces agree.
