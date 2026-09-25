# GIIS Decoder UI Policy

## Status and scope

This policy governs the user-facing decoder interface for GIIS Climate and Ecology. It is independently
executable and does not depend on an earlier numbered UI policy. V17-Turbo supplies the trained model,
verified corpus, active hybrid index, response capacities, and grounding contract, but this document
governs how those components are exposed to a user.

The UI is a research interface, not an unrestricted chatbot. Every released factual answer must be
derived from the active corpus evidence, must report its evidence coverage honestly, and must pass the
grounding gates below. Interface convenience may not weaken retrieval, citation, completeness, or
abstention behavior.

## Bound runtime artifacts

Startup must bind and record:

- the merged student-model path and package-manifest digest;
- the exact tokenizer and direct-answer chat template;
- the active dense and lexical index manifest and binding digests;
- the document and chunk registry digests;
- the retrieval, response, serving, and UI configuration fingerprint; and
- the UI policy and implementation digests.

The model server remains a local authenticated service. The UI may run from another directory only by
using canonical absolute paths or a validated service-binding manifest. Missing, stale, partial, or
foreign bindings fail closed before the interface accepts questions.

## Interaction architecture

The canonical request path is:

1. accept and display the user's question;
2. rewrite it as a standalone corpus query using only permitted recent dialogue;
3. preserve every explicit interrogative clause;
4. perform obligation-aware hybrid retrieval and semantic evidence-coverage validation;
5. decode an answer from the selected evidence and obligation map;
6. validate answer grounding, obligation completion, citations, and finish reason;
7. attempt one bounded repair when a repairable gate fails; and
8. release the validated answer, visible coverage status, references, and source tools.

The UI never exposes hidden reasoning, reviewer prompts, benchmark references, training answers, or
internal chain-of-thought. It may expose evidence, obligation labels, coverage decisions, citations,
validation outcomes, and concise operational status.

## Query and conversation policy

The current question and at most six recent conversation messages may be used to resolve pronouns and
omitted subjects. Rewriting must not answer the question, introduce a new requirement, or silently
drop an explicit clause. Failure returns the original question.

Empty questions do not dispatch retrieval. Clearing the conversation clears the visible chat, pending
query, selected source, coverage display, citations, references, and source-summary state.

## Response modes and token capacity

Response mode is selected from the question's requested work, not from a word target:

| Mode | Completion capacity | Use |
| --- | ---: | --- |
| `standard` | 900 tokens | Narrow, closed, or bounded questions |
| `long_form` | 2,200 tokens | Multi-part synthesis, explanation, comparison, review, and essay requests |
| `source_summary` | 550 tokens | Interactive brief summary of one selected corpus source |

The decoder has no minimum, target, or maximum word-count gate for standard or long-form answers.
Capacity is an allowance rather than a requested length. The model stops when all supported obligations
are resolved. It may use less than the allowance, and it may not pad an answer to fill the lane.

Automatic mode selection remains deterministic. A narrow closed question stays standard. A substantive
multi-part, comparative, review, synthesis, or sufficiently broad how/why question becomes long form.
Long-form generation requires at least two distinct corpus sources unless the user explicitly requests
analysis of one identified source. A future manual mode control may request a mode but may not enlarge
the configured capacity or weaken evidence sufficiency.

A `finish_reason` of `length` fails the response gate. One repair may run with the long-form capacity.
A second truncation or other unresolved gate fails closed.

The brief source summary is a separate interaction, not an SFT response lane. Its approximate
250-word presentation goal is advisory; the hard control is the 550-token completion capacity and
evidence completeness. A longer hierarchical summarizer requires a separately validated design.

## Obligation-first retrieval

The UI requires obligation-aware retrieval to be enabled. The canonical runtime values are:

```toml
enable_obligation_aware_retrieval = true
obligation_candidate_multiplier = 2
obligation_max_chunks_per_document = 10
final_chunks = 12
maximum_context_tokens = 6144
```

The system deterministically preserves explicit question clauses, uses preliminary corpus passages to
derive three to ten atomic evidence obligations, and retrieves separately for each obligation. One
qualifying context position is reserved for each obligation before general relevance and source
diversity fill unused capacity. A single doubled-depth rescue pass may target uncovered obligations.

The semantic coverage validator maps every obligation to exact selected evidence IDs. A partial hint
does not count as coverage. Planner failure falls back to explicit clauses but must be shown as a
planning limitation rather than successful atomic coverage.

## Two-axis coverage status

The UI distinguishes retrieval coverage from answer completion.

### Evidence coverage

For every obligation, record one of:

- `covered`: selected passages explicitly support an answer;
- `uncovered`: the active corpus bundle does not contain sufficient selected evidence;
- `planner_fallback`: deterministic explicit clauses were used because atomic planning failed; or
- `validation_unavailable`: the semantic coverage validator failed or returned malformed output.

### Answer completion

For every obligation, record one of:

- `answered`: a supported answer unit explicitly resolves a covered obligation;
- `limitation_reported`: the response explicitly discloses that an uncovered obligation cannot be
  answered from the supplied corpus evidence;
- `omitted`: the response neither answers the obligation nor reports its evidence limitation; or
- `unsupported_answer`: the response attempts to answer without adequate mapped evidence.

A response is obligation-complete only when every obligation is either `answered` from mapped evidence
or `limitation_reported`. `Omitted` and `unsupported_answer` fail the release gate.

## Visible coverage panel

Every non-trivial answer displays a compact **Evidence coverage** panel. The panel shows:

- aggregate status: `Complete`, `Partial evidence`, `Planner fallback`, or `Validation unavailable`;
- every concise obligation label in question order;
- evidence status and answer-completion status;
- mapped reader-facing citations and supporting pages when covered; and
- the retrieval-pass count, including whether rescue retrieval ran.

The panel must not show hidden reasoning or internal prompts. It remains synchronized with the exact
evidence bundle and answer returned in that interaction. If status is partial, the answer must contain
the corresponding evidence limitation. The panel may collapse by default on narrow questions but must
remain accessible.

## Grounded decoding

The decoder receives only the question, permitted dialogue context, selected evidence, and obligation
map. It must:

- use no remembered or outside factual knowledge;
- address every covered obligation;
- state a limitation for every uncovered obligation;
- preserve source ownership, scope, population, method, units, uncertainty, and disagreement;
- avoid causal conversion not supported by the evidence;
- cite only documents supplied under their own corpus document IDs; and
- attach exact evidence markers to factual prose and substantive table rows.

Thinking mode is disabled. Temperature is zero. Hidden reasoning is neither requested nor rendered.

## Response release gate

Before the main answer becomes visible, validation requires:

1. at least one qualifying evidence item unless the system is abstaining;
2. valid evidence-marker syntax and range;
3. at least 80% citation coverage over factual prose and substantive table rows;
4. no direct citation to a work merely mentioned by a corpus source;
5. no non-corpus secondary citation presented as evidence;
6. complete answer handling for every planned obligation;
7. an explicit evidence limitation for every uncovered obligation;
8. no unsupported answer unit;
9. sufficient source participation for long-form and comparative work; and
10. a non-length completion finish reason.

One repair may preserve supported content, remove unsupported claims, add missing supported obligation
units, add required limitations, and correct evidence markers. Repair may not retrieve outside the
frozen evidence bundle unless the retrieval policy explicitly invokes its one rescue pass before
decoding. Failure after repair returns the configured validation-failure message and retains an
auditable rejection record.

The main answer is decoded and validated before reader-facing progressive rendering. Simulated word
group rendering must never be represented as raw model streaming. No invalid draft is shown first.

## Citations and references

Internal `[E#]` markers remain in the auditable draft. The reader-facing answer replaces them with
author-year citations, combining sources when appropriate and deduplicating repeated document IDs.

The reference panel shows the corpus title, authors, year, publication, supporting page ranges,
retrieval similarity, and a validated DOI link when present. Citation metadata popovers include only
sources actually cited in the answer. The broader reference panel and source selector may include all
retrieved sources, clearly distinguished from cited sources.

The interface must never invent bibliographic metadata or DOI links. Missing fields are displayed as
unavailable.

## Source summaries

The source selector stores an immutable corpus document ID separately from its displayed title. A
stale label or browser state may not change the selected identity. The summary samples up to eight
chunks across the document and uses only that source's evidence.

The canonical brief sections are Purpose, Approach, Findings, and Limitations or Implications, included
only when supported. Because the source is already identified in the reference panel, the summary does
not emit evidence markers or a second reference list. Only one source-summary request runs at a time.
If streaming fails before usable output, the UI reports failure. A terminal validation failure replaces
the streamed draft rather than presenting it as an approved summary.

## Abstention and error behavior

No dense candidate above the calibrated threshold, inadequate evidence for the requested mode, failed
coverage validation, failed answer validation, timeout exhaustion, and server errors remain distinct
machine-readable outcomes. User-facing messages are concise and do not disclose secrets, stack traces,
internal prompts, or model-server arguments.

An evidence shortage yields `No corpus evidence.` or a precise partial-evidence limitation. A grounded
answer that fails validation is not displayed as a successful response. Retrieval and answer errors
must not silently fall back to an ungrounded base-model answer.

## Security and service exposure

The model endpoint binds to localhost and requires a non-default API key. Production UI startup must
fail closed unless authenticated access is configured directly or supplied by a validated reverse
proxy. Public Gradio sharing is disabled in production. Development sharing, if explicitly authorized,
uses credentials and is recorded as a temporary non-production session.

Secrets are supplied through protected environment variables and are never written to configuration,
logs, HTML, links, or error messages. DOI links open with `noopener noreferrer`. User input and corpus
metadata are escaped before insertion into raw HTML.

## Accessibility and responsive layout

Desktop uses a resizable chat/reference split with a keyboard-operable separator and explicit ARIA
attributes. Mobile uses a single-column layout. Focus order, labels, status changes, contrast, modal
close controls, and keyboard submission must remain usable without a pointer. Coverage and failure
states cannot rely on color alone.

## Observability and quality gates

Each interaction records a non-secret request ID, model and index identities, response mode, retrieval
passes, obligation counts, evidence and answer coverage, citations, token use, finish reason, repair
count, abstention or rejection codes, time to first visible validated text, and total latency.

Before production activation, the complete UI path must demonstrate:

- seven concurrent users through the configured queue;
- P95 time to first token or first validated display within five seconds where the workload permits;
- at least 15 decoded tokens per second for a single user;
- no more than 90 seconds for the defined 700-word latency probe;
- no material quality regression greater than 0.5%; and
- passing citation, obligation, abstention, source-summary, responsive-layout, authentication, and
  failure-injection tests.

Performance targets never authorize bypassing validation. If validate-before-display makes the first
visible answer slower, reporting distinguishes model TTFT from validated-display latency.

## Versioning and change control

Any change to model identity, tokenizer, active index, obligation planner, coverage validator, response
capacities, repair allowance, citation threshold, authentication, or displayed coverage semantics
creates a new UI configuration fingerprint and requires focused regression tests. A UI release cannot
reuse an earlier approval after a material decoder or retrieval change.

## Initial conformance work

The present implementation must complete these changes before this policy can be declared active:

1. remove legacy `target_long_form_words`, `maximum_long_form_words`, and
   `minimum_long_form_words` dependencies from decoding and validation;
2. make standard and long-form behavior depend on token capacity and obligation completion;
3. implement post-decoding per-obligation answer-completion validation;
4. render the synchronized visible coverage panel;
5. make production authentication fail closed and disable public sharing by default; and
6. test the assembled production configuration rather than only mocked component settings.

Until those controls pass, the current Gradio interface is a development implementation and not a
canonical production UI release.
