# Qwen 27B and 4B held-out generation results

The fine-tuned Qwen systems answered more questions than their base counterparts. The number of answers rated both fully supported and complete stayed the same within each model family. The fine-tuned 4B FP8 export answered 66 of 77 questions, compared with 69 for the fine-tuned 4B BF16 export.

## Evaluation setup

The comparison covered five deployed variants on the same 77 eligible questions, for 385 reviewed records. It used the full-corpus dense and keyword indexes and obligation-aware retrieval. One case was excluded before scoring because its source paper was deliberately absent from the corpus. The denominator of 77 applies to every variant.

The fine-tuned 27B system served a BF16 merge from the NF4 QLoRA run. The fine-tuned 4B system served a BF16 merge from the BF16 LoRA run, with an FP8 export tested separately. Both merges used checkpoint 230. The comparison included a base control for each family and an FP8 export for the 4B family. See the [fine-tuning results](qwen-sft-results.md) for checkpoint selection and learning curves.

Hidden references were kept out of generation. After predictions were sealed, Codex reviewers applied the same frozen rubric across the entire set. Model labels were withheld during initial review and joined afterward. The tables use the final adjudicated counts, including four documented corrections. The [gold review protocol](../guidebook/STUDENT_HELDOUT_GOLD_REVIEW.md) defines the evidence and scoring rules.

## Answer outcomes

| Variant | Answered | Fully supported and complete | Minor issues | Materially incomplete | Unsupported | Abstained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B base | 52/77 (67.5%) | 5 | 7 | 34 | 6 | 25 |
| Qwen3.8-27B fine-tuned BF16 | 71/77 (92.2%) | 5 | 6 | 45 | 15 | 6 |
| Qwen3.5-4B base BF16 | 45/77 (58.4%) | 1 | 9 | 18 | 17 | 32 |
| Qwen3.5-4B fine-tuned BF16 | 69/77 (89.6%) | 1 | 9 | 33 | 26 | 8 |
| Qwen3.5-4B fine-tuned FP8 | 66/77 (85.7%) | 1 | 9 | 34 | 22 | 11 |

The four answer categories are mutually exclusive and sum to the answered count. The unsupported category takes precedence over incompleteness. Abstentions stay in the full denominator and receive no claim-level correctness or evidence support rating.

## Review dimensions

Counts are per variant out of 77. The rubric assesses scientific correctness, cited evidence support, and completeness separately. An answer can be scientifically accurate yet omit a required point or cite a passage that fails to support a claim.

| Variant | Scientifically accurate | Minor imprecision | Material error | No claims |
| --- | ---: | ---: | ---: | ---: |
| 27B base | 36 | 10 | 6 | 25 |
| 27B fine-tuned BF16 | 52 | 6 | 13 | 6 |
| 4B base BF16 | 16 | 13 | 16 | 32 |
| 4B fine-tuned BF16 | 40 | 8 | 21 | 8 |
| 4B fine-tuned FP8 | 36 | 10 | 20 | 11 |

| Variant | Cited evidence supports claims | Minor support gap | Materially unsupported | Contradicted | No claims | Not itemized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 27B base | 34 | 12 | 5 | 1 | 25 | 0 |
| 27B fine-tuned BF16 | 49 | 7 | 15 | 0 | 6 | 0 |
| 4B base BF16 | 15 | 13 | 15 | 2 | 32 | 0 |
| 4B fine-tuned BF16 | 33 | 9 | 24 | 2 | 8 | 1 |
| 4B fine-tuned FP8 | 32 | 9 | 20 | 2 | 11 | 3 |

The source summary leaves one 4B BF16 and three 4B FP8 evidence ratings unitemized. Those records stay in the denominator without an inferred rating.

| Variant | Complete | Minor omission | Materially incomplete | Absent | Not itemized |
| --- | ---: | ---: | ---: | ---: | ---: |
| 27B base | 6 | 8 | 38 | 25 | 0 |
| 27B fine-tuned BF16 | 7 | 5 | 59 | 6 | 0 |
| 4B base BF16 | 5 | 5 | 31 | 32 | 4 |
| 4B fine-tuned BF16 | 2 | 10 | 57 | 8 | 0 |
| 4B fine-tuned FP8 | 2 | 9 | 55 | 11 | 0 |

The source summary leaves four base 4B completeness ratings unitemized. Those records stay in the denominator without an inferred rating.

## Comparison within each model family

- **27B fine-tuned vs. base.** The fine-tuned run produced 19 more substantive answers. It also produced 16 more scientifically accurate answers and 15 more answers supported by their cited evidence. Material errors rose from 6 to 13. Among the 47 questions both systems answered, 27 answers were materially incomplete on both sides, and two were fully supported and complete on both sides.
- **4B fine-tuned BF16 vs. base BF16.** The fine-tuned run produced 24 more substantive answers, 24 more scientifically accurate answers, and 18 more supported answers. Material errors rose from 16 to 21. Among the 41 common answered questions, both were materially incomplete on 10 and both fell into the unsupported category on five.
- **4B FP8 vs. fine-tuned BF16.** The FP8 run produced three fewer substantive answers, four fewer scientifically accurate answers, and one fewer supported answer. Both produced one fully supported and complete answer overall. Among 61 common answered questions, both were materially incomplete on 23 and both fell into the unsupported category on 14. The base 4B and FP8 runs answered 41 questions in common.

## Interpretation

These are comparisons of complete retrieval and generation runs. The passages retrieved for a question could differ by variant, so the results do not isolate the causal effects of fine-tuning or FP8 quantization. The common-answered comparisons apply only to questions both systems answered. Some prompts referred to studies imprecisely, and reviewers used source evidence to resolve conflicts with the frozen reference. The judgments are from structured Codex review and have no independent human validation.
