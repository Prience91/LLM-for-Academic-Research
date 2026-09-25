# Six-reviewer benchmark and hybrid semantic-review decision

Status: frozen benchmark evidence and operational decision record. This note does not alter the active V17 plan, candidate dispositions, or training-release boundary.

## Purpose and scope

Six locally served open-weight reviewers were tested on the same 349 blinded, Python-passing Qwen 27B V17 candidates. Each reviewer used the component-mapped dual-axis protocol: one axis independently assessed coverage and material omissions, and the other adversarially assessed factual support. Reviewer outputs were compared with the completed independent gold census.

The gold population contained 265 complete, 13 minor-omission, 17 materially incomplete, and 54 unsupported candidates. It comprised 263 single-source, 72 multi-source, and 14 table/comparison candidates. Model output errors failed closed and were counted separately.

## Reviewer profiles

| Reviewer | Architecture | Served weight format | Approximate activated parameters per token |
| --- | --- | --- | ---: |
| Qwen-27B | Dense | Native BF16 | about 27B |
| GPT-OSS-120B | Mixture of experts | MXFP4 | about 5.1B |
| Llama-3.3-70B | Dense | FP8 | about 70B |
| Qwen2.5-72B | Dense | FP8 | about 72B |
| Qwen3.5-122B-A10B-GPTQ-Int4 | Mixture of experts | GPTQ Int4 | about 10B |
| Nemotron-3-Super-120B-A12B-NVFP4 | Mixture of experts | NVFP4 | about 12B |

The activated-parameter figures describe the nominal model architecture and are not memory-footprint measurements. KV-cache quantization and serving parameters are separate from checkpoint-weight format.

## Classification convention and formulas

For the admission decision, **positive** means the reviewer rejects or quarantines a candidate as non-complete; **negative** means it admits the candidate as complete.

- True positive (TP): gold non-complete and reviewer non-complete.
- False positive (FP): gold complete but reviewer non-complete; recoverable unnecessary rejection.
- False negative (FN): gold non-complete but reviewer complete; admitted contamination.
- True negative (TN): gold complete and reviewer complete.
- Unnecessary gating: `FP / (TP + FP)`; the false share of valid reviewer rejections.
- Insufficient gating: `FN / (FN + TN)`; the contaminated share of valid reviewer admissions.
- Accuracy: `(TP + TN) / valid outputs`.
- Rejection precision: `TP / (TP + FP)`.
- Rejection recall: `TP / (TP + FN)`.
- Data casualty rate: `FP / valid outputs`.
- Data contamination rate: `FN / valid outputs`.
- Error rate: `errors / all candidates`; errors fail closed and are excluded from TP/FP/FN/TN rate denominators.

## Confusion counts

### Single-source lane

| Reviewer | TP | FP | FN | TN | Valid | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-27B | 20 | 9 | 34 | 196 | 259 | 4 |
| GPT-OSS-120B | 34 | 39 | 19 | 165 | 257 | 6 |
| Llama-3.3-70B | 5 | 7 | 48 | 185 | 245 | 18 |
| Qwen2.5-72B | 7 | 9 | 43 | 191 | 250 | 13 |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 23 | 42 | 25 | 161 | 251 | 12 |
| Nemotron-3-Super-120B-A12B-NVFP4 | 6 | 3 | 47 | 204 | 260 | 3 |

### Multi-source lane

| Reviewer | TP | FP | FN | TN | Valid | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-27B | 8 | 5 | 12 | 43 | 68 | 4 |
| GPT-OSS-120B | 20 | 24 | 2 | 20 | 66 | 6 |
| Llama-3.3-70B | 10 | 5 | 12 | 42 | 69 | 3 |
| Qwen2.5-72B | 2 | 0 | 16 | 42 | 60 | 12 |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 5 | 7 | 5 | 28 | 45 | 27 |
| Nemotron-3-Super-120B-A12B-NVFP4 | 0 | 2 | 21 | 42 | 65 | 7 |

### Creative lane

| Reviewer | TP | FP | FN | TN | Valid | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-27B | 2 | 0 | 4 | 8 | 14 | 0 |
| GPT-OSS-120B | 5 | 2 | 1 | 4 | 12 | 2 |
| Llama-3.3-70B | 3 | 2 | 3 | 6 | 14 | 0 |
| Qwen2.5-72B | 0 | 0 | 4 | 7 | 11 | 3 |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 2 | 1 | 2 | 5 | 10 | 4 |
| Nemotron-3-Super-120B-A12B-NVFP4 | 1 | 0 | 5 | 8 | 14 | 0 |

### Combined lane

| Reviewer | TP | FP | FN | TN | Valid | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-27B | 30 | 14 | 50 | 247 | 341 | 8 |
| GPT-OSS-120B | 59 | 65 | 22 | 189 | 335 | 14 |
| Llama-3.3-70B | 18 | 14 | 63 | 233 | 328 | 21 |
| Qwen2.5-72B | 9 | 9 | 63 | 240 | 321 | 28 |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 30 | 50 | 32 | 194 | 306 | 43 |
| Nemotron-3-Super-120B-A12B-NVFP4 | 7 | 5 | 73 | 254 | 339 | 10 |

## Selected risk metrics

### Unnecessary gating

| Reviewer | Single-source | Multi-source | Creative | Combined |
| --- | ---: | ---: | ---: | ---: |
| Qwen-27B | 31.0% | 38.5% | 0.0% | 31.8% |
| GPT-OSS-120B | 53.4% | 54.5% | 28.6% | 52.4% |
| Llama-3.3-70B | 58.3% | 33.3% | 40.0% | 43.8% |
| Qwen2.5-72B | 56.2% | 0.0% | N/A | 50.0% |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 64.6% | 58.3% | 33.3% | 62.5% |
| Nemotron-3-Super-120B-A12B-NVFP4 | 33.3% | 100.0% | 0.0% | 41.7% |

### Insufficient gating

| Reviewer | Single-source | Multi-source | Creative | Combined |
| --- | ---: | ---: | ---: | ---: |
| Qwen-27B | 14.8% | 21.8% | 33.3% | 16.8% |
| GPT-OSS-120B | 10.3% | 9.1% | 20.0% | 10.4% |
| Llama-3.3-70B | 20.6% | 22.2% | 33.3% | 21.3% |
| Qwen2.5-72B | 18.4% | 27.6% | 36.4% | 20.8% |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 13.4% | 15.2% | 28.6% | 14.2% |
| Nemotron-3-Super-120B-A12B-NVFP4 | 18.7% | 33.3% | 38.5% | 22.3% |

### Data casualty rate

| Reviewer | Single-source | Multi-source | Creative | Combined |
| --- | ---: | ---: | ---: | ---: |
| Qwen-27B | 3.5% | 7.4% | 0.0% | 4.1% |
| GPT-OSS-120B | 15.2% | 36.4% | 16.7% | 19.4% |
| Llama-3.3-70B | 2.9% | 7.2% | 14.3% | 4.3% |
| Qwen2.5-72B | 3.6% | 0.0% | 0.0% | 2.8% |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 16.7% | 15.6% | 10.0% | 16.3% |
| Nemotron-3-Super-120B-A12B-NVFP4 | 1.2% | 3.1% | 0.0% | 1.5% |

### Data contamination rate

| Reviewer | Single-source | Multi-source | Creative | Combined |
| --- | ---: | ---: | ---: | ---: |
| Qwen-27B | 13.1% | 17.6% | 28.6% | 14.7% |
| GPT-OSS-120B | 7.4% | 3.0% | 8.3% | 6.6% |
| Llama-3.3-70B | 19.6% | 17.4% | 21.4% | 19.2% |
| Qwen2.5-72B | 17.2% | 26.7% | 36.4% | 19.6% |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 10.0% | 11.1% | 20.0% | 10.5% |
| Nemotron-3-Super-120B-A12B-NVFP4 | 18.1% | 32.3% | 35.7% | 21.5% |

### Errors

| Reviewer | Single-source | Multi-source | Creative | Combined |
| --- | ---: | ---: | ---: | ---: |
| Qwen-27B | 4 | 4 | 0 | 8 |
| GPT-OSS-120B | 6 | 6 | 2 | 14 |
| Llama-3.3-70B | 18 | 3 | 0 | 21 |
| Qwen2.5-72B | 13 | 12 | 3 | 28 |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 12 | 27 | 4 | 43 |
| Nemotron-3-Super-120B-A12B-NVFP4 | 3 | 7 | 0 | 10 |

## Overall conventional metrics

| Reviewer | Accuracy | Rejection precision | Rejection recall | Admission-safe rate including fail-closed errors |
| --- | ---: | ---: | ---: | ---: |
| Qwen-27B | 81.2% | 68.2% | 37.5% | 86.0% |
| GPT-OSS-120B | 74.0% | 47.6% | 72.8% | 94.6% |
| Llama-3.3-70B | 76.5% | 56.2% | 22.2% | 84.0% |
| Qwen2.5-72B | 77.6% | 50.0% | 12.5% | 81.9% |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 73.2% | 37.5% | 48.4% | 90.8% |
| Nemotron-3-Super-120B-A12B-NVFP4 | 77.0% | 58.3% | 8.8% | 80.2% |

## Findings

Qwen 27B provided the strongest balanced classifier. It achieved 81.2% combined accuracy, only 4.1% data casualty, and only eight fail-closed errors. Its weakness for admission was a 14.7% combined contamination rate, including 13.1% for single-source, 17.6% for multi-source, and 28.6% for creative candidates. This made it attractive when retaining data mattered equally, but unsafe for a contamination-first training gate.

GPT-OSS-120B produced the lowest contamination rate for every lane: 7.4% single-source, 3.0% multi-source, 8.3% creative, and 6.6% combined. It also produced the lowest insufficient-gating rate for every lane and the highest admission-safe rate, 94.6%, when output errors failed closed. The cost was aggressive rejection: 19.4% combined data casualty, 52.4% unnecessary gating among valid rejects, and 14 output errors.

Llama 3.3 70B, Qwen2.5 72B, Qwen3.5 122B A10B, and Nemotron 120B did not improve the contamination objective. Their combined contamination rates were 19.2%, 19.6%, 10.5%, and 21.5%, respectively. Qwen3.5 122B was fast and ranked second to GPT-OSS on contamination, but it had 43 fail-closed errors and a 12.3% error rate. Nemotron was fast and conservative in casualty terms, but missed too many genuinely non-complete candidates.

## Decision rationale

The project assigns greater cost to false admission than to false rejection. A false-positive rejection remains in quarantine, can be sampled or fully reviewed later, and can be replaced by generating from additional papers. A false-negative admission silently contaminates training data and cannot be found without reviewing the admitted population. With approximately 20,000 papers available, additional generation time and candidate wastage are more affordable than undetected semantic contamination.

GPT-OSS-120B was therefore selected as the automated semantic gate because it minimized the risk that a materially incomplete or unsupported answer would enter the eligible pool. Qwen 27B remains the best balanced reviewer and a useful diagnostic or secondary challenger, but its lower rejection cost does not compensate for its higher contamination under the project objective.

The hybrid routing decision is:

1. Run the component-mapped coverage and adversarial-support protocol on every candidate assigned to GPT-OSS. Any axis error fails closed.
2. Use GPT-OSS as the final semantic gate for the dominant single-source lane. Only candidates judged complete and supported may advance. Every other single-source output remains quarantined and receives no gold review.
3. Send every multi-source, table, and comparison candidate to the gold reviewer because cross-source synthesis, attribution, comparison dimensions, and structured cells produced greater semantic risk. These lanes comprise 86/349, or 24.6%, of the prior V17 population, so complete gold review is materially cheaper than reviewing the entire dataset.
4. Inflate the starting corpus by approximately 40% when planning a production batch so conservative single-source rejection does not prevent the target clean-candidate yield. This is a capacity allowance, not permission to weaken the semantic gate.

The completed Qwen 122B 80-paper experiment established the routing inherited by the 1,000-paper Mini run: GPT-OSS-120B performs the complete single-source census, while every multi-source, table, and comparison candidate receives Codex gold review before training release. The earlier artifacts are evidence for this decision and are not runtime dependencies of the new run.

## Limits and release boundary

The benchmark contains 349 candidates from one Qwen 27B V17 generation run. Model rankings may change with new teacher outputs, response lengths, obligation distributions, or domain mix. GPT-OSS still admitted 22 gold-non-complete candidates among 335 valid outputs; its 6.6% measured contamination is lower, not zero. The gate therefore reduces automated single-source contamination risk but does not establish perfect semantic quality.

No automated verdict creates a training release. Candidate hashes, reviewer outputs, fail-closed errors, quarantine decisions, gold-review results, and final dispositions must remain auditable. Only an operator-approved immutable release may enter SFT.

## Evidence artifacts

- Selected metric table: `artifacts/reviewer_replication/reviewer_selected_metrics_by_lane.csv` (SHA-256 `d0cff723b16eec975a68d61688ae50939bab216ee3263794ce48821b81729fda`).
- Qwen-27B summary: `artifacts/reviewer_replication/qwen27b_component_dual_axis_full349/full349_summary.json` (SHA-256 `77fa0b2924c1e675d23a3a05a2d0645508938d7f12b63a446d7f5b34935712b4`).
- GPT-OSS-120B summary: `artifacts/reviewer_replication/gpt_oss_120b_mxfp4/component_dual_axis_full349/full349_summary.json` (SHA-256 `44471bce831d70a34f94dd70b782d14eed5e510426c1b81199aa00b1e0a5131e`).
- Llama-3.3-70B summary: `artifacts/reviewer_replication/llama33_70b_fp8/component_dual_axis_full349/full349_summary.json` (SHA-256 `8f6cf29ef37ab6fe7d5e9dd73ba2c1ed28a4435abd2cfa9eb09797920dee462e`).
- Qwen2.5-72B summary: `artifacts/reviewer_replication/qwen25_72b_fp8/component_dual_axis_full349/full349_summary.json` (SHA-256 `062e7c730d8d3b9c86561d4c6d3c5233ea3e5407a1eb740a94886dc39dcb7f5d`).
- Qwen3.5-122B-A10B-GPTQ-Int4 summary: `artifacts/reviewer_replication/qwen35_122b_a10b_gptq_int4/component_dual_axis_full349/full349_summary.json` (SHA-256 `57b3b8a9194d9880339676dde4a3b8832e1b26e5950480d35982dc7dad28301e`).
- Nemotron-3-Super-120B-A12B-NVFP4 summary: `artifacts/reviewer_replication/nemotron_nvfp4/component_dual_axis_full349/full349_summary.json` (SHA-256 `d814a20e764b1dbdb9b9d4e10adaa01970cd3a852f00e9b543d5d4ab7fd7c2c9`).
- Gold semantic census: `artifacts/v17/canary_80_v17/semantic_review/full_census/summary.json`.

