# Trace Taxonomy Analysis

## Definition

- Stable-success: full correct, early local re-entry correctness, and local correctness stays reasonably high.
- Late-success: full correct but not stable-success.
- Fragile-success: some local re-entry correctness, but final full trace wrong.
- Persistent-failure: no local re-entry correctness and final full trace wrong.

## Overall Counts

| Category | Count | Share | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: |
| stable-success | 178 | 0.253 | 0.228 | 0.948 |
| late-success | 96 | 0.137 | 0.739 | 0.561 |
| fragile-success | 83 | 0.118 | 0.427 | 0.447 |
| persistent-failure | 346 | 0.492 | 1.000 | 0.000 |

## Direct vs Re-entry

| Direct full correctness | N | Any small correct | Last small correct | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| True | 274 | 0.945 | 0.883 | 0.407 | 0.812 |
| False | 429 | 0.193 | 0.061 | 0.889 | 0.087 |

## Transition Matrix

- full correct & any small correct: `259`
- full correct & no small correct: `15`
- full wrong & any small correct: `83`
- full wrong & no small correct: `346`

## Benchmark Breakdown

| Benchmark | Stable | Late | Fragile | Persistent | Mean first-correct frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| livebench_reasoning | 59 | 56 | 48 | 76 | 0.642 |
| olympiadbench | 119 | 40 | 35 | 270 | 0.732 |

## Family Breakdown

| Family | Stable | Late | Fragile | Persistent | Mean first-correct frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| google | 0 | 0 | 0 | 6 | 1.000 |
| meta-llama | 43 | 23 | 33 | 128 | 0.741 |
| mistralai | 47 | 33 | 33 | 125 | 0.720 |
| qwen | 88 | 40 | 17 | 87 | 0.635 |

## Notes

- The taxonomy uses an early cutoff of 1/3 of the trace for stable-success.
- Tasks with full correctness but no local small-model correctness are absorbed into late-success under this definition.
- The key hypothesis is competence-conditioned rescue: re-entry helps most when the trace already has partial signal.
