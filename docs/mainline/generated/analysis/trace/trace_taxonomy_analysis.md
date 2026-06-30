# Trace Taxonomy Analysis

## Definition

- Stable-success: full correct, early local re-entry correctness, and local correctness stays reasonably high.
- Late-success: full correct but not stable-success.
- Late-failure: some local re-entry correctness, but final full trace wrong.
- Persistent-failure: no local re-entry correctness and final full trace wrong.

## Overall Counts

| Category | Count | Share | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: |
| stable-success | 266 | 0.271 | 0.234 | 0.950 |
| late-success | 141 | 0.144 | 0.727 | 0.589 |
| late-failure | 127 | 0.129 | 0.458 | 0.446 |
| persistent-failure | 448 | 0.456 | 1.000 | 0.000 |

## Direct vs Re-entry

| Direct full correctness | N | Any small correct | Last small correct | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| True | 407 | 0.951 | 0.887 | 0.405 | 0.825 |
| False | 575 | 0.221 | 0.073 | 0.880 | 0.098 |

## Transition Matrix

- full correct & any small correct: `387`
- full correct & no small correct: `20`
- full wrong & any small correct: `127`
- full wrong & no small correct: `448`

## Benchmark Breakdown

| Benchmark | Stable | Late | Late-failure | Persistent | Mean first-correct frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| livebench_reasoning | 146 | 101 | 92 | 178 | 0.641 |
| olympiadbench | 120 | 40 | 35 | 270 | 0.731 |

## Family Breakdown

| Family | Stable | Late | Late-failure | Persistent | Mean first-correct frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| google | 0 | 0 | 0 | 6 | 1.000 |
| meta-llama | 43 | 23 | 33 | 128 | 0.741 |
| mistralai | 47 | 33 | 33 | 125 | 0.720 |
| ollama | 73 | 32 | 39 | 96 | 0.647 |
| qwen | 103 | 53 | 22 | 93 | 0.629 |

## Notes

- The taxonomy uses an early cutoff of 1/3 of the trace for stable-success.
- Tasks with full correctness but no local small-model correctness are absorbed into late-success under this definition.
- The key hypothesis is competence-conditioned rescue: re-entry helps most when the trace already has partial signal.
