# Trace Taxonomy Analysis (Ollama Only)

## Definition

- Stable-success: full correct, early local re-entry correctness, and local correctness stays reasonably high.
- Late-success: full correct but not stable-success.
- Late-failure: some local re-entry correctness, but final full trace wrong.
- Persistent-failure: no local re-entry correctness and final full trace wrong.

## Overall Counts

| Category | Count | Share | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: |
| stable-success | 73 | 0.304 | 0.253 | 0.951 |
| late-success | 32 | 0.133 | 0.628 | 0.562 |
| late-failure | 39 | 0.163 | 0.527 | 0.424 |
| persistent-failure | 96 | 0.400 | 1.000 | 0.000 |

## Direct vs Re-entry

| Direct full correctness | N | Any small correct | Last small correct | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| True | 105 | 0.962 | 0.905 | 0.368 | 0.833 |
| False | 135 | 0.289 | 0.096 | 0.863 | 0.123 |

## Transition Matrix

- full correct & any small correct: `101`
- full correct & no small correct: `4`
- full wrong & any small correct: `39`
- full wrong & no small correct: `96`

## Benchmark Breakdown

| Benchmark | Stable | Late | Late-failure | Persistent | Mean first-correct frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| livebench_reasoning | 73 | 32 | 39 | 96 | 0.647 |

## Family Breakdown

| Family | Stable | Late | Late-failure | Persistent | Mean first-correct frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| ollama | 73 | 32 | 39 | 96 | 0.647 |
