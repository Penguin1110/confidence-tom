# Trace Persistence Subtypes

## Question

- Split the main taxonomy into more persistence-oriented subtypes.
- Ask whether `late-success` and `late-failure` differ more by onset timing or by post-onset survival.

## Category Summary

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Tail correct rate | Collapse count | Longest correct streak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 21 | 0.153 | 0.884 | 1.000 | 0.900 | 0.905 | 7.619 |
| late-success | 23 | 0.710 | 0.326 | 0.957 | 0.878 | 0.522 | 2.522 |
| late-failure | 26 | 0.467 | 0.332 | 0.269 | 0.545 | 1.000 | 1.923 |
| persistent-failure | 71 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Subtype Summary

| Subtype | Tasks | First-correct frac | Tail correct rate | Collapse count | Streak after onset |
| --- | ---: | ---: | ---: | ---: | ---: |
| early-lock-in | 21 | 0.153 | 0.900 | 0.905 | 7.619 |
| delayed-lock-in | 17 | 0.835 | 1.000 | 0.000 | 2.647 |
| oscillatory-success | 5 | 0.400 | 0.609 | 2.200 | 2.400 |
| fragile-flash | 8 | 0.467 | 0.262 | 1.000 | 1.000 |
| rescue-then-collapse | 11 | 0.300 | 0.494 | 1.455 | 2.182 |
| terminal-near-miss | 7 | 0.728 | 0.948 | 0.286 | 2.571 |
| never-correct | 71 | 1.000 | 0.000 | 0.000 | 0.000 |

## Family Breakdown

### `gemma4`

- tasks: `29`
- category counts: `{"late-success": 9, "persistent-failure": 7, "stable-success": 9, "late-failure": 4}`
- subtype counts: `{"delayed-lock-in": 5, "never-correct": 7, "early-lock-in": 9, "rescue-then-collapse": 1, "oscillatory-success": 4, "terminal-near-miss": 3}`

### `ministral`

- tasks: `29`
- category counts: `{"persistent-failure": 22, "stable-success": 1, "late-success": 3, "late-failure": 3}`
- subtype counts: `{"never-correct": 22, "early-lock-in": 1, "oscillatory-success": 1, "delayed-lock-in": 2, "rescue-then-collapse": 1, "fragile-flash": 2}`

### `mistral7`

- tasks: `29`
- category counts: `{"late-failure": 7, "persistent-failure": 21, "late-success": 1}`
- subtype counts: `{"rescue-then-collapse": 4, "never-correct": 21, "terminal-near-miss": 1, "delayed-lock-in": 1, "fragile-flash": 2}`

### `qwen25`

- tasks: `27`
- category counts: `{"late-failure": 7, "persistent-failure": 11, "stable-success": 5, "late-success": 4}`
- subtype counts: `{"fragile-flash": 4, "never-correct": 11, "early-lock-in": 5, "delayed-lock-in": 3, "rescue-then-collapse": 3, "anomalous-late-success": 1}`

### `qwen3`

- tasks: `27`
- category counts: `{"late-failure": 5, "persistent-failure": 10, "late-success": 6, "stable-success": 6}`
- subtype counts: `{"terminal-near-miss": 3, "never-correct": 10, "delayed-lock-in": 6, "early-lock-in": 6, "rescue-then-collapse": 2}`

## Main Read

- `late-success` is later on average, but once correctness appears it is much more likely to survive.
- `late-failure` contains at least two distinct modes: a fragile one-flash mode and a rescue-then-collapse mode.
- This supports a `state formation vs state preservation` decomposition rather than a single-step sufficiency story.
