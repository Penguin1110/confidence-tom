# Hidden State At First-Correct Onset

## Question

- Look only at tasks that ever become locally correct.
- Take the hidden state at the first correct prefix.
- Ask whether that onset state predicts preservation to the end.

## Aggregate AUROC

| Pooling | Predict full-trace correct | Predict last-small-correct | Predict collapse-after-onset |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.354 | 0.389 | 0.667 |
| last_token_hidden | 0.438 | 0.500 | 0.833 |

## Family Breakdown

### `gemma4_no_outliers`

- tasks with any local correctness: `22`
- mean onset frac: `0.423`
- mean tail correct rate: `0.843`
- category counts: `{"late-failure": 4, "late-success": 9, "stable-success": 9}`

| Pooling | Full-trace correct AUROC | Last-small-correct AUROC | Collapse-after-onset AUROC |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.667 | n/a | 1.000 |
| last_token_hidden | 1.000 | n/a | 1.000 |

### `ministral_no_outliers`

- tasks with any local correctness: `7`
- mean onset frac: `0.489`
- mean tail correct rate: `0.665`
- category counts: `{"late-failure": 3, "late-success": 3, "stable-success": 1}`

| Pooling | Full-trace correct AUROC | Last-small-correct AUROC | Collapse-after-onset AUROC |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.000 | 0.000 | n/a |
| last_token_hidden | 0.000 | 0.000 | n/a |

### `mistral7_no_outliers`

- tasks with any local correctness: `8`
- mean onset frac: `0.468`
- mean tail correct rate: `0.535`
- category counts: `{"late-success": 1, "late-failure": 7}`

| Pooling | Full-trace correct AUROC | Last-small-correct AUROC | Collapse-after-onset AUROC |
| --- | ---: | ---: | ---: |
| mean_hidden | n/a | n/a | n/a |
| last_token_hidden | n/a | n/a | n/a |

### `qwen25_no_outliers`

- tasks with any local correctness: `16`
- mean onset frac: `0.429`
- mean tail correct rate: `0.653`
- category counts: `{"late-success": 4, "stable-success": 5, "late-failure": 7}`

| Pooling | Full-trace correct AUROC | Last-small-correct AUROC | Collapse-after-onset AUROC |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.000 | 0.500 | 0.000 |
| last_token_hidden | 0.000 | 0.500 | 0.500 |

### `qwen3_no_outliers`

- tasks with any local correctness: `17`
- mean onset frac: `0.490`
- mean tail correct rate: `0.902`
- category counts: `{"late-success": 6, "stable-success": 6, "late-failure": 5}`

| Pooling | Full-trace correct AUROC | Last-small-correct AUROC | Collapse-after-onset AUROC |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.750 | 0.667 | 1.000 |
| last_token_hidden | 0.750 | 1.000 | 1.000 |

## Main Read

- This analysis isolates the onset state: not whether correctness ever appears, but whether the first correct state is already a preservable state.
- If onset-state prediction works, it supports a `state persistence bottleneck` interpretation.
