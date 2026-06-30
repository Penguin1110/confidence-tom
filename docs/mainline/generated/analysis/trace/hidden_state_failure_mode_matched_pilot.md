# Matched Hidden-State Failure Mode Pilot

## Question

- Repeat the reset-vs-ordinary failure pilot after matching on step fraction and token counts.
- This checks whether separability survives after removing the most obvious prefix-position confound.

## Aggregate

| Feature | Matched repeated mean AUROC | Matched orientation-free separability | Families |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.599 | 0.856 | 4 |
| last_token_hidden | 0.527 | 0.774 | 4 |
| controls_step_tokens | 0.502 | 0.771 | 4 |

## Family Breakdown

### `gemma4_no_outliers`

- matched pairs: `97` (reset original `97`, ordinary original `132`)
- matched step frac: reset `0.409`, ordinary `0.427`
- matched prefix tokens: reset `405.0`, ordinary `1038.2`

| Feature | Matched repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.724 (sep=0.744, n=50) |
| last_token_hidden | 0.648 (sep=0.689, n=50) |
| controls_step_tokens | 0.632 (sep=0.720, n=50) |

| Pooling | Matched centroid cosine | Matched centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.995 | 13.385 |
| last_token_hidden | 0.980 | 39.309 |

### `ministral_no_outliers`

- matched pairs: `10` (reset original `10`, ordinary original `160`)
- matched step frac: reset `0.368`, ordinary `0.368`
- matched prefix tokens: reset `136.2`, ordinary `156.5`

| Feature | Matched repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.581 (sep=0.995, n=29) |
| last_token_hidden | 0.436 (sep=0.909, n=29) |
| controls_step_tokens | 0.726 (sep=0.881, n=29) |

| Pooling | Matched centroid cosine | Matched centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.965 | 34.094 |
| last_token_hidden | 0.984 | 46.763 |

### `mistral7_no_outliers`

- matched pairs: `9` (reset original `9`, ordinary original `179`)
- matched step frac: reset `0.500`, ordinary `0.507`
- matched prefix tokens: reset `278.7`, ordinary `248.2`

| Feature | Matched repeated split AUROC |
| --- | ---: |
| mean_hidden | n/a |
| last_token_hidden | n/a |
| controls_step_tokens | n/a |

| Pooling | Matched centroid cosine | Matched centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.696 | 113.032 |
| last_token_hidden | 0.924 | 129.097 |

### `qwen25_no_outliers`

- matched pairs: `18` (reset original `18`, ordinary original `83`)
- matched step frac: reset `0.528`, ordinary `0.521`
- matched prefix tokens: reset `331.5`, ordinary `338.4`

| Feature | Matched repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.530 (sep=0.844, n=40) |
| last_token_hidden | 0.343 (sep=0.755, n=40) |
| controls_step_tokens | 0.300 (sep=0.745, n=40) |

| Pooling | Matched centroid cosine | Matched centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.993 | 15.162 |
| last_token_hidden | 0.997 | 20.612 |

### `qwen3_no_outliers`

- matched pairs: `43` (reset original `43`, ordinary original `100`)
- matched step frac: reset `0.493`, ordinary `0.486`
- matched prefix tokens: reset `606.1`, ordinary `622.5`

| Feature | Matched repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.562 (sep=0.841, n=45) |
| last_token_hidden | 0.680 (sep=0.745, n=45) |
| controls_step_tokens | 0.350 (sep=0.737, n=45) |

| Pooling | Matched centroid cosine | Matched centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.995 | 10.509 |
| last_token_hidden | 0.997 | 12.430 |

## Read

- A meaningful hidden-state result should remain above the matched step/token control.
- If the control remains competitive, the failure-mode distinction is not yet clean enough for a central claim.
