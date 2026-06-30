# Hidden-State Failure Mode Pilot

## Question

- Go/no-go pilot: can we separate two failure modes using hidden state?
- `reset-failure`: original full trace is correct, but this prefix re-entry is wrong.
- `ordinary-failure`: original full trace is wrong, and this prefix re-entry is wrong.
- Both groups use only `reentry_exact_correct = 0`, so the classifier cannot win by simply detecting local correctness.

## Aggregate AUROC

| Feature | Mean AUROC | Families with valid split |
| --- | ---: | ---: |
| mean_hidden | 0.289 | 4 |
| last_token_hidden | 0.410 | 4 |
| controls_step_tokens | 0.271 | 4 |

## Repeated Task-Split Robustness

| Feature | Repeated mean AUROC | Repeated orientation-free separability |
| --- | ---: | ---: |
| mean_hidden | 0.601 | 0.804 |
| last_token_hidden | 0.571 | 0.764 |
| controls_step_tokens | 0.647 | 0.727 |

## Family Breakdown

### `gemma4_no_outliers`

- rows: `229`; tasks: `28`; reset-failure rows: `97`; ordinary-failure rows: `132`
- mean step frac: reset `0.409`, ordinary `0.526`

| Feature | AUROC |
| --- | ---: |
| mean_hidden | 0.380 |
| last_token_hidden | 0.322 |
| controls_step_tokens | 0.140 |

| Feature | Repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.766 (sep=0.776, n=50) |
| last_token_hidden | 0.728 (sep=0.734, n=50) |
| controls_step_tokens | 0.693 (sep=0.755, n=50) |

| Pooling | Centroid cosine | Centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.984 | 22.873 |
| last_token_hidden | 0.964 | 52.806 |

### `ministral_no_outliers`

- rows: `170`; tasks: `29`; reset-failure rows: `10`; ordinary-failure rows: `160`
- mean step frac: reset `0.368`, ordinary `0.576`

| Feature | AUROC |
| --- | ---: |
| mean_hidden | 0.000 |
| last_token_hidden | 0.021 |
| controls_step_tokens | 0.700 |

| Feature | Repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.619 (sep=0.801, n=33) |
| last_token_hidden | 0.323 (sep=0.762, n=33) |
| controls_step_tokens | 0.749 (sep=0.749, n=33) |

| Pooling | Centroid cosine | Centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.965 | 34.196 |
| last_token_hidden | 0.977 | 56.237 |

### `mistral7_no_outliers`

- rows: `188`; tasks: `29`; reset-failure rows: `9`; ordinary-failure rows: `179`
- mean step frac: reset `0.500`, ordinary `0.581`

| Feature | AUROC |
| --- | ---: |
| mean_hidden | n/a |
| last_token_hidden | n/a |
| controls_step_tokens | n/a |

| Feature | Repeated split AUROC |
| --- | ---: |
| mean_hidden | n/a |
| last_token_hidden | n/a |
| controls_step_tokens | n/a |

| Pooling | Centroid cosine | Centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.707 | 110.474 |
| last_token_hidden | 0.926 | 126.267 |

### `qwen25_no_outliers`

- rows: `101`; tasks: `23`; reset-failure rows: `18`; ordinary-failure rows: `83`
- mean step frac: reset `0.528`, ordinary `0.612`

| Feature | AUROC |
| --- | ---: |
| mean_hidden | 0.500 |
| last_token_hidden | 0.625 |
| controls_step_tokens | 0.146 |

| Feature | Repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.468 (sep=0.789, n=40) |
| last_token_hidden | 0.433 (sep=0.739, n=40) |
| controls_step_tokens | 0.440 (sep=0.628, n=40) |

| Pooling | Centroid cosine | Centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.993 | 15.582 |
| last_token_hidden | 0.995 | 24.916 |

### `qwen3_no_outliers`

- rows: `143`; tasks: `21`; reset-failure rows: `43`; ordinary-failure rows: `100`
- mean step frac: reset `0.493`, ordinary `0.555`

| Feature | AUROC |
| --- | ---: |
| mean_hidden | 0.278 |
| last_token_hidden | 0.671 |
| controls_step_tokens | 0.098 |

| Feature | Repeated split AUROC |
| --- | ---: |
| mean_hidden | 0.550 (sep=0.849, n=40) |
| last_token_hidden | 0.799 (sep=0.821, n=40) |
| controls_step_tokens | 0.706 (sep=0.773, n=40) |

| Pooling | Centroid cosine | Centroid L2 |
| --- | ---: | ---: |
| mean_hidden | 0.993 | 13.187 |
| last_token_hidden | 0.995 | 17.036 |

## Read

- If hidden-state AUROC is clearly above the step/token control, the two failure modes have different directions in representation space.
- If hidden-state AUROC is near control or near 0.5, this line is probably not worth months of extra work without new data.
- This is an existing-data pilot, not the final causal test. It compares re-entry prompt hidden states, not original-generation KV-cache states.
