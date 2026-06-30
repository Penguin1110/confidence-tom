# Hidden State: Late-Success Vs Late-Failure

## Question

- Restrict to tasks that show some reusable local correctness signal.
- Ask whether hidden state can separate `late-success` from `late-failure`.

## Aggregate

| Pooling | Macro AUROC | Weighted AUROC | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| mean_hidden | 0.443 | 0.490 | 0.499 | 0.655 |
| last_token_hidden | 0.513 | 0.564 | 0.652 | 0.674 |

## Family Breakdown

### `gemma4_no_outliers`

- rows: `156`
- tasks: `13`
- selected layer: `42`
- hidden dim: `2560`

| Category | Rows | Tasks | Mean first-correct frac | Mean local correct rate | Mean last-small-correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| late-success | 117 | 9 | 0.496 | 0.410 | 1.000 |
| late-failure | 39 | 4 | 0.718 | 0.308 | 0.769 |

| Pooling | AUROC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| mean_hidden | 0.500 | 1.000 | 1.000 | 1.000 |
| last_token_hidden | 0.500 | 0.946 | 1.000 | 0.897 |

| Step Bin | Pooling | AUROC | F1 |
| --- | --- | ---: | ---: |
| step1 | mean_hidden | 0.500 | 1.000 |
| step2 | mean_hidden | 0.500 | 1.000 |
| step3 | mean_hidden | 0.500 | 1.000 |
| step4plus | mean_hidden | 0.500 | 1.000 |
| step1 | last_token_hidden | 0.500 | 1.000 |
| step2 | last_token_hidden | 0.500 | 1.000 |
| step3 | last_token_hidden | 0.500 | 1.000 |
| step4plus | last_token_hidden | 0.500 | 1.000 |

### `ministral_no_outliers`

- rows: `45`
- tasks: `6`
- selected layer: `36`
- hidden dim: `4096`

| Category | Rows | Tasks | Mean first-correct frac | Mean local correct rate | Mean last-small-correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| late-success | 17 | 3 | 0.647 | 0.471 | 1.000 |
| late-failure | 28 | 3 | 0.393 | 0.250 | 0.000 |

| Pooling | AUROC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| mean_hidden | 0.010 | 0.552 | 0.381 | 1.000 |
| last_token_hidden | 0.173 | 0.552 | 0.381 | 1.000 |

| Step Bin | Pooling | AUROC | F1 |
| --- | --- | ---: | ---: |
| step1 | mean_hidden | 1.000 | 0.667 |
| step2 | mean_hidden | 0.000 | 0.667 |
| step3 | mean_hidden | 0.000 | 0.667 |
| step4plus | mean_hidden | 0.220 | 0.500 |
| step1 | last_token_hidden | 0.000 | 0.667 |
| step2 | last_token_hidden | 0.000 | 0.667 |
| step3 | last_token_hidden | 0.000 | 0.667 |
| step4plus | last_token_hidden | 0.320 | 0.500 |

### `mistral7_no_outliers`

- rows: `60`
- tasks: `8`
- selected layer: `32`
- hidden dim: `4096`

| Category | Rows | Tasks | Mean first-correct frac | Mean local correct rate | Mean last-small-correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| late-success | 10 | 1 | 1.000 | 0.100 | 1.000 |
| late-failure | 50 | 7 | 0.340 | 0.260 | 0.100 |

| Pooling | AUROC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| mean_hidden | n/a | n/a | n/a | n/a |
| last_token_hidden | n/a | n/a | n/a | n/a |

| Step Bin | Pooling | AUROC | F1 |
| --- | --- | ---: | ---: |

### `qwen25_no_outliers`

- rows: `63`
- tasks: `11`
- selected layer: `48`
- hidden dim: `5120`

| Category | Rows | Tasks | Mean first-correct frac | Mean local correct rate | Mean last-small-correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| late-success | 23 | 4 | 0.609 | 0.304 | 0.696 |
| late-failure | 40 | 7 | 0.450 | 0.250 | 0.000 |

| Pooling | AUROC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| mean_hidden | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | 0.500 | 0.667 | 1.000 | 0.500 |

| Step Bin | Pooling | AUROC | F1 |
| --- | --- | ---: | ---: |
| step1 | mean_hidden | 0.500 | 0.000 |
| step2 | mean_hidden | 0.500 | 0.000 |
| step3 | mean_hidden | 0.500 | 0.000 |
| step4plus | mean_hidden | 0.500 | 0.000 |
| step1 | last_token_hidden | 0.500 | 0.000 |
| step2 | last_token_hidden | 0.500 | 1.000 |
| step3 | last_token_hidden | 0.500 | 0.000 |
| step4plus | last_token_hidden | 0.500 | 0.800 |

### `qwen3_no_outliers`

- rows: `96`
- tasks: `11`
- selected layer: `40`
- hidden dim: `5120`

| Category | Rows | Tasks | Mean first-correct frac | Mean local correct rate | Mean last-small-correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| late-success | 51 | 6 | 0.941 | 0.176 | 1.000 |
| late-failure | 45 | 5 | 0.444 | 0.422 | 0.511 |

| Pooling | AUROC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| mean_hidden | 0.764 | 0.444 | 0.286 | 1.000 |
| last_token_hidden | 0.880 | 0.444 | 0.286 | 1.000 |

| Step Bin | Pooling | AUROC | F1 |
| --- | --- | ---: | ---: |
| step1 | mean_hidden | 0.500 | 0.500 |
| step2 | mean_hidden | 0.000 | 0.500 |
| step3 | mean_hidden | 0.000 | 0.500 |
| step4plus | mean_hidden | 0.947 | 0.424 |
| step1 | last_token_hidden | 0.500 | 0.500 |
| step2 | last_token_hidden | 0.500 | 0.500 |
| step3 | last_token_hidden | 1.000 | 0.500 |
| step4plus | last_token_hidden | 0.654 | 0.424 |
