# Hidden State Analysis For LiveBench Re-entry

## Setup

- Source: `outputs/results/imported/results-livebench-reentry-a07127a`
- Scope: only `*_no_outliers` LiveBench re-entry runs
- Split: stable task-level train/test split by `task_id`
- Taxonomy: same definition as `analyze_trace_taxonomy.py`

## Aggregate Probe Quality

| Pooling | Target | Macro AUROC | Weighted AUROC | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| mean_hidden | full_trace_correct | 0.335 | 0.370 | 0.267 | 0.332 |
| mean_hidden | reentry_exact_correct | 0.556 | 0.524 | 0.320 | 0.315 |
| mean_hidden | late_failure_task | 0.610 | 0.627 | 0.167 | 0.205 |
| mean_hidden | stable_success_task | 0.500 | 0.500 | 0.000 | 0.000 |
| mean_hidden | persistent_failure_task | 0.700 | 0.661 | 0.448 | 0.345 |
| last_token_hidden | full_trace_correct | 0.479 | 0.463 | 0.302 | 0.276 |
| last_token_hidden | reentry_exact_correct | 0.753 | 0.757 | 0.336 | 0.329 |
| last_token_hidden | late_failure_task | 0.551 | 0.566 | 0.077 | 0.095 |
| last_token_hidden | stable_success_task | 0.500 | 0.500 | 0.000 | 0.000 |
| last_token_hidden | persistent_failure_task | 0.657 | 0.633 | 0.448 | 0.345 |

## Family Breakdown

### `gemma4_no_outliers`

- rows: `401`
- tasks: `29`
- selected layer: `42`
- hidden dim: `2560`
- category counts: `{"late-success": 117, "persistent-failure": 105, "stable-success": 140, "late-failure": 39}`

| Pooling | Target | AUROC | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| mean_hidden | full_trace_correct | 0.500 | 0.582 | 1.000 | 0.410 |
| mean_hidden | reentry_exact_correct | 0.429 | 0.600 | 1.000 | 0.429 |
| mean_hidden | late_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | persistent_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | full_trace_correct | 0.500 | 0.267 | 1.000 | 0.154 |
| last_token_hidden | reentry_exact_correct | 0.926 | 0.444 | 1.000 | 0.286 |
| last_token_hidden | late_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | persistent_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |

| Step Bin | Pooling | Target | AUROC | F1 |
| --- | --- | --- | ---: | ---: |
| step1 | mean_hidden | full_trace_correct | 0.500 | 0.667 |
| step1 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | mean_hidden | full_trace_correct | 0.500 | 0.667 |
| step2 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | mean_hidden | full_trace_correct | 0.500 | 0.667 |
| step3 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | mean_hidden | full_trace_correct | 0.500 | 0.565 |
| step4plus | mean_hidden | reentry_exact_correct | 0.619 | 0.000 |
| step1 | last_token_hidden | full_trace_correct | 0.500 | 0.667 |
| step1 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | last_token_hidden | full_trace_correct | 0.500 | 1.000 |
| step2 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | last_token_hidden | full_trace_correct | 0.500 | 1.000 |
| step3 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | last_token_hidden | full_trace_correct | 0.500 | 0.465 |
| step4plus | last_token_hidden | reentry_exact_correct | 0.897 | 0.320 |

### `ministral_no_outliers`

- rows: `190`
- tasks: `29`
- selected layer: `36`
- hidden dim: `4096`
- category counts: `{"persistent-failure": 139, "stable-success": 6, "late-success": 17, "late-failure": 28}`

| Pooling | Target | AUROC | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| mean_hidden | full_trace_correct | 0.435 | 0.421 | 0.267 | 1.000 |
| mean_hidden | reentry_exact_correct | 0.481 | 0.000 | 0.000 | 0.000 |
| mean_hidden | late_failure_task | 0.752 | 0.000 | 0.000 | 0.000 |
| mean_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | persistent_failure_task | 1.000 | 0.488 | 0.323 | 1.000 |
| last_token_hidden | full_trace_correct | 0.060 | 0.065 | 0.043 | 0.125 |
| last_token_hidden | reentry_exact_correct | 0.861 | 0.348 | 0.211 | 1.000 |
| last_token_hidden | late_failure_task | 0.359 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | persistent_failure_task | 0.952 | 0.488 | 0.323 | 1.000 |

| Step Bin | Pooling | Target | AUROC | F1 |
| --- | --- | --- | ---: | ---: |
| step1 | mean_hidden | full_trace_correct | 0.000 | 0.500 |
| step1 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | mean_hidden | full_trace_correct | 0.000 | 0.500 |
| step2 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | mean_hidden | full_trace_correct | 0.000 | 0.500 |
| step3 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | mean_hidden | full_trace_correct | 0.153 | 0.370 |
| step4plus | mean_hidden | reentry_exact_correct | 0.264 | 0.333 |
| step1 | last_token_hidden | full_trace_correct | 0.000 | 0.000 |
| step1 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | last_token_hidden | full_trace_correct | 0.000 | 0.000 |
| step2 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | last_token_hidden | full_trace_correct | 0.500 | 0.500 |
| step3 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | last_token_hidden | full_trace_correct | 0.612 | 0.385 |
| step4plus | last_token_hidden | reentry_exact_correct | 0.569 | 0.444 |

### `mistral7_no_outliers`

- rows: `202`
- tasks: `29`
- selected layer: `32`
- hidden dim: `4096`
- category counts: `{"late-failure": 50, "persistent-failure": 142, "late-success": 10}`

| Pooling | Target | AUROC | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| mean_hidden | full_trace_correct | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | reentry_exact_correct | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | late_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | persistent_failure_task | 0.500 | 1.000 | 1.000 | 1.000 |
| last_token_hidden | full_trace_correct | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | reentry_exact_correct | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | late_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | persistent_failure_task | 0.500 | 1.000 | 1.000 | 1.000 |

| Step Bin | Pooling | Target | AUROC | F1 |
| --- | --- | --- | ---: | ---: |
| step1 | mean_hidden | full_trace_correct | 0.500 | 0.000 |
| step1 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | mean_hidden | full_trace_correct | 0.500 | 0.000 |
| step2 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | mean_hidden | full_trace_correct | 0.500 | 0.000 |
| step3 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | mean_hidden | full_trace_correct | 0.500 | 0.000 |
| step4plus | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step1 | last_token_hidden | full_trace_correct | 0.500 | 0.000 |
| step1 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | last_token_hidden | full_trace_correct | 0.500 | 0.000 |
| step2 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | last_token_hidden | full_trace_correct | 0.500 | 0.000 |
| step3 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | last_token_hidden | full_trace_correct | 0.500 | 0.000 |
| step4plus | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |

### `qwen25_no_outliers`

- rows: `138`
- tasks: `27`
- selected layer: `48`
- hidden dim: `5120`
- category counts: `{"late-failure": 40, "persistent-failure": 53, "stable-success": 22, "late-success": 23}`

| Pooling | Target | AUROC | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| mean_hidden | full_trace_correct | 0.000 | 0.000 | 0.000 | 0.000 |
| mean_hidden | reentry_exact_correct | 0.806 | 0.667 | 0.500 | 1.000 |
| mean_hidden | late_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | persistent_failure_task | 1.000 | 0.750 | 0.600 | 1.000 |
| last_token_hidden | full_trace_correct | 0.667 | 0.714 | 0.625 | 0.833 |
| last_token_hidden | reentry_exact_correct | 0.889 | 0.600 | 0.429 | 1.000 |
| last_token_hidden | late_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | persistent_failure_task | 0.833 | 0.750 | 0.600 | 1.000 |

| Step Bin | Pooling | Target | AUROC | F1 |
| --- | --- | --- | ---: | ---: |
| step1 | mean_hidden | full_trace_correct | 0.000 | 0.000 |
| step1 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | mean_hidden | full_trace_correct | 0.000 | 0.000 |
| step2 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | mean_hidden | full_trace_correct | 0.000 | 0.000 |
| step3 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | mean_hidden | full_trace_correct | 0.000 | 0.000 |
| step4plus | mean_hidden | reentry_exact_correct | 0.000 | 0.000 |
| step1 | last_token_hidden | full_trace_correct | 0.000 | 0.000 |
| step1 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | last_token_hidden | full_trace_correct | 1.000 | 0.500 |
| step2 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | last_token_hidden | full_trace_correct | 0.500 | 0.000 |
| step3 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | last_token_hidden | full_trace_correct | 0.667 | 0.000 |
| step4plus | last_token_hidden | reentry_exact_correct | 0.778 | 0.571 |

### `qwen3_no_outliers`

- rows: `236`
- tasks: `27`
- selected layer: `40`
- hidden dim: `5120`
- category counts: `{"late-failure": 45, "persistent-failure": 74, "late-success": 51, "stable-success": 66}`

| Pooling | Target | AUROC | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| mean_hidden | full_trace_correct | 0.240 | 0.333 | 0.231 | 0.600 |
| mean_hidden | reentry_exact_correct | 0.565 | 0.333 | 0.500 | 0.250 |
| mean_hidden | late_failure_task | 0.796 | 0.837 | 1.000 | 0.720 |
| mean_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| mean_hidden | persistent_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | full_trace_correct | 0.668 | 0.467 | 0.350 | 0.700 |
| last_token_hidden | reentry_exact_correct | 0.589 | 0.286 | 0.333 | 0.250 |
| last_token_hidden | late_failure_task | 0.896 | 0.387 | 1.000 | 0.240 |
| last_token_hidden | stable_success_task | 0.500 | 0.000 | 0.000 | 0.000 |
| last_token_hidden | persistent_failure_task | 0.500 | 0.000 | 0.000 | 0.000 |

| Step Bin | Pooling | Target | AUROC | F1 |
| --- | --- | --- | ---: | ---: |
| step1 | mean_hidden | full_trace_correct | 0.500 | 0.667 |
| step1 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | mean_hidden | full_trace_correct | 0.500 | 0.500 |
| step2 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | mean_hidden | full_trace_correct | 0.000 | 0.500 |
| step3 | mean_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | mean_hidden | full_trace_correct | 0.940 | 0.438 |
| step4plus | mean_hidden | reentry_exact_correct | 0.614 | 0.300 |
| step1 | last_token_hidden | full_trace_correct | 0.500 | 0.500 |
| step1 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step2 | last_token_hidden | full_trace_correct | 0.000 | 0.500 |
| step2 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step3 | last_token_hidden | full_trace_correct | 0.000 | 0.500 |
| step3 | last_token_hidden | reentry_exact_correct | 0.500 | 0.000 |
| step4plus | last_token_hidden | full_trace_correct | 0.624 | 0.424 |
| step4plus | last_token_hidden | reentry_exact_correct | 0.398 | 0.167 |
