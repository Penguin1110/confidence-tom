# Hidden-State Routing Layer Sweep

這份分析在 `qwen + mistral` 子集上，對 hidden-state routing 做 family-specific layer sweep。

- hidden state feature: `mean_pooled`
- qwen candidate layers: `0, 9, 19, 29, 39`
- mistral candidate layers: `0, 9, 19, 29, 35`
- routing search: `max_step in {1,2,3}`, `failure_threshold in {0.5, 0.6, 0.7}`

## Text Baseline

- best config: `max_step=2`, `failure_threshold=0.6`
- policy accuracy: `0.562`
- gain over small: `0.078`
- route rate: `0.484`

## Best Hidden-State Policy

- qwen layer: `19`
- mistral layer: `9`
- best config: `max_step=2`, `failure_threshold=0.6`
- policy accuracy: `0.562`
- gain over small: `0.078`
- route rate: `0.531`
- oracle accuracy under same budget: `0.594`
- gap to oracle: `0.031`

## Top Hidden-State Combinations

| qwen layer | mistral layer | max_step | fail threshold | policy acc | gain | oracle acc | gap | route rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 1 | 0.6 | 0.562 | 0.078 | 0.578 | 0.016 | 0.562 |
| 29 | 0 | 1 | 0.6 | 0.562 | 0.078 | 0.578 | 0.016 | 0.594 |
| 9 | 0 | 1 | 0.5 | 0.562 | 0.078 | 0.578 | 0.016 | 0.609 |
| 29 | 0 | 1 | 0.5 | 0.562 | 0.078 | 0.578 | 0.016 | 0.625 |
| 39 | 0 | 1 | 0.6 | 0.562 | 0.078 | 0.578 | 0.016 | 0.641 |
| 39 | 0 | 1 | 0.5 | 0.562 | 0.078 | 0.578 | 0.016 | 0.656 |
| 19 | 9 | 2 | 0.6 | 0.562 | 0.078 | 0.594 | 0.031 | 0.531 |
| 9 | 9 | 2 | 0.6 | 0.562 | 0.078 | 0.594 | 0.031 | 0.578 |
| 19 | 9 | 2 | 0.5 | 0.562 | 0.078 | 0.594 | 0.031 | 0.578 |
| 9 | 9 | 2 | 0.5 | 0.562 | 0.078 | 0.594 | 0.031 | 0.594 |
