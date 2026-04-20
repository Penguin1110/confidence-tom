# Hidden-State Routing Baseline

這份分析把 `Qwen + Mistral` 的 prefix hidden states 直接接到 routing simulator，做最小可跑版比較：

- `text_state_plus_family`: 現有 text/state-feature predictor
- `hidden_state_mean_pool_layer9`: 用各 family 的 layer-9 mean-pooled hidden state 預測 `P(small failure)`

兩個版本都只在 `qwen + mistral` 子集上訓練與評估，並使用相同的 stable task split、相同的 routing simulator、相同的 oracle comparator。

## text_state_plus_family

| max_step | fail threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 0.484 | 0.562 | 0.078 | 0.578 | 0.016 | 0.688 |
| 1 | 0.6 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.281 |
| 1 | 0.7 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.078 |
| 2 | 0.5 | 0.484 | 0.547 | 0.062 | 0.594 | 0.047 | 0.797 |
| 2 | 0.6 | 0.484 | 0.562 | 0.078 | 0.594 | 0.031 | 0.484 |
| 2 | 0.7 | 0.484 | 0.531 | 0.047 | 0.594 | 0.062 | 0.188 |
| 3 | 0.5 | 0.484 | 0.547 | 0.062 | 0.609 | 0.062 | 0.812 |
| 3 | 0.6 | 0.484 | 0.562 | 0.078 | 0.609 | 0.047 | 0.516 |
| 3 | 0.7 | 0.484 | 0.531 | 0.047 | 0.609 | 0.078 | 0.219 |

- best config: `max_step=2`, `failure_threshold=0.6`
- small baseline accuracy: `0.484`
- policy accuracy: `0.562`
- gain over small: `0.078`
- oracle accuracy under same budget: `0.594`
- gap to oracle: `0.031`
- route rate: `0.484`

## hidden_state_mean_pool_layer9

| max_step | fail threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.531 |
| 1 | 0.6 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.500 |
| 1 | 0.7 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.453 |
| 2 | 0.5 | 0.484 | 0.562 | 0.078 | 0.594 | 0.031 | 0.594 |
| 2 | 0.6 | 0.484 | 0.562 | 0.078 | 0.594 | 0.031 | 0.578 |
| 2 | 0.7 | 0.484 | 0.547 | 0.062 | 0.594 | 0.047 | 0.516 |
| 3 | 0.5 | 0.484 | 0.562 | 0.078 | 0.609 | 0.047 | 0.672 |
| 3 | 0.6 | 0.484 | 0.562 | 0.078 | 0.609 | 0.047 | 0.641 |
| 3 | 0.7 | 0.484 | 0.547 | 0.062 | 0.609 | 0.062 | 0.562 |

- best config: `max_step=2`, `failure_threshold=0.6`
- small baseline accuracy: `0.484`
- policy accuracy: `0.562`
- gain over small: `0.078`
- oracle accuracy under same budget: `0.594`
- gap to oracle: `0.031`
- route rate: `0.578`
