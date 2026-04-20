# Hidden-State Positive-Only Routing

這份分析直接用 `positive takeover` 當 routing label，比較：

- `text_positive_only`: 用 prefix predictor state features 預測 `delta_positive`
- `hidden_state_positive_only`: 用 family-specific hidden state 預測 `delta_positive`

兩個版本都只在 `qwen + mistral` 子集上訓練與評估，並使用相同的 test task split 與 oracle comparator。

## text_positive_only

| max_step | positive threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2 | 0.484 | 0.562 | 0.078 | 0.578 | 0.016 | 0.531 |
| 1 | 0.3 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.453 |
| 1 | 0.4 | 0.484 | 0.531 | 0.047 | 0.578 | 0.047 | 0.062 |
| 1 | 0.5 | 0.484 | 0.516 | 0.031 | 0.578 | 0.062 | 0.031 |
| 2 | 0.2 | 0.484 | 0.562 | 0.078 | 0.594 | 0.031 | 0.531 |
| 2 | 0.3 | 0.484 | 0.547 | 0.062 | 0.594 | 0.047 | 0.453 |
| 2 | 0.4 | 0.484 | 0.531 | 0.047 | 0.594 | 0.062 | 0.062 |
| 2 | 0.5 | 0.484 | 0.516 | 0.031 | 0.594 | 0.078 | 0.031 |
| 3 | 0.2 | 0.484 | 0.562 | 0.078 | 0.609 | 0.047 | 0.531 |
| 3 | 0.3 | 0.484 | 0.547 | 0.062 | 0.609 | 0.062 | 0.453 |
| 3 | 0.4 | 0.484 | 0.531 | 0.047 | 0.609 | 0.078 | 0.062 |
| 3 | 0.5 | 0.484 | 0.516 | 0.031 | 0.609 | 0.094 | 0.031 |

- best config: `max_step=1`, `positive_threshold=0.2`
- policy accuracy: `0.562`
- gain over small: `0.078`
- oracle accuracy: `0.578`
- gap to oracle: `0.016`
- route rate: `0.531`

## hidden_state_positive_only

| max_step | positive threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.484 |
| 1 | 0.3 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.453 |
| 1 | 0.4 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.422 |
| 1 | 0.5 | 0.484 | 0.547 | 0.062 | 0.578 | 0.031 | 0.406 |
| 2 | 0.2 | 0.484 | 0.547 | 0.062 | 0.594 | 0.047 | 0.562 |
| 2 | 0.3 | 0.484 | 0.547 | 0.062 | 0.594 | 0.047 | 0.516 |
| 2 | 0.4 | 0.484 | 0.547 | 0.062 | 0.594 | 0.047 | 0.484 |
| 2 | 0.5 | 0.484 | 0.547 | 0.062 | 0.594 | 0.047 | 0.453 |
| 3 | 0.2 | 0.484 | 0.547 | 0.062 | 0.609 | 0.062 | 0.578 |
| 3 | 0.3 | 0.484 | 0.547 | 0.062 | 0.609 | 0.062 | 0.531 |
| 3 | 0.4 | 0.484 | 0.547 | 0.062 | 0.609 | 0.062 | 0.516 |
| 3 | 0.5 | 0.484 | 0.547 | 0.062 | 0.609 | 0.062 | 0.453 |

- best config: `max_step=1`, `positive_threshold=0.2`
- policy accuracy: `0.547`
- gain over small: `0.062`
- oracle accuracy: `0.578`
- gap to oracle: `0.031`
- route rate: `0.484`
