# Advanced Routing Policies

這份分析把原本只看 `P(small failure)` 的 routing baseline，升級成三種更強的 policy：

- `dual_signal`: failure + positive takeover
- `risk_aware`: failure + positive takeover + negative risk
- `benchmark_aware_risk_aware`: 為每個 benchmark 分開選 risk-aware threshold

| policy | max_step | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| failure_only | 1 | 0.438 | 0.562 | 0.125 | 0.635 | 0.073 | 0.667 |
| dual_signal | 2 | 0.438 | 0.573 | 0.135 | 0.656 | 0.083 | 0.635 |
| risk_aware | 2 | 0.438 | 0.573 | 0.135 | 0.656 | 0.083 | 0.635 |
| benchmark_aware_risk_aware | 2 | 0.438 | 0.573 | 0.135 | 0.656 | 0.083 | 0.635 |

## failure_only

- validation-selected config: `{"policy_type": "failure_only", "max_step": 1, "fail_threshold": 0.5}`
- test small baseline accuracy: `0.438`
- test policy accuracy: `0.562`
- test gain over small: `0.125`
- test oracle accuracy: `0.635`
- test route rate: `0.667`

### By Benchmark

#### olympiadbench

- tasks: `78`
- small baseline accuracy: `0.449`
- policy accuracy: `0.474`
- gain over small: `0.026`
- route rate: `0.654`

#### livebench_reasoning

- tasks: `18`
- small baseline accuracy: `0.389`
- policy accuracy: `0.944`
- gain over small: `0.556`
- route rate: `0.722`


## dual_signal

- validation-selected config: `{"policy_type": "dual_signal", "max_step": 2, "fail_threshold": 0.6, "positive_threshold": 0.2}`
- test small baseline accuracy: `0.438`
- test policy accuracy: `0.573`
- test gain over small: `0.135`
- test oracle accuracy: `0.656`
- test route rate: `0.635`

### By Benchmark

#### olympiadbench

- tasks: `78`
- small baseline accuracy: `0.449`
- policy accuracy: `0.500`
- gain over small: `0.051`
- route rate: `0.628`

#### livebench_reasoning

- tasks: `18`
- small baseline accuracy: `0.389`
- policy accuracy: `0.889`
- gain over small: `0.500`
- route rate: `0.667`


## risk_aware

- validation-selected config: `{"policy_type": "risk_aware", "max_step": 2, "fail_threshold": 0.6, "positive_threshold": 0.2, "negative_threshold": 0.2}`
- test small baseline accuracy: `0.438`
- test policy accuracy: `0.573`
- test gain over small: `0.135`
- test oracle accuracy: `0.656`
- test route rate: `0.635`

### By Benchmark

#### olympiadbench

- tasks: `78`
- small baseline accuracy: `0.449`
- policy accuracy: `0.500`
- gain over small: `0.051`
- route rate: `0.628`

#### livebench_reasoning

- tasks: `18`
- small baseline accuracy: `0.389`
- policy accuracy: `0.889`
- gain over small: `0.500`
- route rate: `0.667`


## benchmark_aware_risk_aware

- validation-selected config: `{"policy_type": "benchmark_aware_risk_aware", "max_step": 2, "per_benchmark": {"livebench_reasoning": {"fail_threshold": 0.4, "positive_threshold": 0.2, "negative_threshold": 0.2}, "olympiadbench": {"fail_threshold": 0.6, "positive_threshold": 0.2, "negative_threshold": 0.2}}}`
- test small baseline accuracy: `0.438`
- test policy accuracy: `0.573`
- test gain over small: `0.135`
- test oracle accuracy: `0.656`
- test route rate: `0.635`

### By Benchmark

#### olympiadbench

- tasks: `78`
- small baseline accuracy: `0.449`
- policy accuracy: `0.500`
- gain over small: `0.051`
- route rate: `0.628`

#### livebench_reasoning

- tasks: `18`
- small baseline accuracy: `0.389`
- policy accuracy: `0.889`
- gain over small: `0.500`
- route rate: `0.667`
