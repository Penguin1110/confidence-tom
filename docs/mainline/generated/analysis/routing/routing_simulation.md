# Routing Simulation Baseline

這份模擬把 Early Decision predictor 直接接到 prefix-level routing policy：當 `P(small failure)` 超過門檻時，就在最早可用 prefix 交給 large takeover。

- test tasks: `96`

## Top Configs By Gain

| max_step | fail threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate | mean trigger step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.6 | 0.438 | 0.583 | 0.146 | 0.656 | 0.073 | 0.625 | 1.383 |
| 3 | 0.6 | 0.438 | 0.583 | 0.146 | 0.667 | 0.083 | 0.656 | 1.460 |
| 1 | 0.5 | 0.438 | 0.573 | 0.135 | 0.635 | 0.062 | 0.646 | 1.000 |
| 2 | 0.5 | 0.438 | 0.562 | 0.125 | 0.656 | 0.094 | 0.802 | 1.195 |
| 3 | 0.5 | 0.438 | 0.562 | 0.125 | 0.667 | 0.104 | 0.812 | 1.218 |

## Best Config Breakdown

- best config: `max_step=2`, `failure_threshold=0.6`
- small baseline accuracy: `0.438`
- policy accuracy: `0.583`
- gain over small: `0.146`
- oracle accuracy under same budget: `0.656`
- route rate: `0.625`

### By Benchmark

#### olympiadbench

- tasks: `78`
- small baseline accuracy: `0.449`
- policy accuracy: `0.500`
- policy gain over small: `0.051`
- route rate: `0.603`

#### livebench_reasoning

- tasks: `18`
- small baseline accuracy: `0.389`
- policy accuracy: `0.944`
- policy gain over small: `0.556`
- route rate: `0.722`
