# Prefix Predictor Failure Analysis

## 核心問題

這份分析不是再追求更高分，而是回答：為什麼目前的 baseline predictor 只顯示有限訊號？

## 1. 類別不平衡與 family heterogeneity

- overall positive rate: `0.177`

| Pair | Count | Positive Rate |
| --- | ---: | ---: |
| `llama->anthropic` | 383 | 0.248 |
| `llama->openai` | 390 | 0.249 |
| `mistral->anthropic` | 219 | 0.132 |
| `mistral->openai` | 229 | 0.175 |
| `qwen->anthropic` | 213 | 0.056 |
| `qwen->openai` | 230 | 0.091 |

解讀：如果不同 family pairing 的正例率差很多，pooled baseline 很容易先學到 family prior，而不是更細的 state signal。

## 2. 單一特徵的分離度

| Feature | Positive Mean | Non-Positive Mean | Effect Size |
| --- | ---: | ---: | ---: |
| `self_correction_cue_density` | 0.001 | 0.001 | 0.287 |
| `prefix_tokens` | 224.687 | 173.407 | 0.253 |
| `prefix_text_tokens` | 224.687 | 173.407 | 0.253 |
| `step_index` | 4.565 | 3.993 | 0.162 |
| `prefix_segments_count` | 4.565 | 3.993 | 0.162 |
| `backtracking_flag` | 0.000 | 0.008 | -0.127 |
| `backtracking_mentions` | 0.000 | 0.008 | -0.127 |
| `commitment_score` | 0.003 | 0.004 | -0.093 |
| `current_segment_tokens` | 42.241 | 40.818 | 0.050 |
| `certainty_density` | 0.004 | 0.004 | -0.021 |

解讀：如果 effect size 普遍不大，就表示單一表面特徵本身對正 gain 的分離度有限。

## 3. 依 small family 分開看

### `llama`

| Feature | Effect Size |
| --- | ---: |
| `self_correction_cue_density` | 0.326 |
| `prefix_tokens` | 0.178 |
| `prefix_text_tokens` | 0.178 |
| `semantic_drift_score` | 0.169 |
| `commitment_score` | -0.144 |
| `backtracking_flag` | -0.118 |
| `backtracking_mentions` | -0.118 |
| `step_index` | 0.070 |

### `mistral`

| Feature | Effect Size |
| --- | ---: |
| `prefix_tokens` | -0.278 |
| `prefix_text_tokens` | -0.278 |
| `step_index` | -0.228 |
| `prefix_segments_count` | -0.228 |
| `semantic_drift_score` | -0.172 |
| `commitment_score` | 0.164 |
| `certainty_density` | 0.148 |
| `backtracking_flag` | -0.103 |

### `qwen`

| Feature | Effect Size |
| --- | ---: |
| `semantic_drift_score` | -0.368 |
| `step_index` | -0.320 |
| `prefix_segments_count` | -0.320 |
| `self_correction_cue_density` | -0.298 |
| `prefix_tokens` | -0.245 |
| `prefix_text_tokens` | -0.245 |
| `backtracking_flag` | -0.157 |
| `backtracking_mentions` | -0.157 |

## 4. Baseline 結果

- `structural_only`: AUROC=0.382, F1=0.135, Precision=0.082, Recall=0.388
- `state_signals`: AUROC=0.411, F1=0.170, Precision=0.110, Recall=0.367
- `state_plus_family`: AUROC=0.600, F1=0.275, Precision=0.172, Recall=0.694

## 暫時結論

1. baseline 弱，不是因為完全沒訊號，而是因為正例本來就稀少。
2. pooled data 有明顯 family heterogeneity，會稀釋單一 signal 的效果。
3. confidence-related features 有 modest 增量，但目前還不足以單獨撐起 predictor。
4. 下一步若要再進一步，最合理的是補更接近 reasoning state 的 fragility signal，或改做 family-conditioned predictor。
