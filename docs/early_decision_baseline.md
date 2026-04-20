# Early Decision Baseline

## 設定
- target: `small_full_trace_success`
- row: 每個 prefix 一列
- label: 該題 small model 的 full trace 最後是否答對
- dataset: `OlympiadBench + LiveBench` 共 `2357` 個 prefix rows

輸出：
- [results/_early_decision_v1/early_decision_rows.csv](/Users/powerarena/Documents/GitHub/confidence-tom/results/_early_decision_v1/early_decision_rows.csv)
- [results/_early_decision_v1/dataset_meta.json](/Users/powerarena/Documents/GitHub/confidence-tom/results/_early_decision_v1/dataset_meta.json)
- [results/_early_decision_v1/baseline_results.json](/Users/powerarena/Documents/GitHub/confidence-tom/results/_early_decision_v1/baseline_results.json)

## 整體結果
| Experiment | Test AUROC | Test F1 | 解讀 |
| --- | ---: | ---: | --- |
| `structural_only` | 0.446 | 0.577 | 只靠步數與長度幾乎不夠 |
| `state_signals` | 0.559 | 0.601 | 加入 drift / confidence-like signals 後有改善 |
| `state_plus_family` | **0.592** | 0.587 | 第一版最佳整體 AUROC |
| `state_plus_family_plus_benchmark` | 0.576 | 0.575 | benchmark one-hot 沒再往上推 |
| `olympiad_only_state_plus_family` | 0.504 | 0.595 | OlympiadBench 幾乎接近難預測 |
| `livebench_only_state_plus_family` | **0.880** | **0.739** | LiveBench 上 early decision 很有戲 |
| `train Olympiad -> test LiveBench` | 0.804 | 0.723 | 從 Olympiad 學到的 failure/success signal 可部分轉移 |
| `train LiveBench -> test Olympiad` | 0.545 | 0.541 | 從簡單 domain 轉到複雜 domain 明顯變差 |

## Step-bucket 結果
### `state_plus_family`
- step `1`: AUROC `0.596`
- step `2`: AUROC `0.615`
- step `3`: AUROC `0.676`
- step `4+`: AUROC `0.533`

### `livebench_only_state_plus_family`
- step `1`: AUROC `1.000` (n=18)
- step `2`: AUROC `0.927` (n=16)
- step `3`: AUROC `0.855` (n=16)
- step `4+`: AUROC `0.814` (n=19)

## 初步解讀
1. `Early Decision` 這條線是有訊號的，但強度很 benchmark-dependent。
2. `LiveBench` 很適合先做，因為前 1-2 步就已經有很強的 outcome signal。
3. `OlympiadBench` 難很多，表示長鏈 heterogeneous reasoning 的 early diagnosability 比較弱。
4. `state_plus_family_plus_benchmark` 沒有比 `state_plus_family` 更好，代表這個任務目前更像在學 small-family-specific success geometry，而不是 takeover predictor 那種 benchmark-aware geometry。
5. pooled setting 裡 `step 3` 最好，說明「稍早但不是最早」的 prefix 可能最有資訊。

## 現在可以保守講的話
> Early Decision 在 LiveBench 上很有希望：只看很前面的 prefix，就已經能不錯地預測 small model 這條 reasoning 最後會不會成功。相對地，OlympiadBench 上這件事明顯更難，表示不同 benchmark 的 early diagnosability 也不一樣。
