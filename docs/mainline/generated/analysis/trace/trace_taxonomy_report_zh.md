# Trace Taxonomy Report (ZH)

## 核心問題

目前主線問題是：

**小模型先生成完整 Chain-of-Thought 之後，這條推理在中途不同 prefix 是否已經包含足夠資訊，讓模型從該 prefix 重新接著走到正解？**

方法上不是直接中途打斷模型，而是：

1. 先讓小模型生成完整 step-by-step CoT。
2. 把完整 trace 切成不同長度的 prefix。
3. 把 prefix 餵回模型做 re-entry。
4. 檢查從不同 prefix 出發時，小模型能否得到局部正確結果。

## 目前結論

這次分析使用的是從 `192.168.107.10:/home/karl/confidence-tom/results/` 同步回來的結果目錄，並直接以 repo 根目錄下的 `results/` 作為輸入重新跑分析腳本，不是用本機臨時 smoke 輸出。

目前最重要的結論是：

- partial reasoning 的可重用性**不是由 step count 單獨決定**。
- 比起找 universal re-entry step，**trace type 更有解釋力**。
- re-entry 更像是 **competence-conditioned rescue**，不是憑空創造新能力。

## Trace Taxonomy

目前四類 trace 定義為：

- `stable-success`: full trace 正確，而且很早就出現可重用的正確 prefix，之後局部正確率維持高。
- `late-success`: full trace 正確，但正確 prefix 出現得比較晚，或局部正確率不夠穩。
- `late-failure`: 中途曾出現局部正確 prefix，但 final full trace 仍然錯。
- `persistent-failure`: 從頭到尾都沒有形成可重用 prefix，final full trace 也錯。

本次總 task 數：`982`

| Category | Count | Share | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: |
| stable-success | 266 | 0.271 | 0.234 | 0.950 |
| late-success | 141 | 0.144 | 0.727 | 0.589 |
| late-failure | 127 | 0.129 | 0.458 | 0.446 |
| persistent-failure | 448 | 0.456 | 1.000 | 0.000 |

這代表：

- `stable-success` 的確存在，而且通常很早就能看到足夠資訊。
- `late-success` 表示有些題目要到很後面才累積到可用 prefix。
- `late-failure` 是最值得研究的一類：中途曾經有局部正確訊號，但它沒有被完整推理保留下來。
- `persistent-failure` 仍然是最大宗，表示很多題目從頭到尾都沒有形成可重用 prefix。

## Direct Competence vs Re-entry

`direct_vs_reentry` 的結果：

- 若 full trace 本來正確：
  - `any_small_correct = 0.951`
  - `last_small_correct = 0.887`
- 若 full trace 本來錯誤：
  - `any_small_correct = 0.221`
  - `last_small_correct = 0.073`

這支持目前的主要解讀：

> re-entry 幫助最大的情況，是模型原本就已有部分能力與局部正確訊號，而不是從沒有能力的題目中硬生生創造新能力。

## Re-entry Controls

這次重新跑的 `prefix_reentry_controls` 摘要如下：

- rows: `1220`
- re-entry match rate: `0.256`
- re-entry repeat match rate: `0.602`
- P(full-trace success | re-entry match): `0.647`
- P(full-trace success | re-entry mismatch): `0.339`
- P(positive takeover | re-entry match): `0.109`
- P(positive takeover | re-entry mismatch): `0.272`

這表示：

- 如果 re-entry continuation 和原本 small trace 比較一致，full-trace success 機率更高。
- 但 positive takeover 並不是在「match」時最高，說明 re-entry match 更像是 stability 訊號，而不一定等於 takeover opportunity。

## 目前建議定位

這條研究線目前最適合定位成：

> **一個 trace-level diagnosis / taxonomy framework**，
> 用 stepwise re-entry 去分析 partial reasoning 的可重用性結構。

比起找單一最佳 `T`，現在更重要的是：

- 分清不同 trace type 的形成機制
- 理解哪些題目會早期形成可重用 prefix
- 理解為什麼某些題目中途曾經正確，最後卻沒有保住正確訊號
