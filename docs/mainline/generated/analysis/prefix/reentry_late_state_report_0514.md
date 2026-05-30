# Re-entry Late-State Report 0514

這份報告採用修正版口徑：**「救回來」指的是 prepare 做錯，後來 re-entry 做對**。

因此，本報告不再把 `full_rerun_correct=0 -> reentry_exact_correct=1` 稱為救回來。`full_rerun_correct` 只作為「從頭重跑」的輔助診斷；真正的 late-state taxonomy 以 prepare 結果 `full_trace_correct` 與 re-entry 結果 `reentry_exact_correct` 為準。

## 0. Taxonomy 定義

| 圖片名詞 | 本報告使用的定義 | 中文理解 |

|---|---|---|
| Strong Late-Success | `full_trace_correct=1`, `reentry_exact_correct=1` | prepare 已經對，re-entry 也能保留正確 state |
| Edge Late-Success | `full_trace_correct=0`, `reentry_exact_correct=1` | prepare 做錯，但 re-entry 後做對，也就是「救回來」 |
| Strong Late-Failure | `full_trace_correct=1`, `reentry_exact_correct=0` | prepare 原本對，但 re-entry 後錯，也就是「破壞」 |
| Fragile Late-Failure | `full_trace_correct=0`, `reentry_exact_correct=0` | prepare 錯，re-entry 也沒有救回 |

所有 re-entry 統計都排除 `segment_count_outlier=1`，且排除 error rows。

注意：Math500 目前本地資料已經不是早先的 451-row partial snapshot，而是目前 `reentry_rows.jsonl` 中可讀到的 949 個 non-outlier rows。舊版報告裡的 451-row 結論可視為當時 partial snapshot；本版改用目前資料與正確的「救回來」定義。

## 1. Math500

### 1.1 Prepare 正確率

這裡保留原先 task-level prepare snapshot，單位是題目，不是 prefix row。

| model | tasks | prepare_correct | avg_segments | max_segments |
|---|---:|---:|---:|---:|
| qwen | 50 | 44/50 = 88.0% | 25.0 | 229 |
| gemma | 50 | 35/50 = 70.0% | 6.3 | 47 |
| mistral | 50 | 12/50 = 24.0% | 9.1 | 118 |
| olmo | 50 | 7/50 = 14.0% | 6.6 | 25 |

Qwen / Gemma 在 prepare 階段原本就強；Mistral / OLMo 從頭解題較弱。這點會影響「救回來」的解讀：如果某模型 prepare 幾乎都對，例如目前 Math500 的 qwen rows，能被救回的母體本來就很小。

### 1.2 總體 Late-State Taxonomy

單位是 prefix row。

| metric | value |
|---|---:|
| non-outlier rows | 949 |
| prepare correct | 482/949 = 50.8% |
| reentry_exact correct | 545/949 = 57.4% |
| Strong Late-Success | 441 |
| Edge Late-Success / 救回來 | 104/467 = 22.3% |
| Strong Late-Failure / 破壞 | 41/482 = 8.5% |
| Fragile Late-Failure | 363 |
| net rescue | +63 |

修正口徑後，Math500 仍然是正向：prepare 錯的 rows 裡有 22.3% 被 re-entry 救回，而 prepare 對的 rows 裡有 8.5% 被 re-entry 破壞。整體 net rescue 為 +63。

這個結論比舊版「用 full_rerun 當救回母體」更保守，也更符合實驗語意：我們關心的是原本 prepare 的 trajectory 是否能被 prefix re-entry 改寫成正確結果。

### 1.3 Prefix 位置

| prefix | rows | prepare correct | reentry_exact correct | Strong Late-Success | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | Fragile Late-Failure | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p01 | 154 | 67/154 = 43.5% | 78/154 = 50.6% | 57 | 21/87 = 24.1% | 10/67 = 14.9% | 66 | +11 |
| p02 | 158 | 70/158 = 44.3% | 87/158 = 55.1% | 64 | 23/88 = 26.1% | 6/70 = 8.6% | 65 | +17 |
| p03 | 146 | 61/146 = 41.8% | 78/146 = 53.4% | 58 | 20/85 = 23.5% | 3/61 = 4.9% | 65 | +17 |
| p04-p05 | 236 | 99/236 = 41.9% | 130/236 = 55.1% | 96 | 34/137 = 24.8% | 3/99 = 3.0% | 103 | +31 |
| p06-p10 | 120 | 50/120 = 41.7% | 54/120 = 45.0% | 48 | 6/70 = 8.6% | 2/50 = 4.0% | 64 | +4 |
| p11-p20 | 44 | 44/44 = 100.0% | 39/44 = 88.6% | 39 | 0/0 = n/a | 5/44 = 11.4% | 0 | -5 |
| p21-p50 | 79 | 79/79 = 100.0% | 68/79 = 86.1% | 68 | 0/0 = n/a | 11/79 = 13.9% | 0 | -11 |
| p51+ | 12 | 12/12 = 100.0% | 11/12 = 91.7% | 11 | 0/0 = n/a | 1/12 = 8.3% | 0 | -1 |

修正後，Math500 的 late-prefix 解讀也要改：p11 之後 prepare rows 幾乎全部已經是 correct，因此「救回來」母體消失，不能再說 late prefix 大量 rescue。更準確的說法是：

- early / middle prefix 有實際 rescue 空間，尤其 p01-p05；
- late prefix 更像 state preservation 測試，主要看 Strong Late-Success 是否能維持，以及 Strong Late-Failure 是否出現；
- p11 之後 re-entry 仍大多正確，但會有少量破壞。

### 1.4 分模型比較

| model | rows | prepare correct | reentry_exact correct | Strong Late-Success | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | Fragile Late-Failure | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma | 244 | 172/244 = 70.5% | 194/244 = 79.5% | 160 | 34/72 = 47.2% | 12/172 = 7.0% | 38 | +22 |
| mistral | 254 | 60/254 = 23.6% | 88/254 = 34.6% | 59 | 29/194 = 14.9% | 1/60 = 1.7% | 165 | +28 |
| olmo | 219 | 25/219 = 11.4% | 64/219 = 29.2% | 25 | 39/194 = 20.1% | 0/25 = 0.0% | 155 | +39 |
| qwen | 232 | 225/232 = 97.0% | 199/232 = 85.8% | 197 | 2/7 = 28.6% | 28/225 = 12.4% | 5 | -26 |

分模型後，原本的直覺需要修正：

- Gemma 是最乾淨的正向模型：prepare 已經不弱，仍能救回 47.2% 的 prepare-wrong rows，破壞率只有 7.0%。
- OLMo 和 Mistral 的 prepare 很弱，因此有較大的救援母體。OLMo 的 net rescue 最高，表示 prefix scaffold 確實能把一部分原本錯的 trajectory 拉回來。
- Qwen 在目前 rows 裡 prepare 幾乎都對，所以「救回來」不是主要現象；它的主要風險反而是 re-entry 破壞已經正確的 prepare state。

### 1.5 Prefix × Model

這張表把 prefix 位置和 model 放在一起看。它可以避免一個誤讀：總體 prefix 趨勢常常是模型組成造成的。例如 Math500 的 p11+ 幾乎都是 qwen rows，而 qwen 的 prepare 幾乎全對，所以晚期 prefix 沒有「救回來」母體，只剩 state preservation / damage 問題。

| model | prefix | rows | prepare correct | reentry_exact correct | 救回來 | 破壞 | net |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma | p01 | 47 | 33/47 = 70.2% | 35/47 = 74.5% | 8/14 = 57.1% | 6/33 = 18.2% | +2 |
| gemma | p02 | 47 | 33/47 = 70.2% | 36/47 = 76.6% | 7/14 = 50.0% | 4/33 = 12.1% | +3 |
| gemma | p03 | 47 | 33/47 = 70.2% | 40/47 = 85.1% | 8/14 = 57.1% | 1/33 = 3.0% | +7 |
| gemma | p04-p05 | 75 | 53/75 = 70.7% | 62/75 = 82.7% | 10/22 = 45.5% | 1/53 = 1.9% | +9 |
| gemma | p06-p10 | 28 | 20/28 = 71.4% | 21/28 = 75.0% | 1/8 = 12.5% | 0/20 = 0.0% | +1 |
| mistral | p01 | 47 | 12/47 = 25.5% | 19/47 = 40.4% | 7/35 = 20.0% | 0/12 = 0.0% | +7 |
| mistral | p02 | 47 | 12/47 = 25.5% | 19/47 = 40.4% | 7/35 = 20.0% | 0/12 = 0.0% | +7 |
| mistral | p03 | 47 | 12/47 = 25.5% | 16/47 = 34.0% | 4/35 = 11.4% | 0/12 = 0.0% | +4 |
| mistral | p04-p05 | 76 | 18/76 = 23.7% | 26/76 = 34.2% | 8/58 = 13.8% | 0/18 = 0.0% | +8 |
| mistral | p06-p10 | 37 | 6/37 = 16.2% | 8/37 = 21.6% | 3/31 = 9.7% | 1/6 = 16.7% | +2 |
| olmo | p01 | 44 | 7/44 = 15.9% | 13/44 = 29.5% | 6/37 = 16.2% | 0/7 = 0.0% | +6 |
| olmo | p02 | 44 | 7/44 = 15.9% | 15/44 = 34.1% | 8/37 = 21.6% | 0/7 = 0.0% | +8 |
| olmo | p03 | 40 | 5/40 = 12.5% | 13/40 = 32.5% | 8/35 = 22.9% | 0/5 = 0.0% | +8 |
| olmo | p04-p05 | 59 | 5/59 = 8.5% | 20/59 = 33.9% | 15/54 = 27.8% | 0/5 = 0.0% | +15 |
| olmo | p06-p10 | 32 | 1/32 = 3.1% | 3/32 = 9.4% | 2/31 = 6.5% | 0/1 = 0.0% | +2 |
| qwen | p01 | 16 | 15/16 = 93.8% | 11/16 = 68.8% | 0/1 = 0.0% | 4/15 = 26.7% | -4 |
| qwen | p02 | 20 | 18/20 = 90.0% | 17/20 = 85.0% | 1/2 = 50.0% | 2/18 = 11.1% | -1 |
| qwen | p03 | 12 | 11/12 = 91.7% | 9/12 = 75.0% | 0/1 = 0.0% | 2/11 = 18.2% | -2 |
| qwen | p04-p05 | 26 | 23/26 = 88.5% | 22/26 = 84.6% | 1/3 = 33.3% | 2/23 = 8.7% | -1 |
| qwen | p06-p10 | 23 | 23/23 = 100.0% | 22/23 = 95.7% | 0/0 = n/a | 1/23 = 4.3% | -1 |
| qwen | p11-p20 | 44 | 44/44 = 100.0% | 39/44 = 88.6% | 0/0 = n/a | 5/44 = 11.4% | -5 |
| qwen | p21-p50 | 79 | 79/79 = 100.0% | 68/79 = 86.1% | 0/0 = n/a | 11/79 = 13.9% | -11 |
| qwen | p51+ | 12 | 12/12 = 100.0% | 11/12 = 91.7% | 0/0 = n/a | 1/12 = 8.3% | -1 |

Math500 的 prefix × model 圖像：

- Gemma 的救回來主要集中在 p01-p05，且破壞率快速下降。
- Mistral / OLMo 的 prepare 較弱，所以有較大救援空間；OLMo 在 p04-p05 的 scaffold 效果最明顯。
- Qwen 的 prepare 幾乎全對，因此晚期 prefix 沒有 rescue 母體；它的問題是 re-entry 會破壞一部分已經正確的 state。

## 2. LiveBench Reasoning

### 2.1 Prepare 正確率

單位是題目，不是 prefix row。

| model | tasks | prepare_correct | avg_segments | max_segments |
|---|---:|---:|---:|---:|
| qwen3 | 30 | 13/30 = 43.3% | 18.7 | 243 |
| qwen25 | 30 | 9/30 = 30.0% | 5.9 | 18 |
| gemma4 | 30 | 18/30 = 60.0% | 15.4 | 61 |
| ministral | 30 | 4/30 = 13.3% | 13.0 | 200 |
| mistral7 | 30 | 1/30 = 3.3% | 7.4 | 20 |

LiveBench 的 prepare 分布和 Math500 不同。Gemma4 是最強的 prepare full-trace model；Qwen3 trajectory 較長，但 task-level correctness 比 Gemma4 低；Ministral / Mistral7 從頭解題很弱。

### 2.2 總體 Late-State Taxonomy

| metric | value |
|---|---:|
| non-outlier rows | 1167 |
| prepare correct | 452/1167 = 38.7% |
| reentry_exact correct | 336/1167 = 28.8% |
| Strong Late-Success | 275 |
| Edge Late-Success / 救回來 | 61/715 = 8.5% |
| Strong Late-Failure / 破壞 | 177/452 = 39.2% |
| Fragile Late-Failure | 654 |
| net rescue | -116 |

LiveBench 在修正口徑後是負向：re-entry 救回 prepare-wrong 的比例只有 8.5%，但會破壞 prepare-correct 的比例高達 39.2%。這表示 LiveBench 的 prefix continuation 不是穩定 rescue 機制，反而常無法保留 prepare 已經形成的正確 state。

### 2.3 Prefix 位置

| prefix | rows | prepare correct | reentry_exact correct | Strong Late-Success | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | Fragile Late-Failure | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p01 | 141 | 44/141 = 31.2% | 28/141 = 19.9% | 19 | 9/97 = 9.3% | 25/44 = 56.8% | 88 | -16 |
| p02 | 138 | 44/138 = 31.9% | 25/138 = 18.1% | 16 | 9/94 = 9.6% | 28/44 = 63.6% | 85 | -19 |
| p03 | 138 | 44/138 = 31.9% | 33/138 = 23.9% | 25 | 8/94 = 8.5% | 19/44 = 43.2% | 86 | -11 |
| p04-p05 | 232 | 80/232 = 34.5% | 61/232 = 26.3% | 45 | 16/152 = 10.5% | 35/80 = 43.8% | 136 | -19 |
| p06-p10 | 320 | 120/320 = 37.5% | 91/320 = 28.4% | 74 | 17/200 = 8.5% | 46/120 = 38.3% | 183 | -29 |
| p11-p20 | 167 | 99/167 = 59.3% | 79/167 = 47.3% | 77 | 2/68 = 2.9% | 22/99 = 22.2% | 66 | -20 |
| p21-p50 | 31 | 21/31 = 67.7% | 19/31 = 61.3% | 19 | 0/10 = 0.0% | 2/21 = 9.5% | 10 | -2 |

LiveBench 的 prefix 位置圖像是：越晚的 prefix 越能保留 prepare-correct state，因此破壞率下降；但救回率沒有同步上升。這和舊版「late prefix rescue 變強」不同。修正後更合理的解讀是：LiveBench late prefix 主要改善 state preservation，而不是 rescue。

### 2.4 分模型比較

| model | rows | prepare correct | reentry_exact correct | Strong Late-Success | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | Fragile Late-Failure | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 236 | 117/236 = 49.6% | 93/236 = 39.4% | 74 | 19/119 = 16.0% | 43/117 = 36.8% | 100 | -24 |
| qwen25 | 138 | 45/138 = 32.6% | 37/138 = 26.8% | 27 | 10/93 = 10.8% | 18/45 = 40.0% | 83 | -8 |
| gemma4 | 401 | 257/401 = 64.1% | 172/401 = 42.9% | 160 | 12/144 = 8.3% | 97/257 = 37.7% | 132 | -85 |
| ministral | 190 | 23/190 = 12.1% | 20/190 = 10.5% | 13 | 7/167 = 4.2% | 10/23 = 43.5% | 160 | -3 |
| mistral7 | 202 | 10/202 = 5.0% | 14/202 = 6.9% | 1 | 13/192 = 6.8% | 9/10 = 90.0% | 179 | +4 |

LiveBench 分模型後的重點是：

- Qwen3 是相對最好的 rescue model，但 net 仍為負，因為破壞 prepare-correct 的 rows 太多。
- Gemma4 prepare 很強，但 re-entry 反而破壞大量 prepare-correct rows；因此它不是 rescue model，而是 state preservation 失敗最明顯的模型之一。
- Mistral7 唯一 net 略正，但原因是 prepare-correct rows 很少，damage 母體太小；不是因為它真的有很強救援能力。

### 2.5 Prefix × Model

LiveBench 的 prefix × model 表顯示：總體負向不是單一模型造成，而是多數模型都同時存在低 rescue 與高 damage。晚期 prefix 的確降低 damage，但救回來並沒有明顯變強。

| model | prefix | rows | prepare correct | reentry_exact correct | 救回來 | 破壞 | net |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | p01 | 27 | 12/27 = 44.4% | 8/27 = 29.6% | 2/15 = 13.3% | 6/12 = 50.0% | -4 |
| qwen3 | p02 | 27 | 12/27 = 44.4% | 9/27 = 33.3% | 3/15 = 20.0% | 6/12 = 50.0% | -3 |
| qwen3 | p03 | 27 | 12/27 = 44.4% | 9/27 = 33.3% | 2/15 = 13.3% | 5/12 = 41.7% | -3 |
| qwen3 | p04-p05 | 46 | 21/46 = 45.7% | 16/46 = 34.8% | 3/25 = 12.0% | 8/21 = 38.1% | -5 |
| qwen3 | p06-p10 | 72 | 33/72 = 45.8% | 29/72 = 40.3% | 8/39 = 20.5% | 12/33 = 36.4% | -4 |
| qwen3 | p11-p20 | 37 | 27/37 = 73.0% | 22/37 = 59.5% | 1/10 = 10.0% | 6/27 = 22.2% | -5 |
| qwen25 | p01 | 27 | 9/27 = 33.3% | 9/27 = 33.3% | 3/18 = 16.7% | 3/9 = 33.3% | +0 |
| qwen25 | p02 | 27 | 9/27 = 33.3% | 8/27 = 29.6% | 3/18 = 16.7% | 4/9 = 44.4% | -1 |
| qwen25 | p03 | 27 | 9/27 = 33.3% | 5/27 = 18.5% | 0/18 = 0.0% | 4/9 = 44.4% | -4 |
| qwen25 | p04-p05 | 39 | 14/39 = 35.9% | 13/39 = 33.3% | 4/25 = 16.0% | 5/14 = 35.7% | -1 |
| qwen25 | p06-p10 | 18 | 4/18 = 22.2% | 2/18 = 11.1% | 0/14 = 0.0% | 2/4 = 50.0% | -2 |
| gemma4 | p01 | 29 | 18/29 = 62.1% | 7/29 = 24.1% | 1/11 = 9.1% | 12/18 = 66.7% | -11 |
| gemma4 | p02 | 29 | 18/29 = 62.1% | 6/29 = 20.7% | 1/11 = 9.1% | 13/18 = 72.2% | -12 |
| gemma4 | p03 | 29 | 18/29 = 62.1% | 11/29 = 37.9% | 1/11 = 9.1% | 8/18 = 44.4% | -7 |
| gemma4 | p04-p05 | 56 | 36/56 = 64.3% | 18/56 = 32.1% | 1/20 = 5.0% | 19/36 = 52.8% | -18 |
| gemma4 | p06-p10 | 119 | 74/119 = 62.2% | 54/119 = 45.4% | 7/45 = 15.6% | 27/74 = 36.5% | -20 |
| gemma4 | p11-p20 | 108 | 72/108 = 66.7% | 57/108 = 52.8% | 1/36 = 2.8% | 16/72 = 22.2% | -15 |
| gemma4 | p21-p50 | 31 | 21/31 = 67.7% | 19/31 = 61.3% | 0/10 = 0.0% | 2/21 = 9.5% | -2 |
| ministral | p01 | 29 | 4/29 = 13.8% | 1/29 = 3.4% | 0/25 = 0.0% | 3/4 = 75.0% | -3 |
| ministral | p02 | 26 | 4/26 = 15.4% | 0/26 = 0.0% | 0/22 = 0.0% | 4/4 = 100.0% | -4 |
| ministral | p03 | 26 | 4/26 = 15.4% | 5/26 = 19.2% | 2/22 = 9.1% | 1/4 = 25.0% | +1 |
| ministral | p04-p05 | 44 | 7/44 = 15.9% | 9/44 = 20.5% | 3/37 = 8.1% | 1/7 = 14.3% | +2 |
| ministral | p06-p10 | 59 | 4/59 = 6.8% | 5/59 = 8.5% | 2/55 = 3.6% | 1/4 = 25.0% | +1 |
| ministral | p11-p20 | 6 | 0/6 = 0.0% | 0/6 = 0.0% | 0/6 = 0.0% | 0/0 = n/a | +0 |
| mistral7 | p01 | 29 | 1/29 = 3.4% | 3/29 = 10.3% | 3/28 = 10.7% | 1/1 = 100.0% | +2 |
| mistral7 | p02 | 29 | 1/29 = 3.4% | 2/29 = 6.9% | 2/28 = 7.1% | 1/1 = 100.0% | +1 |
| mistral7 | p03 | 29 | 1/29 = 3.4% | 3/29 = 10.3% | 3/28 = 10.7% | 1/1 = 100.0% | +2 |
| mistral7 | p04-p05 | 47 | 2/47 = 4.3% | 5/47 = 10.6% | 5/45 = 11.1% | 2/2 = 100.0% | +3 |
| mistral7 | p06-p10 | 52 | 5/52 = 9.6% | 1/52 = 1.9% | 0/47 = 0.0% | 4/5 = 80.0% | -4 |
| mistral7 | p11-p20 | 16 | 0/16 = 0.0% | 0/16 = 0.0% | 0/16 = 0.0% | 0/0 = n/a | +0 |

LiveBench 的 prefix × model 圖像：

- Qwen3 的 damage 會隨 prefix 變晚而下降，但救回來比例沒有明顯增加。
- Qwen25 全段偏負，p03 和 p06-p10 特別弱。
- Gemma4 prepare 很強，但 early / middle prefix damage 很高；p21-p50 damage 降到 9.5%，比較像 late state preservation 開始恢復。
- Ministral / Mistral7 的 prepare-correct 母體很小，表面 net 有時為正，但不能解讀成強 rescue；多數 rows 仍是 prepare 錯、re-entry 也錯。

## 3. 跨 Benchmark 比較

| benchmark | rows | prepare correct | reentry_exact correct | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | net rescue |
|---|---:|---:|---:|---:|---:|---:|
| Math500 | 949 | 482/949 = 50.8% | 545/949 = 57.4% | 104/467 = 22.3% | 41/482 = 8.5% | +63 |
| LiveBench reasoning | 1167 | 452/1167 = 38.7% | 336/1167 = 28.8% | 61/715 = 8.5% | 177/452 = 39.2% | -116 |

修正「救回來」定義後，兩個 benchmark 的差異非常清楚：

- Math500：re-entry 有真實 rescue 效果，且 damage 相對低。
- LiveBench：re-entry 整體不是 rescue，而是經常破壞 prepare 已經正確的 state。

這表示 re-entry 的價值不是「給 prefix 就會變好」，而是取決於 benchmark 是否有可保留、可延續的中間 reasoning state。

## 4. 分模型的跨 Benchmark 討論

### 4.1 Qwen 系列

| benchmark | model | rows | prepare correct | reentry_exact correct | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | net |
|---|---|---:|---:|---:|---:|---:|---:|
| Math500 | qwen | 232 | 225/232 = 97.0% | 199/232 = 85.8% | 2/7 = 28.6% | 28/225 = 12.4% | -26 |
| LiveBench | qwen3 | 236 | 117/236 = 49.6% | 93/236 = 39.4% | 19/119 = 16.0% | 43/117 = 36.8% | -24 |
| LiveBench | qwen25 | 138 | 45/138 = 32.6% | 37/138 = 26.8% | 10/93 = 10.8% | 18/45 = 40.0% | -8 |

Qwen 在 Math500 的 prepare 已經幾乎全對，因此它不是 rescue 主角；它的問題反而是 re-entry 會破壞部分已正確 state。LiveBench 上 Qwen3 / Qwen25 都有 rescue，但 damage 更高，所以 net 仍為負。

### 4.2 Gemma 系列

| benchmark | model | rows | prepare correct | reentry_exact correct | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | net |
|---|---|---:|---:|---:|---:|---:|---:|
| Math500 | gemma | 244 | 172/244 = 70.5% | 194/244 = 79.5% | 34/72 = 47.2% | 12/172 = 7.0% | +22 |
| LiveBench | gemma4 | 401 | 257/401 = 64.1% | 172/401 = 42.9% | 12/144 = 8.3% | 97/257 = 37.7% | -85 |

Gemma 是最能說明 benchmark 差異的模型。Math500 上，Gemma 有高 rescue、低 damage；LiveBench 上，Gemma4 prepare 很強，但 re-entry 大量破壞 prepare-correct rows。這表示 Gemma 的 prefix state 在數學題上可延續，在 LiveBench reasoning 上則不穩。

### 4.3 Mistral / Ministral 系列

| benchmark | model | rows | prepare correct | reentry_exact correct | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | net |
|---|---|---:|---:|---:|---:|---:|---:|
| Math500 | mistral | 254 | 60/254 = 23.6% | 88/254 = 34.6% | 29/194 = 14.9% | 1/60 = 1.7% | +28 |
| LiveBench | ministral | 190 | 23/190 = 12.1% | 20/190 = 10.5% | 7/167 = 4.2% | 10/23 = 43.5% | -3 |
| LiveBench | mistral7 | 202 | 10/202 = 5.0% | 14/202 = 6.9% | 13/192 = 6.8% | 9/10 = 90.0% | +4 |

Mistral 在 Math500 有一些 rescue，而且 damage 很低；但 LiveBench 上 rescue 很弱，且 prepare-correct rows 很容易被破壞。Mistral7 的 net 略正主要是因為 prepare-correct 母體很小，不能解讀成強 re-entry model。

### 4.4 OLMo

| benchmark | model | rows | prepare correct | reentry_exact correct | Edge Late-Success / 救回來 | Strong Late-Failure / 破壞 | net |
|---|---|---:|---:|---:|---:|---:|---:|
| Math500 | olmo | 219 | 25/219 = 11.4% | 64/219 = 29.2% | 39/194 = 20.1% | 0/25 = 0.0% | +39 |

OLMo 在 Math500 是典型 scaffold effect：prepare 本身很弱，但 prefix re-entry 可以救回一部分 prepare-wrong rows，而且幾乎沒有破壞。由於目前缺 LiveBench OLMo no-outlier 對照，不能判斷這個 scaffold effect 是否跨 benchmark 穩定。

## 5. 主要結論

修正後的核心結論是：

> 「救回來」必須以 prepare 錯、re-entry 對為準。用這個口徑看，Math500 有明確 rescue 效果；LiveBench 則主要暴露 state preservation failure。

更細地說：

- Math500 的 re-entry 是正向：救回率 22.3%，破壞率 8.5%，net +63。
- LiveBench 的 re-entry 是負向：救回率只有 8.5%，破壞率 39.2%，net -116。
- Math500 的主要價值是 rescue prepare-wrong rows。
- LiveBench 的主要問題是 re-entry 無法保留 prepare-correct state。
- 分模型看，Gemma / OLMo 在 Math500 有明顯 rescue；Qwen 在 Math500 prepare 太強，反而主要看 damage；LiveBench 多數模型都受 state preservation failure 影響。

短版：

> Re-entry 不是單純「prefix 越晚越救」。真正的問題是：prepare 錯的 state 能不能被救回，以及 prepare 對的 state 會不會被破壞。Math500 偏向前者，LiveBench 明顯暴露後者。
