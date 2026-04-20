# Qwen-3 規模效應與 Manager 觀察初步報告

## 一、摘要

這份報告整理了目前兩個相連的實驗：

1. **Qwen-3 family 在困難 open-ended benchmark 上的 worker scaling 結果**
2. **頂尖大模型作為 manager / observer 時，對 worker 推理品質的判斷結果**

目前結果已經支持一個很明確的初步結論：

- 在困難推理任務上，`Qwen-3-8B` 的表現遠低於 `Qwen-3-14B` 與 `Qwen-3-32B`
- 但這個 scaling pattern 並不是平滑上升，而更像：
  - `Qwen-3-8B << Qwen-3-14B ≈ Qwen-3-32B`
- 更重要的是，當 worker 變強之後，即使它仍然答錯，manager 也會更容易相信它

也就是說，較大的 worker 不只整體更強，它的錯誤也更具有欺騙性。

---

## 二、Worker 實驗設定

### 模型

- `Qwen-3-8B`
- `Qwen-3-14B`
- `Qwen-3-32B`

### Benchmark

- `OlympiadBench`
- `LiveBench reasoning`

### 題數

- `OlympiadBench`：`50` 題
- `LiveBench reasoning`：`50` 題
- 每個模型共 `100` 題

### Sampling 設定

- `k = 10`
- `temperature = 1.0`

### 重要解碼設定

本批結果使用 **關閉 thinking mode** 的條件：

- `generator.reasoning_effort=none`
- `enable_thinking=false`

這一點很重要，因為先前的預設 thinking 模式常出現：

- final content 為空
- 隱藏 reasoning 過長
- parsing 不穩

因此，這份報告描述的是 **think-off 條件下** 的結果。

### Parsing 註記

本批主結果中：

- `llm_extract = 0`

也就是這批 worker 主結果沒有依賴 LLM fallback parsing，都是直接由原始輸出解析得到。

---

## 三、Worker 結果

### 整體正確率

| 模型 | 正確率 |
| --- | ---: |
| Qwen-3-8B | `14/100 = 14.0%` |
| Qwen-3-14B | `82/100 = 82.0%` |
| Qwen-3-32B | `82/100 = 82.0%` |

### 分 benchmark 結果

| 模型 | LiveBench reasoning | OlympiadBench |
| --- | ---: | ---: |
| Qwen-3-8B | `14.0%` | `14.0%` |
| Qwen-3-14B | `86.0%` | `78.0%` |
| Qwen-3-32B | `84.0%` | `80.0%` |

### 初步解讀

目前最穩定的 pattern 是：

- `Qwen-3-8B` 明顯遠弱於另外兩個模型
- `Qwen-3-14B` 與 `Qwen-3-32B` 都已經很強
- `Qwen-3-32B` 在這組設定下，沒有明顯超過 `Qwen-3-14B`

因此目前這組資料最適合描述成：

- `Qwen-3-8B << Qwen-3-14B ≈ Qwen-3-32B`

也就是存在明顯 scale effect，但更像是一個 **threshold effect**，不是平滑線性成長。

---

## 四、Confidence 指標說明

本實驗目前區分三個相關量：

- `C_rep`：模型**自報信心**
- `C_beh`：模型**行為正確率**
- `C_consistency`：模型**答案一致性**

### 重要定義說明

在修正後的定義中：

- `C_rep` = `k=10` 次 sample 的平均自報信心
- `C_beh` = `k=10` 次中，**答對的比例**
- `C_consistency` = `k=10` 次中，**多數答案所占比例**

也就是說：

- `C_beh` 反映的是「對正解的行為信心」
- `C_consistency` 反映的是「模型對自己最常輸出的答案有多穩」

先前版本的 `single/static` pipeline 曾將「多數答案比例」記為 `C_beh`。這一點現在已更正；因此若需要最終定稿版，相關數值應以重算後的版本為準。

### 整體 confidence 統計

| 模型 | `C_beh`（答對比例） | `C_consistency`（多數答案比例） | `C_rep` | Gap (`C_rep - C_beh`) |
| --- | ---: | ---: | ---: | ---: |
| Qwen-3-8B | `0.481` | `0.481` | `0.704` | `+0.223` |
| Qwen-3-14B | `0.719` | `0.719` | `0.862` | `+0.143` |
| Qwen-3-32B | `0.667` | `0.663` | `0.896` | `+0.229` |

### 解讀

- 依照修正後定義，三個模型仍然都有 overconfidence 傾向
- `Qwen-3-14B` 的 gap 看起來最小
- `Qwen-3-32B` 雖然 accuracy 與 `14B` 幾乎相同，但自信更高，也更 overconfident

因此，如果同時看能力與 calibration，`Qwen-3-14B` 目前反而是最平衡的一個點。

---

## 五、為什麼 k=10 有價值

`k=10` 的主要價值，是揭露單次無法看到的穩定性差異。

我們直接看每題 10 次自報信心的波動幅度：

| 模型 | 平均 confidence spread | `spread >= 0.3` 的題數 |
| --- | ---: | ---: |
| Qwen-3-8B | `0.438` | `84/100` |
| Qwen-3-14B | `0.261` | `42/100` |
| Qwen-3-32B | `0.248` | `32/100` |

### 解讀

- `Qwen-3-8B` 不只答得差，它的信心本身也非常不穩
- `Qwen-3-14B` 與 `Qwen-3-32B` 顯著更穩定

因此 `k=10` 並不是單純增加成本，而是真的揭露了：

- 模型內部穩定性
- `C_beh` 與 `C_consistency` 的差異
- 自報信心的可靠程度

---

## 六、Manager / Observer 實驗設定

在 worker 結果之上，我們進一步用三個頂尖模型作為 manager / observer，去判斷 worker 的作答是否正確。

### Observer 模型

- `openai/gpt-5.4`
- `anthropic/claude-opus-4.6`
- `google/gemini-3.1-pro-preview`

### 被評估的 worker

- `Qwen-3-8B`
- `Qwen-3-14B`
- `Qwen-3-32B`

### 總評估量

- `3` 個 worker
- `3` 個 observer
- 每個 worker `100` 題
- 共 `900` 筆 manager judgment

### Manager 輸出欄位

每個 manager 會輸出：

- `predicted_correctness`
- `predicted_worker_confidence`
- `predicted_error_type`
- `manager_self_confidence`

observer pipeline 已經完成 smoke test，並通過正式跑批。

---

## 七、Manager 整體結果

### 依 observer 整體比較

| Observer | MAE | Bias | Correlation | Wrong-Answer Trust | Right-Answer Trust |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemini_3_1_pro_preview` | **0.0830** | `+0.0517` | **0.8374** | `0.1656` | **0.9736** |
| `claude_opus_4_6` | `0.1392` | `+0.0342` | `0.8140` | `0.2132` | `0.9116` |
| `gpt_5_4` | `0.2143` | `-0.0869` | `0.6385` | `0.1566` | `0.7462` |

### 解讀

- `Gemini-3.1-Pro` 是目前這批裡最好的 manager
  - MAE 最低
  - correlation 最高
  - 對正確答案給分也最高
- `Claude-Opus-4.6` 次佳
- `GPT-5.4` 在這組設定下最弱
  - correlation 最低
  - 對正確答案信任也偏低
  - 整體較保守

---

## 八、Manager 對錯題的信任：隨 worker 規模上升而增加

這是目前 manager 實驗最重要的結果。

### Claude-Opus-4.6

| Worker | Wrong-Answer Trust |
| --- | ---: |
| Qwen-3-8B | `0.0972` |
| Qwen-3-14B | `0.4878` |
| Qwen-3-32B | `0.4928` |

### Gemini-3.1-Pro

| Worker | Wrong-Answer Trust |
| --- | ---: |
| Qwen-3-8B | `0.0395` |
| Qwen-3-14B | `0.5111` |
| Qwen-3-32B | `0.4222` |

### GPT-5.4

| Worker | Wrong-Answer Trust |
| --- | ---: |
| Qwen-3-8B | `0.0736` |
| Qwen-3-14B | `0.3817` |
| Qwen-3-32B | `0.3283` |

### 核心解讀

三個 observer 全都出現同樣的現象：

- `Qwen-3-8B` 的錯題很容易被看穿
- 但當 worker 換成 `Qwen-3-14B` 或 `Qwen-3-32B`，即使答案是錯的，manager 仍然會明顯提高對其正確性的估計

這是一個很強的訊號。

它表示：

> manager 並不只是根據答案對錯做判斷，它也會被 worker 推理的表面品質、流暢度、以及“看起來像專家”的程度影響。

也就是說，較強 worker 的殘餘錯誤，比小模型的錯誤更有欺騙性。

---

## 九、Manager 的錯誤型態

### Error type 分布

- `Claude-Opus-4.6`
  - `None`: `190`
  - `Logic_Error`: `107`
  - `Observation_Ignored`: `3`

- `Gemini-3.1-Pro`
  - `None`: `182`
  - `Logic_Error`: `105`
  - 另外少量其他細分類別

- `GPT-5.4`
  - `Logic_Error`: `161`
  - `None`: `133`
  - `Observation_Ignored`: `6`

### 解讀

- `GPT-5.4` 比較常直接標成 `Logic_Error`
- `Claude` 與 `Gemini` 更常在較強 worker 上標成 `None`

這也符合前面的量化結果：

- `GPT-5.4` 的風格較挑錯、較保守
- `Claude` 與 `Gemini` 尤其在面對強 worker 時，更容易認為「推理大致合理」

---

## 十、目前最穩的初步結論

### Worker 端

1. `OlympiadBench + LiveBench reasoning` 對 `Qwen-3` family 有很強的區分能力。
2. `Qwen-3-8B` 與 `Qwen-3-14B/32B` 之間存在巨大能力差距。
3. 但 `Qwen-3-14B` 與 `Qwen-3-32B` 之間，accuracy 幾乎持平。
4. `k=10` 有價值，因為它揭露了模型在答案穩定性與信心穩定性上的差異。
5. `Qwen-3-14B` 目前是能力與 calibration 最平衡的點。

### Manager 端

1. `Gemini-3.1-Pro` 是目前這批最好的 observer。
2. `GPT-5.4` 在這組設定下是最弱的 observer。
3. 最重要的發現是：
   - **錯誤答案的可欺騙性，會隨著 worker 規模變大而明顯上升。**

換句話說：

> 更強的 worker 不只更常答對；當它答錯時，它的錯誤也更容易騙過上層 manager。

這個現象非常符合原始研究問題，也很值得作為後續論文主軸。

---

## 十一、目前可以安全主張的說法

根據目前資料，可以提出以下初步主張：

> 在困難 open-ended 推理任務上，Qwen-3 family 從 8B 擴展到 14B 時，worker 能力會大幅提升；然而，這種能力提升也使得殘餘錯誤更具欺騙性。無論是 GPT-5.4、Claude-Opus-4.6 或 Gemini-3.1-Pro，對於較強 worker 的錯題都會給出顯著更高的正確率估計。

這是目前最值得保留、也最穩的核心結果。

---

## 十二、注意事項

1. 本批 worker 結果是在 **think-off** 條件下得到的，不是預設 thinking mode。
2. `single/static` pipeline 的 `C_beh` 定義已修正為「`k` 次中答對比例」，並新增 `C_consistency` 表示答案一致性；舊版報表若仍引用 `C_beh = 多數答案比例`，需以重算版取代。
3. 本批 benchmark 為：
   - `OlympiadBench`
   - `LiveBench reasoning`
4. 雖然這組結果已經明顯強於 pilot，但整體仍屬於初步正式實驗。

---

## 十三、下一步

1. 做 manager false positive 的案例分析
   - 尤其是 `Qwen-3-14B/32B` 錯題但被高估的案例
2. 比較不同 manager 是否被同樣類型的 trace 騙過
3. 延伸同樣流程到 `Qwen-3.5`
4. 決定論文的主軸要放在：
   - worker scaling
   - calibration / stability
   - manager 的可欺騙性
   - 或三者結合
