# Current Caveats And Fix Plan

## 一、目的

這份文件整理目前 prefix-based intervention 線的主要 caveats，以及各自最務實的修法。目標不是一次消滅所有限制，而是把：

- 哪些問題已經足夠小，可以先往外擴
- 哪些問題會直接污染結論，必須先修
- 哪些問題應該靠方法定義或實驗擴張來處理

講清楚。

---

## 二、目前已經站穩的部分

目前這條線已經可以合理主張：

1. `full generation -> parser-based segmentation -> prefix probing` 是可跑的。
2. 在清理主要 label purity 問題後，仍能觀察到 early-prefix takeover gain。
3. `Δ_t` 不是純噪音；不同題目會呈現不同 gain curve。

目前乾淨版 `limit=10` 的摘要為：

- tasks: `10`
- prefix steps: `45`
- positive gain: `11`
- zero gain: `34`
- negative gain: `0`
- tasks with any positive gain: `5/10`

而且 `avg Δ_t vs step` 仍呈現下降趨勢：

- step 1: `0.50`
- step 2: `0.30`
- step 3: `0.20`
- step 4: `0.11`
- step 5: `0.00`

所以目前最穩的敘事是：

> Early prefixes are more likely to benefit from takeover, and this gain decays as the prefix accumulates more structure.

---

## 三、目前最重要的 caveats

### 3.1 Label Purity

這是目前最重要的一類問題，因為它會直接污染 `Δ_t`。

已經修過的：

- `\\boxed{...}` / `\\fbox{...}` normalization
- `\\dfrac` / `\\tfrac` normalization
- final answer extraction 收緊
- parser fallback 可直接抽 final answer，不只做 segmentation

目前殘留風險：

- 仍可能有某些數學格式沒有完全正規化
- 舊結果檔中仍可能保留舊 extractor 產生的髒答案

目標：

- 讓 `Δ_t` 的正負至少不再受明顯格式問題主導

修法：

1. 持續補 evaluator normalization 的常見數學格式
2. 盡量改為 parser-first 的 answer extraction
3. 對舊結果做離線重評估，必要時重跑

---

### 3.2 Takeover 目前比較像 Prefix-Conditioned Re-Solve

目前的 large takeover 並不是嚴格的 append-only continuation。更準確地說，它是：

- 看到題目
- 看到 prefix
- 然後重新完成剩下的推理

所以現在量到的東西，比較像：

> prefix-conditioned takeover / re-solve quality

而不是：

> strict append-only continuation quality

這不一定是壞事，但必須誠實定義。

修法：

1. 先在 methods 中明確寫清楚這點
2. 不急著硬把模型限制成 append-only，避免又把生成行為扭曲回不自然分佈
3. 若之後需要，可補一個 ablation：
   - free takeover
   - minimal-edit takeover

---

### 3.3 ROI / Cost Comparability

目前成本指標還不能直接拿來做太強的經濟性結論。

原因：

- 有些模型 provider 會回報大量 `reasoning_tokens`
- 有些模型 provider 幾乎不回報

所以目前若直接用：

- `input + output + reasoning`

去比不同模型，會有不公平問題。

修法：

之後所有成本分析改成雙報表：

1. **Visible token cost**
- `prompt_tokens + completion_tokens`

2. **Provider-reported total cost**
- `prompt + completion + reasoning`

這樣即使兩種口徑不一致，也能明確地當 limitation 報告。

---

### 3.4 Model-Specific Risk

目前主結果是：

- small worker: `Qwen-3-14B`
- large worker: `GPT-5.4`

因此還不能直接把現象講成完全通論。

這類 caveat 不能靠寫作解掉，只能靠擴張實驗。

修法：

固定 large worker，先換 2 到 3 顆 small worker：

- `Qwen-3-14B`
- `Qwen-3.5-27B`
- `Llama-4-Scout` 或 `Llama-4-Maverick`

並將題數從 `10` 擴到 `30`。

要檢查的不是數值完全相同，而是：

- early gain trend 是否仍存在
- zero-gain subtype 分布是否相似
- task-type clustering 是否仍存在

---

### 3.5 Sample Size Still Small

目前只有 `10` 題，足以支持 early validation，但不足以支撐太強的統計性結論。

修法：

- 先跑到 `30`
- 若 pattern 仍穩，再擴到 `50`

---

## 四、建議的修法層級

### Layer 1: 立刻修

這些是現在就值得做，而且會直接提高資料可信度的：

1. answer extraction 更進一步朝 parser-first 收斂
2. evaluator normalization 持續補齊
3. analysis script 一律用 current evaluator 重算，不信 stored labels
4. methods wording 改成 `prefix-conditioned takeover`

### Layer 2: 用分析補強

這些不一定要先改 runner，但要改分析與報告方式：

1. 成本雙報表
2. zero gain subtype 拆分
3. per-model / per-question pattern summary

### Layer 3: 用實驗修

這些只能靠擴張實驗來處理：

1. 多 small worker
2. 更多題目
3. 之後再接 fragility

---

## 五、目前最建議的執行順序

1. 固定 large worker = `GPT-5.4`
2. 先完成乾淨版 analysis pipeline
3. 跑 `3` 顆 small worker
4. 每顆先跑 `30` 題
5. 比較：
   - `avg Δ_t vs step`
   - `positive / zero / negative`
   - `zero_both_correct / zero_both_wrong`
   - tasks with any positive gain
6. 若 pattern 穩，再接 `fragility`

---

## 六、目前最穩的說法

目前最穩的立場不是：

- 所有 caveats 都已經消失

而是：

我> The most serious label-purity caveats are now substantially reduced. The remaining caveats are primarily about interpretation scope, cost comparability, and cross-model generality, which should be addressed through clearer method definitions and broader experiments rather than more prompt engineering alone.
