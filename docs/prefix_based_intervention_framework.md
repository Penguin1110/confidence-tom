# Prefix-Based Intervention Framework

## 一、目的

這份文件把新的主框架寫成比較接近 methods / algorithm note 的版本，方便後續實作與論文方法章節整理。

主原則是：

> 不再強迫模型逐步生成；改為先完整生成，再做事後切分與 prefix probing。

---

## 二、核心流程

### Step 1：Full Generation

給 small worker 原始題目 `Q`，讓模型用自然方式完成完整推理：

\[
Q \rightarrow (r, a)
\]

其中：

- `r`：完整 reasoning trace
- `a`：最終答案

### Step 2：Segmentation

將完整 trace `r` 切成一組 reasoning segments：

\[
r \rightarrow (s_1, s_2, \dots, s_T)
\]

### Step 3：Prefix Construction

對每個 `t = 1, ..., T`，建立 prefix：

\[
h_t = (s_1, s_2, \dots, s_t)
\]

### Step 4：Counterfactual Prefix Rollouts

對每個 prefix `h_t`，執行兩條 rollout：

1. **Small Continue**
- small worker 從 `h_t` 繼續完成推理

2. **Large Takeover**
- large worker 從 `h_t` 接手完成推理

在目前實作中，這裡的 `takeover` 應理解為：

- **prefix-conditioned takeover / re-solve**

也就是 large worker 看到原題與 prefix 後，重新完成剩餘推理；它不是嚴格的 append-only continuation。

### Step 5：Oracle Gain

計算：

\[
\Delta_t = \mathrm{Perf}(\text{large takeover from } h_t) - \mathrm{Perf}(\text{small continue from } h_t)
\]

### Step 6：Fragility

對 `h_t` 產生若干語義保留擾動 `\tilde{h_t}`，再做 continuation rollout，估計：

\[
\mathrm{Fragility}_t = \mathbb{E}[d(F(\tilde{h_t}), F(h_t))]
\]

### Step 7：Signal Study / Policy

分析哪些 signal 最能預測 `Δ_t`，最後才做 intervention policy。

### Step 8：From Oracle Gain to Deployable Policy

在部署時，我們無法直接觀測 oracle gain `Δ_t`，因此需要學習一個可部署的 predictor：

\[
\hat{\Delta}_t = f(o_t)
\]

其中 `o_t` 是 prefix `h_t` 上可觀測的訊號，例如：

- prefix text / embedding
- continuation divergence
- fragility
- consistency
- confidence-related features

因此，oracle gain 的角色是 supervision target，而 `\hat{\Delta}_t` 才是實際部署時用來決定介入與否的量。

---

## 三、演算法草稿

### Algorithm 1: Oracle Gain Mapping

輸入：
- question `Q`
- small worker `M_s`
- large worker `M_l`
- segmentation function `Seg(·)`
- evaluator `Eval(·)`

輸出：
- prefix-level oracle gain map `{(h_t, Δ_t)}`
- prefix-level cost records `{(h_t, C_s, C_l)}`
- linkage metadata，例如 `trace_id`, `prefix_id`, `parent_prefix_id`

流程：

1. 生成完整 trace：`(r, a_s) = M_s(Q)`
2. 切分：`(s_1, ..., s_T) = Seg(r)`
3. 對每個 `t`：
   - `h_t = (s_1, ..., s_t)`
   - 指派 `prefix_id = (trace_id, t)`，並保存 `parent_prefix_id = (trace_id, t-1)`
   - `y_s = M_s(Q, h_t, continue=True)`
   - `y_l = M_l(Q, h_t, continue=True)`
   - 計算成本：
     - `C_s = C_{small-read}(h_t) + C_{small-generate}`
     - `C_l = C_{large-read}(h_t) + C_{large-generate}`
   - `Δ_t = Eval(y_l) - Eval(y_s)`
4. 回傳所有 `Δ_t` 與對應成本、prefix linkage metadata

成本分析建議同時保留兩種口徑：

1. **Visible token cost**
- `prompt_tokens + completion_tokens`

2. **Provider-reported total cost**
- `prompt_tokens + completion_tokens + reasoning_tokens`

原因是不同 provider 對 `reasoning_tokens` 的暴露程度不一致，因此雙口徑報告會比單一成本口徑更穩。

### Algorithm 2: Fragility Estimation

輸入：
- prefix `h_t`
- perturbation generator `P(·)`
- rollout model `M`
- distance metric `d(·, ·)`

輸出：
- `Fragility_t`

流程：

1. 取樣若干擾動 prefix `\tilde{h_t}^{(1)}, ..., \tilde{h_t}^{(K)}`
   - 主要使用 semantic-preserving paraphrase
   - 可輔以 compression 與 local rewrite
2. 分別 rollout：
   - `y_0 = M(Q, h_t, continue=True)`
   - `y_k = M(Q, \tilde{h_t}^{(k)}, continue=True)`
3. 計算：
   - correctness flip
   - final-answer disagreement
   - optional semantic distance
4. 聚合為 `d(y_k, y_0)`
5. 取平均，得到 `Fragility_t`

---

## 四、Segmentation 設計

### 4.1 Rule-Based

- 顯式 step marker
- 段落切分
- 句號 / 換行切分

優點：
- 便宜
- 可重現

缺點：
- 容易切到語義不完整的位置

### 4.2 Semantic

- 利用 embedding distance
- 利用 discourse transition
- 利用 subgoal shift

優點：
- 更接近推理結構

缺點：
- 較重
- 邊界定義較主觀

### 4.3 Hybrid

- 先 rule-based 初切
- 再用 semantic signal 修正邊界

這是目前最推薦的策略。

---

## 五、為什麼這比 Stepwise Generation 更合理

### 5.1 更符合模型分佈

大多數 LLM 是以：

- `input -> full output`

的方式被訓練，而不是：

- `input -> step 1 -> step 2 -> ...`

因此，prefix-based probing 更符合模型自然行為。

### 5.2 較少 protocol artifact

避免：

- 模型被迫壓縮思考
- 提早收尾
- 偷跑後續答案
- stepwise filler

### 5.3 可保留 step-level分析能力

雖然不是逐步生成，但仍可透過 prefix：

- 定義介入點
- 比較 takeover gain
- 做 fragility 測試

---

## 六、和目前 repo 的關係

目前 repo 已經有的東西可直接重用：

- dataset loader
- evaluator
- model client
- oracle gain runner 的 rollout 架構
- fragility / signal analysis skeleton

真正要改的核心不是整個專案，而是：

- **資料收集協議**

也就是從：

- stepwise generation

轉成：

- full generation + segmentation + prefix probing

---

## 七、實作建議

### Phase 1

- 保留既有 `run_oracle_gain_mapping.py`
- 新增 `run_prefix_oracle_gain_mapping.py`
- 先用 full trace + rule-based segmentation 跑通

### Phase 2

- 加 semantic segmentation
- 比較不同 segmentation 對 `Δ_t` 的穩定性

### Phase 3

- 加 fragility perturbation
- 做 signal ranking

### Phase 4

- 才做 budgeted intervention policy

---

## 八、最小可行版本

1. small worker 完整生成 trace
2. 用簡單規則切成 3 到 8 個 segments
3. 對每個 prefix 跑：
- small continue
- large takeover
4. 算出 `Δ_t`
5. 先分析：
- positive / zero / negative gain 分布
- 哪些 prefix 最常有 gain

這樣就已經足夠形成第一版 paper 核心實驗。
