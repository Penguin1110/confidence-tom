# 基於 Oracle Gain 與 Fragility 的推理介入研究提案

## 一、核心想法

這份提案重新定義多步推理中的介入問題。

傳統做法通常會直接依賴某些表面訊號，例如：

- 模型自報信心低
- 多次採樣答案不一致
- judge score 偏低
- 某些 heuristic anomaly 出現

這些訊號雖然方便，但有一個根本問題：

> 它們不一定穩定，也不一定真的對應「此刻值得介入」。

因此，這份提案主張：

- 不應先假設某個 proxy 就代表風險
- 應先定義「真正值得介入」的 oracle 原則
- 再去研究哪些可觀測訊號在不同 benchmark 下對這個 oracle 最穩定

我們的研究主張可以濃縮成一句話：

> 介入決策應建立在 reasoning-state fragility 或 counterfactual intervention gain 上，而不是單純依賴表面不確定性 proxy。

---

## 二、研究動機

在多步推理任務中，小模型雖然成本低，但常在中間某一步出現推理漂移。一旦某一步邏輯出錯，後續整條推理可能失效。大模型則較穩定，但推理成本高，不可能在所有步驟都全程接手。

因此，真正重要的問題不是：

- 哪些步看起來不太有信心？

而是：

- 哪些 prefix 如果現在不介入，後面最可能崩壞？
- 哪些 prefix 如果現在換成大模型，最可能帶來實際收益？

這使得介入問題更接近一個 **反事實增益估計（counterfactual gain estimation）** 問題，而不是單純的 confidence estimation 問題。

另外，我們目前的 pilot 也揭露了一個方法論上的瓶頸：

- LLM 並不自然地適合「一次只生成下一步」
- 強迫 stepwise generation 容易讓模型提早收尾、偷跑答案、或輸出不自然的 filler
- 因此，若直接把這種被扭曲的中間狀態當成 decision signal，研究結果本身可能被 protocol artifact 汙染

因此，本提案明確轉向：

> **full generation -> post-hoc segmentation -> prefix probing**

也就是先保留模型自然生成的完整推理，再在事後切分出虛擬步驟（virtual steps），最後對每個 prefix 做 oracle gain 與 fragility 分析。

---

## 三、研究問題

### RQ1

如何定義多步推理中某個 prefix 的「真實介入價值」？

### RQ2

哪些 reasoning-state 訊號，與真實介入增益的關聯，在不同 benchmark 下最穩定？

### RQ3

相較於 confidence、self-consistency、judge score 這些表面 proxy，基於 fragility 的結構性訊號是否更 robust？

### RQ4

相較於強制 stepwise generation，基於 **post-hoc segmented prefix** 的分析框架是否更穩定、更符合模型自然分佈？

### RQ5

在有限介入預算下，是否應採用每題內部相對排序，而不是跨 benchmark 共用同一個絕對 threshold？

---

## 四、核心定義

### 4.1 Oracle Intervention Gain

對於某題第 `t` 個 prefix 狀態 `h_t`，定義：

\[
\Delta_t = \mathrm{Perf}(\text{intervene at } t) - \mathrm{Perf}(\text{no intervention})
\]

其中：

- `Perf(intervene at t)`：在第 `t` 個 prefix 改由更強模型接手後的最終表現
- `Perf(no intervention)`：原小模型從同一 prefix 繼續完成後的最終表現

這個 `Δ_t` 是本研究的 **oracle objective**。

它代表：

- 這個 prefix 如果介入，究竟帶來多少真實收益

這個量在測試時通常拿不到，但在 offline analysis 階段可以藉由反事實 rollout 估計。

---

### 4.2 Fragility

給定第 `t` 個 prefix `h_t`，我們對其施加小擾動，產生一組語義保留的變體 `\tilde{h_t}`：

- paraphrase：用獨立 paraphraser 對當前 prefix 做語義保留重寫
- compression：刪除低資訊密度的冗語或重複句
- local rewrite：對單一 segment 做等價改寫
- optional reorder：只對語義上可交換的局部句子做微調

重要的是，這些 perturbation 不應是隨機破壞輸入，而應盡可能保留原始邏輯內容。形式上可寫為：

\[
\tilde{h}_t \sim \mathrm{Paraphrase}(h_t)
\]

或更一般地：

\[
\tilde{h_t} \sim P(h_t)
\]

然後比較擾動前後，後續 rollout 的差異：

\[
\mathrm{Fragility}_t =
\mathbb{E}_{\tilde{h_t} \sim P(h_t)}
\left[
d(F(\tilde{h_t}), F(h_t))
\right]
\]

其中：

- `h_t`：原始 prefix history
- `\tilde{h_t}`：小擾動後的 prefix history
- `F(\cdot)`：從該 prefix 繼續 rollout 的後續推理
- `d(\cdot,\cdot)`：後續結果差異的量測函數

在實作上，`d(\cdot,\cdot)` 可採用多層指標：

- correctness flip（最終 correctness 是否翻轉）
- final-answer disagreement（最終答案是否改變）
- optional semantic distance（最終回答的語義差異）

本提案的關鍵 claim 是：

> High fragility implies that the current reasoning state is unstable, and is therefore more likely to benefit from intervention.

也就是說，如果一個 prefix 對語義保留擾動非常敏感，那它就更可能位於需要介入的脆弱區域。

---

## 五、研究假設

### H1

真實介入增益 `Δ_t` 在 prefix 間高度不均，只有少數 prefix 具備顯著正增益。

### H2

Fragility-based 訊號與 `Δ_t` 的關聯，會比單純 confidence 或 self-consistency 更穩定。

### H3

跨 benchmark 不變的，不一定是某個分數本身，而是某個關係：

- fragility 高的 prefix，更可能有高 oracle gain

### H4

相較於強制 stepwise generation，post-hoc segmented prefix framework 能更好保留原始推理品質，因此得到的 signal 更可靠。

### H5

在有限預算下，以「每題內部排序」做介入決策，會比固定絕對 threshold 更 robust。

---

## 六、方法設計：Full Generation + Prefix Probing

### 6.1 完整生成與事後切分（Post-hoc Segmentation）

第一步不再要求 small worker 一次只生成下一步，而是先讓它以自然方式生成完整 reasoning trace：

\[
Q \rightarrow (r, a)
\]

其中：

- `Q`：原始題目
- `r`：完整 reasoning trace
- `a`：最終答案

接著再將 `r` 切分成一組虛擬步驟（virtual steps）：

\[
r \rightarrow (s_1, s_2, \dots, s_T)
\]

Segmentation 可採用：

- 顯式 step marker（若模型自然產生 `Step 1`, `Step 2`）
- 句子 / 段落切分
- rule-based chunking
- semantic segmentation

我們關心的不是切出唯一正確的 step，而是切出一組：

- 語義上大致完整
- 能夠形成合理 prefix
- 足以支撐後續 rollout probing

---

### 6.2 Prefix-based Oracle Gain Analysis

第一階段不急著學 policy，而是先建立每題每個 prefix 的 oracle gain map。

具體做法：

1. 讓 small worker 生成完整 trace
2. 將 trace 切分成 `s_1, \dots, s_T`
3. 對每個 prefix `h_t = (s_1, \dots, s_t)`：
   - 建立 `no intervention` rollout：
     - 給 small worker：`[Question] + [prefix h_t]`，請其完成剩餘推理並給最終答案
   - 建立 `intervene at t` rollout：
     - 給 large worker：`[Question] + [prefix h_t]`，請其從當前狀態接續推理並給最終答案
4. 比較兩者最終結果，得到 `Δ_t`

輸出將是：

- 每題每個 prefix 的 oracle intervention gain
- 每條 trace 的 gain curve
- 哪些 prefix 是：
  - positive gain
  - zero gain
  - negative gain

這一步是整個研究最關鍵的基礎。

此外，在後續成本分析中，介入成本不應只計算大模型後續生成的 token，還必須包含：

- **prefix reading cost**：大模型讀取 `h_t` 所消耗的 token
- **continuation generation cost**：大模型從 `h_t` 繼續完成推理的 token

因此，prefix 越長，takeover 成本越高。這也是為什麼 prefix-based framework 能自然刻畫 early intervention 與 late intervention 的成本差異。

---

### 6.3 Fragility Analysis on Prefixes

在估出 oracle gain 之後，再對每個 prefix 做局部擾動。

對每個 `h_t = (s_1, \dots, s_t)`：

1. 產生若干語義保留擾動版本 `\tilde{h_t}`
2. 對每個擾動版本 rollout
3. 量測：
   - 最終答案是否改變
   - 最終 correctness 是否翻轉
   - 後續推理路徑差異
   - confidence 是否劇烈改變

輸出：

- `fragility_t`
- `prefix continuation stability`
- `rollback recoverability`

然後分析這些量與 `Δ_t` 的關聯。

---

### 6.4 Signal Comparison

我們比較以下幾類訊號與 oracle gain 的關係：

#### A. Confidence 類

- self-reported confidence
- confidence drop
- maximum drop intensity

#### B. Prefix Consistency 類

- self-consistency
- prefix continuation agreement
- rollout variance across multiple completions

#### C. Fragility 類

- perturbation sensitivity
- semantic drift after perturbation
- rollout variance under paraphrase

#### D. Trajectory / Recovery 類

- backtracking
- self-correction depth
- rollback recoverability

不是直接問哪個 score 最大，而是問：

> 哪一類 signal 與 prefix-level `Δ_t` 的 rank correlation 在不同 benchmark 下最穩？

---

### 6.5 Relative / Budgeted Intervention

最終 policy 不採用固定 threshold：

- 不用 `risk > 0.7` 就介入
- 不用 `disagreement > x` 就介入

而改成：

- 對同一題內部所有 prefix 做排序
- 在有限 budget 下，只介入 top-k 最值得介入的 prefix

可以寫成：

\[
\mathrm{Priority}_t = \frac{S_t}{\sum_{i=1}^{T} S_i}
\]

其中 `S_t` 是某個 signal 或 gain predictor。

這樣做的好處是：

- 不要求不同 benchmark 的 score 在同一尺度上可比
- 更符合有限資源配置的真實需求

---

## 七、資料收集

### 7.1 Full Trace Collection

先讓 small worker 自然生成完整 reasoning trace，而不是強制單步生成。

必要欄位包括：

- 原始完整推理文字
- 最終答案
- 自報信心（若可得）
- 多次採樣版本（若要做 consistency）

之後再透過 segmentation 將其轉為：

- `s_1, s_2, ..., s_T`
- prefix `h_t = (s_1, ..., s_t)`

### 7.2 Segmentation Protocol

切分可以採多種策略，研究上可同時比較：

1. **rule-based segmentation**
- 依句子、段落、顯式 step marker 切分

2. **semantic segmentation**
- 依語義轉折、subgoal 變化、或 embedding 距離切分

3. **hybrid segmentation**
- 先 rule-based，再用輕量模型做邊界修正

我們不要求 segmentation 絕對正確，而是要求它：

- 足夠穩定
- 能構成合理 prefix
- 能用於跨方法比較

### 7.3 Counterfactual Prefix Rollouts

對每個 prefix 做：

- small continue
- large takeover

必要時也可以加：

- perturbed small continue
- perturbed large takeover

這些資料會構成：

- `Δ_t`
- `fragility_t`
- 相對排序 supervision

---

## 八、From Oracle Gain to Deployable Policy

Oracle gain `Δ_t` 是本研究的 supervision target，但它在真實 inference 時不可直接取得。因此，方法的最終目標不是直接在測試時使用 oracle，而是學習一個可部署的 gain predictor：

\[
\hat{\Delta}_t = f(o_t)
\]

其中 `o_t` 是在 prefix `h_t` 上可觀測的訊號，例如：

- prefix text / prefix embedding
- continuation divergence
- fragility under perturbation
- prefix consistency
- confidence-related features
- rollback / self-correction related features

也就是說，本研究的 pipeline 分成兩層：

1. **Oracle analysis layer**
- 用 counterfactual rollouts 估計 `Δ_t`
- 釐清什麼樣的 prefix 真正值得介入

2. **Deployable policy layer**
- 學習 `\hat{\Delta}_t = f(o_t)`
- 在測試時僅依靠可觀測訊號做決策

這一點很重要，因為它把本研究從純分析工作延伸成可部署的方法框架。

同時，我們的方法 claim 可以表述為：

> We propose a prefix-based intervention framework that estimates the counterfactual value of handing off a partial reasoning trajectory to a stronger model.

> We approximate this value using prefix-conditioned signals, including fragility under semantic-preserving perturbations and continuation divergence.

## 八、評估指標

### 8.1 分析層

- Spearman rank correlation between signal and `Δ_t`
- Kendall tau across benchmarks
- cross-benchmark stability of top-k intervention ranking
- segmentation strategy sensitivity

### 8.2 政策層

- final success rate
- total token cost
- gain per unit cost
- Pareto frontier
- intervention precision / recall

---

## 九、預期貢獻

### Contribution 1

提出一個比 heuristic 更乾淨的介入定義：

- oracle intervention gain

### Contribution 2

提出一個 **不強迫模型逐步生成** 的 prefix-based intervention framework：

- full generation
- post-hoc segmentation
- prefix probing

### Contribution 3

證明 fragility 是比 confidence / disagreement 更 robust 的介入訊號候選。

### Contribution 4

說明在多 benchmark 情境中，穩定的不是單一分數，而是某些結構性關係。

### Contribution 5

提出 relative / budgeted intervention policy，避免使用跨 benchmark 不穩定的全域 threshold。

---

## 十、與目前專案的連結

目前 repo 已經完成了兩個重要前置：

1. `Qwen-3` worker scaling 結果
2. observer 對 worker trace 的可欺騙性分析

這些結果說明：

- 較強模型的錯誤更具有欺騙性
- 單靠表面 confidence 或 observer 判斷，未必足以 robust 地做介入決策

此外，最近的 stepwise oracle-gain pilot 還提供了一個方法上的反證：

- 很多模型在「一次只輸出下一步」的協議下會出現不自然行為
- 包括提早收尾、偷跑答案、格式近似正確但 schema 不穩、以及推理品質下降

因此，下一步自然不是直接把某個 judge score 當成 router，也不應再把硬切 step 當成主假設，而是：

- 先保留完整自然推理
- 再做 post-hoc segmentation
- 最後以 prefix 為單位定義 oracle gain 與 fragility

這也讓目前的 scale/observer 工作，成為 intervention 研究的理論與實證前導。

---

## 十一、最小可行版本（MVP）

### MVP-1：Full Trace + Prefix Oracle Gain Mapping

- 任務：`OlympiadBench`
- small worker：一個可穩定完成完整推理的模型
- large worker：更強的大模型
- 對每題完整 trace 做 segmentation
- 對每個 prefix 估 `Δ_t`

### MVP-2：Fragility Estimation

- 對每個 prefix 做 2~3 種語義保留擾動
- 估測 prefix fragility

### MVP-3：Signal Ranking

- 比較：
  - confidence
  - prefix consistency
  - fragility
  - rollback-related signals

### MVP-4：Stepwise Generation Baseline

- 保留目前的 stepwise generation pipeline
- 但將其降為 baseline / ablation
- 用來驗證：
  - 強制單步生成是否會扭曲 signal
  - 與 post-hoc segmented prefix framework 相比，哪種方式更穩

### MVP-5：Budgeted Policy

- 每題只介入 top-k 步
- 與固定 threshold baseline 比較

---

## 十二、目前最推薦的主線

如果要追求 robust 與研究價值，最推薦的路線是：

1. **Oracle gain analysis first**
2. **Full generation + post-hoc segmentation**
3. **Fragility-based signal comparison**
4. **Budgeted relative intervention policy**

也就是：

- 先定義正確目標
- 再用自然生成取得 prefix
- 再比較 proxy
- 最後才做 policy

而不是一開始就直接相信某個 heuristic，或強迫模型以不自然的方式逐步生成。
