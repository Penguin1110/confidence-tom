
# 實驗目標

這個實驗要回答的核心問題不是「哪個模型單純比較強」，而是：

> 當 small model 推理到一半時，讓更強的 large model 從當前 prefix 接手，是否真的有價值？而且這個價值是否取決於當下的 reasoning state？

更具體地說，我們想檢查三件事：

1. **takeover value 是否存在**
- 也就是 large model 從某個 prefix 接手後，是否比 small model 自己繼續更容易答對。

2. **takeover value 是否是 state-dependent**
- 同一題在不同 prefix 上，接手價值可能不同。
- 有些 prefix 已經很穩，不需要介入；有些 prefix 雖然還沒明顯答錯，但其實很脆弱，接手就可能有很大收益。

3. **這個現象是否跨 model family 成立**
- 如果只有某一組 model pairing 會出現，就比較像模型特例。
- 如果在不同 small / large family 上都反覆觀察到，就更像一般性的 reasoning-state phenomenon。

## 方法概述

我們現在用的是 **prefix-based intervention** 架構，而不是先前那種強制 stepwise JSON 的方式。

整體流程如下：

1. **Full Generation**
- 先讓 small model 對原題自然完整生成一條 reasoning trace。

2. **Segmentation**
- 再把完整 trace 切成幾個 reasoning segments。
- 每個 segment 代表一個相對完整的推理單位。

3. **Prefix Construction**
- 對每個切點建立 prefix：
  - 第 1 段
  - 前 2 段
  - 前 3 段
  - ...

4. **雙 rollout 比較**
- 對每個 prefix，同時做兩種 continuation：
  - `small continue`：讓 small model 自己繼續做完
  - `large takeover`：讓 large model 從這裡接手做完

5. **計算 takeover gain**
- 如果：
  - small 錯、large 對
  - 這個 prefix 就有正的 takeover gain
- 如果兩邊都對或都錯，就是 zero gain
- 如果 small 對、large 錯，就是 negative gain

因此，這個實驗本質上是在量化：

> **prefix-conditioned takeover value**

而不是只看整題最後誰的 accuracy 比較高。

## 實驗設定

這份文件整理的是 OlympiadBench 上已完成的 50 題 prefix-based oracle-gain 實驗結果。

Small model families：
- Qwen: `qwen/qwen3-14b:nitro`
- Llama: `meta-llama/llama-4-scout`
- Mistral: `mistralai/ministral-8b-2512`

Large model families：
- OpenAI: `openai/gpt-5.4`
- Anthropic: `anthropic/claude-opus-4.6`

評估說明：
- 答案抽取使用目前的 parser-first 與較穩定的 LaTeX-aware extraction。
- full-trace answer normalization 已處理 `\boxed`、`\fbox`、`\dfrac`、`\tfrac`。
- prefix-level label 採二元 correctness difference：
  - positive：large 對、small 錯
  - zero：兩邊都對或兩邊都錯
  - negative：large 錯、small 對

## 總表

| Run | 題數 | Full-trace 正確 | Prefix steps | Positive | Zero | Negative | 至少有一個 positive 的題數 | Small success rate | Large success rate | 平均 segments | 最大 segments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen_to_openai_50` | 50 | 24 | 230 | 21 | 198 | 11 | 8 | 0.504 | 0.548 | 4.60 | 17 |
| `qwen_to_anthropic_50` | 50 | 24 | 213 | 12 | 194 | 7 | 7 | 0.455 | 0.479 | 4.26 | 7 |
| `llama_to_openai_50` | 50 | 17 | 390 | 97 | 276 | 17 | 20 | 0.279 | 0.485 | 7.80 | 25 |
| `llama_to_anthropic_50` | 50 | 20 | 383 | 95 | 283 | 5 | 16 | 0.300 | 0.535 | 7.66 | 25 |
| `mistral_to_openai_50` | 50 | 20 | 229 | 40 | 184 | 5 | 19 | 0.349 | 0.502 | 4.58 | 7 |
| `mistral_to_anthropic_50` | 50 | 15 | 219 | 29 | 188 | 2 | 13 | 0.333 | 0.457 | 4.38 | 6 |

## 前幾步的 `Delta_t` 曲線

前 8 步的平均 `Delta_t`：

- `qwen_to_openai_50`: `{1: 0.000, 2: 0.040, 3: 0.020, 4: 0.122, 5: 0.067, 6: 0.143, 7: 0.000, 8: 0.000}`
- `qwen_to_anthropic_50`: `{1: 0.080, 2: 0.020, 3: 0.020, 4: 0.000, 5: -0.056, 6: 0.000, 7: 0.000}`
- `llama_to_openai_50`: `{1: 0.220, 2: 0.180, 3: 0.240, 4: 0.180, 5: 0.244, 6: 0.182, 7: 0.160, 8: 0.350}`
- `llama_to_anthropic_50`: `{1: 0.160, 2: 0.220, 3: 0.180, 4: 0.229, 5: 0.227, 6: 0.235, 7: 0.273, 8: 0.200}`
- `mistral_to_openai_50`: `{1: 0.260, 2: 0.140, 3: 0.100, 4: 0.130, 5: 0.083, 6: 0.286, 7: 0.000}`
- `mistral_to_anthropic_50`: `{1: 0.140, 2: 0.140, 3: 0.102, 4: 0.175, 5: 0.000, 6: 0.143}`

## `Delta_t` 的 pattern：目前最核心的 scientific finding

如果把這個實驗拆開看，我們現在其實在回答三個不同層次的問題：

1. **takeover value 是否存在**
- 也就是大模型接手後，是否真的能改善結果。

2. **`Delta_t` 本身有沒有可解釋的 pattern**
- 例如：
  - early prefix 的 gain 是否普遍高於 late prefix
  - 有些題是否反覆呈現「可救」
  - 有些題是否反覆呈現「不可救」

3. **能不能用便宜的 signal 預測 `Delta_t`**
- 也就是不真的把 large takeover 全跑一遍，而是先用較便宜的特徵估計：
  - 這個 prefix 值不值得介入

這三層裡，現在資料最支持、也最像 scientific finding 的，是第 2 層：

> **`Delta_t` 不是隨機噪音，而是有結構的。**

目前最清楚的 pattern 包括：

- **early prefix 往往比 late prefix 更有 takeover gain**
  - 這在多個 family pairing 中都看得到，只是強度不同。
  - 代表接手價值不是固定常數，而會隨 reasoning state 改變。

- **有些題是穩定可救的**
  - 例如 `olympiadbench_1977_0022`、`olympiadbench_2127_0029`
  - 它們在 `6/6` family pairing 中都出現 positive gain。

- **有些題是穩定不可救的**
  - 這類題會大量落在 `zero both-wrong`
  - 代表在目前 protocol 下，即使換更強模型接手，也不一定能從該 prefix 恢復。

- **不同 small-family 的 `Delta_t` regime 不一樣**
  - Qwen 比較像：
    - baseline 強
    - 很多 zero 是 `both_correct`
    - 因此 marginal gain 較小
  - Llama 比較像：
    - state 較脆弱
    - 很多 zero 是 `both_wrong`
    - 同時 positive gain 也更大
  - Mistral 介於兩者之間

這些 pattern 合起來支持一個比較穩的說法：

> **我們量到的不是單純的模型強弱差，而是 prefix state 本身是否處在「可恢復」或「已穩定」的區域。**

## 便宜 signal 能不能預測 `Delta_t`

這是這條研究線下一個很自然、也很關鍵的問題。

因為如果 `Delta_t` 有結構，那下一步就不是只做 oracle analysis，而是問：

> **能不能不用真的把 large model 叫出來，就先預測這個 prefix 值不值得接手？**

目前最合理的便宜 signal family 包括：

- **prefix length / step index**
  - 最簡單，也最穩定
  - 目前資料已經支持：
    - early prefix 往往比 late prefix 更有 gain

- **segmentation-based complexity**
  - 例如：
    - prefix 已累積多少 segments
    - 當前 segment 是否異常冗長
  - 這些特徵反映的是 reasoning state 是否已經過度展開或變得混亂

- **inconsistency / instability**
  - 例如：
    - 局部推理是否反覆改寫
    - 是否出現明顯重述、枚舉、來回修正
  - 這類 signal 直覺上更接近「當前 state 是否脆弱」

- **fragility**
  - 這是最值得延伸的一支
  - 也就是：
    - 小擾動之下，small model 的 continuation 是否容易改變
    - 如果某個 prefix 很脆弱，理論上 takeover value 應該更高

換句話說，我們最後要學的不是直接預測答案，而是預測：

`Delta_t`

也就是：
- 這個 prefix 是否值得介入
- 介入的期望收益大不大

因此，目前這份 family comparison 的角色，可以理解成：

- 第一步：證明 `Delta_t` 不是亂數，而是有 pattern
- 第二步：找哪些便宜 signal 可以逼近這個 pattern
- 第三步：把 oracle takeover 變成可部署的 intervention policy

## 信心與 fragility 的結合方式

原本的「信心」想法不需要丟掉，但在目前這條 prefix-based pipeline 裡，比較合理的角色不是把它當成主假說本身，而是把它視為一類 **confidence-related signals**，再和 **fragility-related signals** 一起比較。

這樣做有兩個好處：

1. **不需要假設 self-report confidence 一定可靠**
- 目前這條 pipeline 沒有穩定、原生的 self-reported confidence。
- 如果硬把 proxy 當成真實信心，說服力會比較弱。

2. **可以讓資料自己決定哪類 signal 更有用**
- 我們不用先驗地說「信心一定最重要」。
- 可以直接比較：
  - confidence-related signals
  - fragility-related signals
  - 結構特徵
  到底哪一類更能預測 `Delta_t`。

### A. confidence-related signals

這一類描述的是 prefix 表面上呈現出的不確定性或承諾強度，例如：

- `hedge_density`
- `confidence_proxy`
- 回頭修正 / self-correction 痕跡
- backtracking

它們不一定等於「真實信心」，但至少提供了 prefix-level 的 uncertainty cue。

### B. fragility-related signals

這一類描述的是 prefix 在小擾動下的穩定性，也就是更接近行為層面的信心。

在目前設定下，我們比較適合用：

- **deterministic input perturbation**
  - 保持 `temperature=0`
  - 對 prefix 做語義保持的小擾動
  - 再看 small continuation 是否明顯改變

這樣得到的 signal 例如：

- perturbation 後答案是否改變
- perturbation 後 correctness 是否 flip
- continuation divergence 是否變大

如果某個 prefix 對這種小擾動非常敏感，就表示它的 reasoning state 較 fragile，也更可能有 takeover value。

### C. 下一步最自然的 predictor 問題

因此，下一步比較好的研究問題不是：

> 信心是不是最重要？

而是：

> **在 prefix-level intervention setting 中，confidence-related signals 與 fragility-related signals，哪一種更能預測 oracle takeover gain？**

這樣的好處是：

- 保留原本的信心直覺
- 不把信心講得太滿
- 同時把 fragility 正式拉進來，形成更完整的 predictor 框架

### D. 實作上可分成三層特徵

1. **structure**
- `step_index`
- `prefix length`
- `prefix tokens`

2. **confidence-related**
- `hedge_density`
- `confidence_proxy`
- backtracking / self-correction 類特徵

3. **fragility-related**
- perturbation answer changed
- perturbation correctness changed
- perturbation continuation divergence

最後預測的不是最終答案，而是：

- `delta_positive`
或
- `Delta_t`

也就是：
- 這個 prefix 值不值得介入
- 介入的期望收益大不大

## 主要觀察

### 1. 這個現象不是單一 small-model family 的特例

三個 small-model families：
- Qwen
- Llama
- Mistral

都能在同一個 benchmark 上產生可用的 prefix-based oracle-gain data。

這支持一個比較穩的說法：
- prefix-conditioned takeover 不是某一顆模型的偶發怪癖
- 而是跨 family 可觀察的現象

### 2. Qwen 是目前最強、也最乾淨的 small-model baseline

Qwen 在三個 families 中有最高的 full-trace correctness：
- 兩個 large-model family 下都是 `24/50`

它的 segmentation 也最乾淨：
- 平均 segments：`4.26` 到 `4.60`
- 最大 segments：Anthropic run 是 `7`，OpenAI run 是 `17`

解讀：
- Qwen 本身比較容易產生穩定、較 coarse-grained 的推理軌跡
- takeover 還是有幫助，但邊際收益相對較小

### 3. Llama 的 takeover signal 最強，但 granularity 也最碎

Llama 的 positive-gain prefix 數量最高：
- `97`（`llama_to_openai_50`）
- `95`（`llama_to_anthropic_50`）

至少有一個 positive-gain prefix 的題數也最高：
- `20`
- `16`

但代價也很明顯：
- 平均 segments 約 `7.7`
- 最大 segments `25`

解讀：
- Llama 比較容易 externalize 出脆弱、細粒度的 reasoning state
- 因此 takeover 更常有明顯收益
- 但也會拖高 runtime，並讓 granularity control 變得更重要

### 4. Mistral 是最穩的中間組

Mistral 不是最強的 full-trace baseline，但它很穩：
- full-trace correctness：OpenAI 下 `20/50`，Anthropic 下 `15/50`
- segmentation 很乾淨：平均 `4.38` 到 `4.58`，最大 `6` 到 `7`

它的 takeover signal 明顯存在，但沒有 Llama 那麼極端：
- positive prefixes：`40`（OpenAI）、`29`（Anthropic）
- 至少一個 positive-gain 的題數：`19`、`13`

解讀：
- Mistral 很適合當 control family
- 因為它結構乾淨，又能保有可觀察的 takeover signal

### 5. large-model family 的差異是看得到的，尤其在 Llama 上

在 Llama 設定下，Anthropic 看起來比 OpenAI 更像有效 takeover model：
- large success rate：`0.535` vs `0.485`
- negative prefixes：`5` vs `17`

這是目前最明顯的 large-family 差異。

對 Qwen 和 Mistral 而言，large-family 差異比較小，也沒有這麼單方向。

解讀：
- takeover value 不只取決於 small model，也取決於 large model family
- 目前最清楚的 large-family effect 出現在 Llama 設定下

### 6. Negative takeover 確實存在，而且不同 family pairing 的風險不同

Negative prefixes 雖然不是主體，但不是零：
- Qwen -> OpenAI: `11`
- Qwen -> Anthropic: `7`
- Llama -> OpenAI: `17`
- Llama -> Anthropic: `5`
- Mistral -> OpenAI: `5`
- Mistral -> Anthropic: `2`

解讀：
- takeover 不是永遠安全
- 不同 family pairing 的「誤接手風險」差很多

## 目前最穩的總結

一個目前可以站得住的說法是：

> Takeover gain 並不是在所有 model family 上都一樣強。當 small model externalize 出較脆弱、較細粒度的 reasoning states 時，takeover gain 較大；當 small model 本身已經產生相對穩定、較 coarse 的 trace 時，takeover 的邊際收益就較小。

在目前資料裡：
- Llama 最能代表 high-gain / high-fragility 的 regime
- Qwen 最能代表 strong-baseline / low-marginal-gain 的 regime
- Mistral 是最乾淨的中間控制組

## Zero-Gain Breakdown

Zero-gain prefixes 可以再拆成兩種有意義的 subtype：
- `both_correct`：small 本來就夠好了，不需要 takeover
- `both_wrong`：在這個 protocol 下，這個 prefix 對兩邊都還是不可救

| Run | Zero both-correct | Zero both-wrong |
| --- | ---: | ---: |
| `qwen_to_openai_50` | 105 | 93 |
| `qwen_to_anthropic_50` | 90 | 104 |
| `llama_to_openai_50` | 92 | 184 |
| `llama_to_anthropic_50` | 110 | 173 |
| `mistral_to_openai_50` | 75 | 109 |
| `mistral_to_anthropic_50` | 71 | 117 |

解讀：
- Qwen 的 `both_correct` 比例相對較高，符合它 baseline 較強的圖像
- Llama 的 `both_wrong` 明顯更多，符合它 reasoning state 較脆弱、也較難恢復的圖像
- Mistral 仍然落在中間位置

## Positive-Gain Overlap

至少有一個 positive-gain prefix 的題，跨 run 的 pairwise overlap（Jaccard similarity）如下：

| Pair | Intersection | Union | Jaccard |
| --- | ---: | ---: | ---: |
| `qwen_to_openai_50` vs `qwen_to_anthropic_50` | 3 | 12 | 0.250 |
| `qwen_to_openai_50` vs `llama_to_openai_50` | 7 | 21 | 0.333 |
| `qwen_to_openai_50` vs `llama_to_anthropic_50` | 2 | 22 | 0.091 |
| `qwen_to_openai_50` vs `mistral_to_openai_50` | 8 | 19 | 0.421 |
| `qwen_to_openai_50` vs `mistral_to_anthropic_50` | 3 | 18 | 0.167 |
| `qwen_to_anthropic_50` vs `llama_to_openai_50` | 4 | 23 | 0.174 |
| `qwen_to_anthropic_50` vs `llama_to_anthropic_50` | 5 | 18 | 0.278 |
| `qwen_to_anthropic_50` vs `mistral_to_openai_50` | 4 | 22 | 0.182 |
| `qwen_to_anthropic_50` vs `mistral_to_anthropic_50` | 6 | 14 | 0.429 |
| `llama_to_openai_50` vs `llama_to_anthropic_50` | 13 | 23 | 0.565 |
| `llama_to_openai_50` vs `mistral_to_openai_50` | 15 | 24 | 0.625 |
| `llama_to_openai_50` vs `mistral_to_anthropic_50` | 10 | 23 | 0.435 |
| `llama_to_anthropic_50` vs `mistral_to_openai_50` | 10 | 25 | 0.400 |
| `llama_to_anthropic_50` vs `mistral_to_anthropic_50` | 11 | 18 | 0.611 |
| `mistral_to_openai_50` vs `mistral_to_anthropic_50` | 9 | 23 | 0.391 |

最強的 overlap 是：
- `llama_to_openai_50` vs `mistral_to_openai_50`: `0.625`
- `llama_to_anthropic_50` vs `mistral_to_anthropic_50`: `0.611`
- `llama_to_openai_50` vs `llama_to_anthropic_50`: `0.565`

解讀：
- positive-gain tasks 不是隨機噪音
- 跨 family pairing 的 recurrence 很明顯
- Llama 和 Mistral 共享的 positive-gain tasks 比我原本預期高
- Qwen 的 overlap 較弱，這也符合它 baseline 較穩、比較少需要接手

## 最常共享的 Positive-Gain Tasks

出現在最多 family pairing 中的 positive-gain tasks：

- `olympiadbench_1977_0022`：`6/6`
- `olympiadbench_2127_0029`：`6/6`
- `olympiadbench_1798_0047`：`5/6`
- `olympiadbench_2666_0010`：`5/6`
- `olympiadbench_2973_0006`、`olympiadbench_3008_0035`、`olympiadbench_2744_0031`、`olympiadbench_1866_0034`：`4/6`

解讀：
- 這些題很適合當 case study
- 因為它們的 takeover value 不是某一組 pairing 才有，而是跨 family 都存在
- 這更像是 benchmark 結構下真正穩定的 reasoning-state phenomenon

## Case Study 候選與發現

### 候選 1：`olympiadbench_2127_0029`

這題是目前最乾淨、最穩的主正例。

它在 `6/6` runs 中都出現 positive gain，而且多數 pairing 不是只出現單一個 `+1`，而是多個 prefix 都有正收益：

- `qwen_to_openai`: `[1, 1, 1, 1, 1, 1]`
- `qwen_to_anthropic`: `[1, 1, 1, 1]`
- `llama_to_openai`: `[1, 0, 1, 0, 1, 1, 0, 1, 0]`
- `llama_to_anthropic`: `[1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]`
- `mistral_to_openai`: `[1, 1, 1, 0, 0, 1, 0]`
- `mistral_to_anthropic`: `[1, 1]`

**發現**
- 這題非常適合當主案例，因為它支持一個最核心的 claim：
  - takeover value 不是單一 family 的偶發現象
  - 同一個 reasoning task 在不同 small / large family 配對下，都反覆出現正收益
- 它也顯示 takeover value 不一定只集中在第一步，有些題會在多個 prefix 上持續維持可救狀態。

### 候選 2：`olympiadbench_1977_0022`

這題也是 `6/6` 全中，但它更適合拿來展示 family interaction。

各組 delta pattern：
- `qwen_to_openai`: `[0, 0, 0, 1]`
- `qwen_to_anthropic`: `[1, 1, 1]`
- `llama_to_openai`: `[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]`
- `llama_to_anthropic`: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- `mistral_to_openai`: `[1, 0, 0, 0]`
- `mistral_to_anthropic`: `[1, 1, 1]`

**發現**
- 同一題在不同 family pairing 下，gain 的強度和分布明顯不同。
- 特別是 `llama_to_anthropic` 幾乎整段都正 gain，說明：
  - 有些 pairing 不是只有 early takeover 有價值
  - 而是整條 reasoning trace 都維持脆弱，large model 幾乎整段都能接出更好的後續
- 這題很適合支持：
  - takeover value 不只取決於 task
  - 也取決於 small family 與 large family 的互動

### 候選 3：`olympiadbench_1798_0047`

這題在 `5/6` runs 中有 positive gain，但同時也是目前最好的邊界案例，因為它混合了 positive、zero、negative。

各組 delta pattern：
- `qwen_to_openai`: `[0, 0, 1, 0, 0]`
- `qwen_to_anthropic`: `[1, -1, 0, -1, -1]`
- `llama_to_openai`: `[0, 0, 1, 0, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]`
- `llama_to_anthropic`: `[-1, -1, 0, 0, -1]`
- `mistral_to_openai`: `[1, 1, 0, 1, 1]`
- `mistral_to_anthropic`: `[0, 0, 1, 0, -1]`

**發現**
- takeover 並不是單向安全的。
- 同一題在某些 pairing 下會有明顯 gain，但在另一些 pairing 下反而出現 harm。
- 這題很適合支持：
  - 為什麼不能只看「有沒有 positive」
  - 還要看 family pairing 與 prefix state 的交互作用
- 如果要向老師說明「這不是簡單的大模型接手就一定更好」，這題非常有用。

### 備選：`olympiadbench_2666_0010`

這題在 `5/6` runs 中也有 positive gain，但 pattern 比 `1798_0047` 更乾淨：

- `qwen_to_openai`: `[0, 0, 0, 0]`
- `qwen_to_anthropic`: `[1, 0, 0]`
- `llama_to_openai`: `[0, 0, 1, 0, 0]`
- `llama_to_anthropic`: `[1, 1, 1]`
- `mistral_to_openai`: `[0, 1, 0, 0, 0]`
- `mistral_to_anthropic`: `[0, 0, 0, 1]`

**發現**
- 這題更像「次級正例」或 supplementary example。
- 它適合拿來展示：
  - 有些題不是每一組都大幅 gain
  - 但跨 family 仍然有穩定出現的 takeover opportunity

### Case Study 總結

如果只挑 3 題，我目前最推薦：

1. `olympiadbench_2127_0029`
   - 最穩的主正例
2. `olympiadbench_1977_0022`
   - 最能展示 family interaction 的正例
3. `olympiadbench_1798_0047`
   - 最能展示 positive / negative 並存的邊界案例

這三題放在一起，最能完整呈現目前實驗的核心故事：
- takeover value 可以跨 family 穩定存在
- family pairing 會影響 gain 的強弱與分布
- takeover 不是單向安全，而是 state-dependent 也 pairing-dependent

## Caveats

- 目前結果依賴的是 binary correctness，不是 partial credit。
- 現在的 takeover protocol 更準確地說是 `prefix-conditioned takeover / re-solve`，不是 strict append-only continuation。
- Llama 仍然有少數 segmentation 長尾問題。
- 如果之後要談 cost，仍然應該用一致的 token-accounting 口徑。

## 下一步

1. 從最常共享的 positive-gain tasks 中挑 2 到 3 題做 case study。
2. 把 early-step gain 另做一張 family-level 小表。
3. 重新收斂 Llama 的 full-trace prompt，壓低過細 decomposition。
4. 在這個 family 結果上接 fragility 分析。
