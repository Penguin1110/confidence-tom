# Prefix Case Study Candidates

這份文件從 6 組 family pairing 的結果中，挑出目前最值得深入分析的 case study 題目。

評選原則：
- 優先選擇在多個 family pairing 下都出現 positive gain 的題
- 優先選擇能代表不同類型 insight 的題目，而不只是一味挑最穩的正例
- 兼顧：
  - 穩定正例
  - 跨 family recurrence
  - 邊界 / 風險案例

## 建議主選 1：`olympiadbench_2127_0029`

### 為什麼選它
這題是目前最乾淨、最穩的「通用正例」。

它在 `6/6` runs 中都出現 positive gain，而且大多數 family pairing 下不只是一個孤立的 `+1`，而是多個 prefix 都有 gain：

- `qwen_to_openai`: `[1, 1, 1, 1, 1, 1]`
- `qwen_to_anthropic`: `[1, 1, 1, 1]`
- `llama_to_openai`: `[1, 0, 1, 0, 1, 1, 0, 1, 0]`
- `llama_to_anthropic`: `[1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]`
- `mistral_to_openai`: `[1, 1, 1, 0, 0, 1, 0]`
- `mistral_to_anthropic`: `[1, 1]`

### 它代表的 insight
- 這題很適合當「takeover value 不是單一 family 偶發現象」的主展示題
- 它支持一種比較強的說法：
  - 當前 prefix 確實包含可以被 stronger model 持續利用的 state
  - 而且這件事跨 small / large family 都成立

### 最適合在文中扮演的角色
- 主案例（main positive example）
- paper / report 裡最先介紹的 case

## 建議主選 2：`olympiadbench_1977_0022`

### 為什麼選它
這題也是 `6/6` 全中，但它比 `2127_0029` 更有「family interaction」的味道。

各組 delta pattern：
- `qwen_to_openai`: `[0, 0, 0, 1]`
- `qwen_to_anthropic`: `[1, 1, 1]`
- `llama_to_openai`: `[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]`
- `llama_to_anthropic`: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- `mistral_to_openai`: `[1, 0, 0, 0]`
- `mistral_to_anthropic`: `[1, 1, 1]`

### 它代表的 insight
這題很適合展示：
- 同一題在不同 family pairing 下，gain 的強度和分布會明顯不同
- 特別是 `llama_to_anthropic` 幾乎整段都正 gain，代表：
  - 某些 pairing 不是只有 early takeover 有價值
  - 而是整條 reasoning trace 都維持脆弱狀態

### 最適合在文中扮演的角色
- family-sensitive positive example
- 用來展示 small family / large family 互動差異

## 建議主選 3：`olympiadbench_1798_0047`

### 為什麼選它
這題不是最乾淨的正例，但正因為它混合了 positive、zero、negative，反而很適合當「邊界案例」。

它在 `5/6` runs 中出現 positive gain，但不同 pairing 下也明顯出現 negative：

- `qwen_to_openai`: `[0, 0, 1, 0, 0]`
- `qwen_to_anthropic`: `[1, -1, 0, -1, -1]`
- `llama_to_openai`: `[0, 0, 1, 0, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]`
- `llama_to_anthropic`: `[-1, -1, 0, 0, -1]`
- `mistral_to_openai`: `[1, 1, 0, 1, 1]`
- `mistral_to_anthropic`: `[0, 0, 1, 0, -1]`

### 它代表的 insight
- takeover 不是單向安全的
- 同一題在某些 pairing 下會有 gain，在另一些 pairing 下反而有 harm
- 這題很適合支持：
  - 為什麼我們不能只看「有沒有 positive」
  - 還要看 family pairing 與 prefix state 的互動

### 最適合在文中扮演的角色
- 邊界案例 / failure-aware example
- 支持「takeover is not universally safe」

## 備選題：`olympiadbench_2666_0010`

### 為什麼可當備選
這題在 `5/6` runs 中也有 positive gain，但 pattern 比 `1798_0047` 更乾淨、也更單純：

- `qwen_to_openai`: `[0, 0, 0, 0]`
- `qwen_to_anthropic`: `[1, 0, 0]`
- `llama_to_openai`: `[0, 0, 1, 0, 0]`
- `llama_to_anthropic`: `[1, 1, 1]`
- `mistral_to_openai`: `[0, 1, 0, 0, 0]`
- `mistral_to_anthropic`: `[0, 0, 0, 1]`

它比較適合當：
- 次級正例
- 或 supplementary example

## 我目前的推薦組合
如果只挑 3 題，我會建議：

1. `olympiadbench_2127_0029`
   - 最穩的主正例
2. `olympiadbench_1977_0022`
   - 最能展示 family interaction 的正例
3. `olympiadbench_1798_0047`
   - 最能展示正負並存、takeover 風險的邊界案例

## 一句話總結
- `2127_0029`：主正例
- `1977_0022`：family interaction 正例
- `1798_0047`：邊界 / 風險案例

這三題放在一起，最能把目前資料裡的核心故事講完整。
