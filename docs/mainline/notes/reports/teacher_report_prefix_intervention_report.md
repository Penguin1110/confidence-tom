# Prefix-Based Intervention 研究進度報告

## 1. 這個研究在做什麼

這條研究線要回答的核心問題不是「哪個模型比較強」，而是：

> 當一個 small model 已經推理到一半時，讓更強的 large model 從當前 prefix 接手，是否真的有價值？而且這個價值是否取決於當下的 reasoning state？

更具體地說，我們要回答三件事：

1. large model 接手是否真的能改善結果。
2. 這個接手收益是否是 state-dependent，也就是不同 prefix 的價值不同。
3. 這個現象是否跨 model family、甚至跨 benchmark 成立。

這個問題和一般的 confidence estimation 不太一樣。這裡真正重要的不是「模型看起來有沒有自信」，而是：

- 哪些 prefix 其實已經進入脆弱狀態。
- 哪些 prefix 如果現在介入，large model 最有機會帶來真實收益。

所以這條研究線更接近：

> prefix-level intervention value / oracle gain estimation

而不是單純的 confidence prediction。

---

## 2. 方法設計

### 2.1 核心框架：full generation -> post-hoc segmentation -> prefix probing

我們目前不用強制 stepwise JSON generation，因為那會扭曲模型原本的推理分佈。現在採用的流程是：

1. 先讓 small model 對原題自然完整生成一條 reasoning trace。
2. 再把這條完整 trace 事後切成幾個 reasoning segments。
3. 對每個切點建立 prefix。
4. 對每個 prefix 比較兩種 continuation：
   - `small continue`：small model 自己繼續做完。
   - `large takeover`：large model 從這裡接手做完。
5. 比較兩者最終表現差異，得到 prefix-level 的 takeover gain。

### 2.2 Oracle Gain 定義

對於某一題的第 `t` 個 prefix，定義：

- `positive`：large 對、small 錯
- `zero`：兩邊都對或兩邊都錯
- `negative`：large 錯、small 對

也可以把它寫成 `Δ_t`：

- `Δ_t > 0`：接手有正收益
- `Δ_t = 0`：接手沒差
- `Δ_t < 0`：接手反而變差

這裡的 `Δ_t` 是整個研究目前最重要的 oracle label。

### 2.3 為什麼這個設計重要

這個設計的價值在於：

- 我們不是只比整題 accuracy。
- 我們在量「在某個中間狀態介入」的真正價值。
- 因此可以把問題從模型排名，轉成 state-dependent intervention problem。

---

## 3. 目前已完成的主要實驗

### 3.1 OlympiadBench family sweep

已完成 6 組 model family pairing，每組 50 題：

- `qwen_to_openai_50`
- `qwen_to_anthropic_50`
- `llama_to_openai_50`
- `llama_to_anthropic_50`
- `mistral_to_openai_50`
- `mistral_to_anthropic_50`

### 3.2 LiveBench reasoning family sweep

已完成 6 組 model family pairing，每組 30 題：

- `livebench_qwen_to_openai_30`
- `livebench_qwen_to_anthropic_30`
- `livebench_llama_to_openai_30`
- `livebench_llama_to_anthropic_30`
- `livebench_mistral_to_openai_30`
- `livebench_mistral_to_anthropic_30`

這批資料已檢查過：

- `30/30` 全完成
- `partials = 0`
- `bad_rows = 0`
- `delta_correctness` 與 `small/large correctness` 一致

### 3.3 Predictor baseline

已經建立 prefix-level predictor dataset，並完成：

- OlympiadBench only baseline
- LiveBench only baseline
- pooled cross-benchmark baseline
- benchmark-aware baseline
- cross-domain evaluation

### 3.4 Representation pilot

已經做過第一版 raw prefix embedding pilot，測試 raw prefix text embedding 是否能直接對齊 intervention value。

### 3.5 Fragility pilot

已完成第一版小型 fragility pilot：

- 24 個 prefix
- 包含 OlympiadBench 與 LiveBench
- 比較：
  - 原始 prefix continuation
  - rule-based normalize 後的 continuation
  - LLM rewrite 後的 continuation

---

## 4. OlympiadBench 的主要結果

### 4.1 總表

| Run | 題數 | Full-trace 正確 | Prefix steps | Positive | Zero | Negative | 至少有一個 positive 的題數 | Small success rate | Large success rate | 平均 segments | 最大 segments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen_to_openai_50` | 50 | 24 | 230 | 21 | 198 | 11 | 8 | 0.504 | 0.548 | 4.60 | 17 |
| `qwen_to_anthropic_50` | 50 | 24 | 213 | 12 | 194 | 7 | 7 | 0.455 | 0.479 | 4.26 | 7 |
| `llama_to_openai_50` | 50 | 17 | 390 | 97 | 276 | 17 | 20 | 0.279 | 0.485 | 7.80 | 25 |
| `llama_to_anthropic_50` | 50 | 20 | 383 | 95 | 283 | 5 | 16 | 0.300 | 0.535 | 7.66 | 25 |
| `mistral_to_openai_50` | 50 | 20 | 229 | 40 | 184 | 5 | 19 | 0.349 | 0.502 | 4.58 | 7 |
| `mistral_to_anthropic_50` | 50 | 15 | 219 | 29 | 188 | 2 | 13 | 0.333 | 0.457 | 4.38 | 6 |

### 4.2 主要發現

1. **這個現象不是單一 family 特例**
- Qwen、Llama、Mistral 三個 small families 都能觀察到 prefix-level takeover gain。

2. **Qwen baseline 最強、分段最乾淨、接手收益較小**
- full-trace correctness 在 3 個 small families 中最高。
- segmentation 最短、最乾淨。
- 代表它本身產生的 reasoning trace 比較穩。

3. **Llama 的 takeover gain 最大，但 state 也最脆弱**
- positive prefixes 非常多。
- `tasks_with_positive` 也最高。
- segmentation 更長、更碎，代表 reasoning state 更容易 externalize 成細粒度、脆弱的中間狀態。

4. **Mistral 是穩定的中間組**
- 不是 baseline 最強，但有乾淨的 segmentation。
- 同時保有可觀察的 takeover signal。

5. **Anthropic 對某些 small families 的接手比較穩**
- 特別是在 `Llama -> Anthropic` 上，negative prefixes 比 `Llama -> OpenAI` 更少。

---

## 5. LiveBench reasoning 的主要結果

### 5.1 總表

| Run | 題數 | Full-trace 正確 | Prefix steps | Positive | Zero | Negative | 至少有一個 positive 的題數 | Small success rate | Large success rate | 平均 segments | 最大 segments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `livebench_qwen_to_openai_30` | 30 | 22 | 116 | 13 | 101 | 2 | 8 | 0.750 | 0.845 | 3.867 | 10 |
| `livebench_qwen_to_anthropic_30` | 30 | 21 | 96 | 15 | 79 | 2 | 7 | 0.677 | 0.812 | 3.200 | 8 |
| `livebench_llama_to_openai_30` | 30 | 10 | 125 | 56 | 68 | 1 | 21 | 0.368 | 0.808 | 4.167 | 8 |
| `livebench_llama_to_anthropic_30` | 30 | 8 | 116 | 61 | 53 | 2 | 24 | 0.328 | 0.836 | 3.867 | 9 |
| `livebench_mistral_to_openai_30` | 30 | 15 | 117 | 36 | 72 | 9 | 24 | 0.470 | 0.701 | 3.900 | 6 |
| `livebench_mistral_to_anthropic_30` | 30 | 10 | 123 | 61 | 61 | 1 | 25 | 0.333 | 0.821 | 4.100 | 6 |

### 5.2 主要發現

1. **這個現象不只存在於 OlympiadBench 數學題**
- 在 `LiveBench reasoning` 這種較短鏈的 open-ended reasoning benchmark 上，也能看到明顯 takeover signal。

2. **LiveBench 的 trace 比較短、結構更規整**
- 平均 segments 約在 `3.2` 到 `4.2`。
- 明顯短於 OlympiadBench。

3. **LiveBench 更支持 `early > late` 這個 pattern**
- pooled 後的 `P(Δ_t > 0)`：
  - step 1：`0.394`
  - step 2：`0.389`
  - step 3：`0.382`
  - step 4+：`0.247`
- 也就是越早的 prefix 越值得接手。

4. **family pattern 大致延續**
- `Qwen`：baseline 強，gain 較小。
- `Llama`：gain 大，而且幾乎整段 reasoning 都偏脆弱。
- `Mistral`：最像乾淨的 intervention timing pattern，early > late 很清楚。

5. **LiveBench 也有跨 family 穩定可救題**
- 不只是 benchmark-specific noise。
- 代表某些 takeover opportunity 是可重複觀察的。

---

## 6. 跨 benchmark 的結論

目前一個很重要的結論是：

> prefix takeover 是跨 benchmark 都存在的現象，但不同 benchmark 的 intervention geometry 不同。

### OlympiadBench
- 長鏈
- segmentation 多
- heterogeneous
- predictor 比較難學

### LiveBench reasoning
- 短鏈
- `early > late` 更乾淨
- pattern 更規整
- predictor 比較容易學

所以不是「只有某個 benchmark 有 takeover」，而是：

> 不同 benchmark 的 reasoning-state geometry 不同，因此 predictor 的難度與訊號形狀也不同。

---

## 7. Predictor 的目前結果

### 7.1 資料規模

目前 prefix predictor dataset 共：

- `2357` 個 prefix rows

### 7.2 主要 baseline 結果

| Experiment | Test AUROC | Test F1 | 解讀 |
| --- | ---: | ---: | --- |
| `structural_only` | 0.478 | 0.252 | 只靠步數/長度不夠 |
| `state_signals` | 0.548 | 0.292 | 加入 drift / confidence-related signals 後有小幅改善 |
| `state_plus_family` | 0.660 | 0.340 | family 差異很重要 |
| `state_plus_family_plus_benchmark` | **0.705** | **0.388** | benchmark-aware 的 pooled predictor 最強 |
| `olympiad_only_state_plus_family` | 0.600 | 0.275 | OlympiadBench 較難預測 |
| `livebench_only_state_plus_family` | **0.869** | **0.779** | LiveBench 結構更乾淨，容易預測 |
| `train Olympiad -> test LiveBench` | 0.663 | 0.561 | 有可轉移 signal |
| `train LiveBench -> test Olympiad` | 0.604 | 0.324 | 轉到 OlympiadBench 比較差 |

### 7.3 目前的解讀

1. **family heterogeneity 很強**
- 不同 small family 的正例率差異很大。

2. **benchmark heterogeneity 也很強**
- 不同 benchmark 的 intervention geometry 不同。

3. **便宜表面特徵不是完全沒用，但不夠**
- step / prefix length / family / benchmark 這些有訊號。
- 但如果只靠表面 signal，仍然不足以完全抓住 takeover value。

4. **這也支持需要更本質的 state signal**
- 例如 fragility。
- 或更 task-conditioned 的 representation。

---

## 8. Representation pilot 的結果

我們做過一版 raw prefix embedding pilot，使用 OpenRouter embedding model 對 prefix text 直接編碼，再用簡單 probe 測試它能不能預測不同 target。

### 8.1 主要結果

- `delta_positive` probe：AUROC `0.396`
- `benchmark_is_livebench`：AUROC `1.000`
- `small_family_is_llama`：AUROC `0.876`

### 8.2 解讀

這個結果很重要，因為它表示：

- raw prefix embedding 很容易學到 benchmark 與 family style
- 但還學不到 takeover value

也就是說：

> 如果只是把 raw prefix text 丟去 embed，representation 會先學到 domain/style geometry，而不是 intervention-relevant state geometry。

這反過來支持：

- benchmark-aware predictor 是必要的
- 若要學到更本質的 state representation，需要更接近 intervention 任務的 supervision，或加入 fragility 這類行為訊號

---

## 9. Fragility pilot 的結果

### 9.1 設計

第一版 fragility pilot 不是用 sampling，而是比較三種 continuation：

1. 原始 prefix 的 small continuation
2. `normalize` 後的 small continuation
3. `rewrite` 後的 small continuation

其中：

- `normalize` 是 rule-based 的表面處理，不是 LLM 改寫。
- `rewrite` 是受約束的 semantic-preserving rewrite，由強模型產生。

### 9.2 目前的結果

- rows：`24`
- `normalized_correctness_changed_rate = 0.292`
- `rewritten_correctness_changed_rate = 0.292`
- `normalized_answer_changed_rate = 0.708`
- `rewritten_answer_changed_rate = 0.792`

按 benchmark 看：

- `livebench_reasoning`
  - normalize correctness change：`0.250`
  - rewrite correctness change：`0.167`

- `olympiadbench`
  - normalize correctness change：`0.333`
  - rewrite correctness change：`0.417`

按 small family 粗看：

- `llama`：normalize `0.125`，rewrite `0.000`
- `mistral`：normalize `0.125`，rewrite `0.250`
- `qwen`：normalize `0.625`，rewrite `0.625`

### 9.3 目前的解讀

這個 pilot 的重點不是數字有多大，而是：

> 小擾動之下，small continuation 的 correctness 真的會翻轉。

這表示 fragility 不是空想，而是可以行為化地量到。

### 9.4 目前的 caveat

這裡有一個方法論 caveat 要先講清楚：

- 現在的 `normalize` 非常保守，24 筆裡有 21 筆和原 prefix 幾乎完全相同。
- 因此它有點混到了 rerun variability。
- 現在的 `rewrite` 雖然 prompt 已經收緊，但仍然有少數例子會把「分析目標」改得更像「已確定事實」。

所以目前比較穩的說法是：

> fragility pilot 已經看到小擾動導致 correctness flip 的行為證據，但還需要更乾淨的 surface-rewrite 與 equivalence control，才能把 semantic fragility 講得更硬。

---

## 10. 目前最重要的 scientific findings

如果要把到目前為止的結果濃縮成幾條最值得報告的發現，我會整理成下面幾點。

### Finding 1

> **prefix takeover gain 是跨 family、跨 benchmark 都存在的現象。**

它不是單一模型 pairing 的偶發怪癖，也不是只有 OlympiadBench 數學題才成立。

### Finding 2

> **takeover value 是 state-dependent 的。**

同一題不同 prefix 的接手價值不同，因此這個問題不能簡化成整題 accuracy 比較。

### Finding 3

> **不同 small family 對應不同的 intervention regime。**

- `Qwen`：baseline 強、trace 穩、marginal gain 較小
- `Llama`：baseline 弱、state 較脆弱、gain 最大
- `Mistral`：中間組，結構乾淨，特別適合看 intervention timing

### Finding 4

> **不同 benchmark 對應不同的 intervention geometry。**

- `OlympiadBench`：長鏈、heterogeneous、較難預測
- `LiveBench`：短鏈、pattern 更規整、較好預測

### Finding 5

> **便宜 signal 有一些預測力，但還不夠。**

family 與 benchmark-aware predictor 已經能到不錯的 AUROC，但 raw text embedding 和單純表面特徵還不足以直接抓住真正的 takeover value。

### Finding 6

> **fragility 已經有初步行為證據。**

某些 prefix 在 very small perturbation 下就會導致 continuation correctness flip，這支持「reasoning state 的穩定性」是值得進一步研究的核心訊號。

---

## 11. 目前限制

### 11.1 資料量仍有限

- OlympiadBench：6 組 x 50 題
- LiveBench：6 組 x 30 題
- fragility pilot：24 rows

這對第一輪研究來說已經夠看 pattern，但還不夠做很強的統計定論。

### 11.2 fragility 還在 pilot 階段

- normalize 太保守，接近 rerun-control
- rewrite 雖然收緊，但還需要更強的 semantic-equivalence control

### 11.3 predictor 目前還是 baseline

- 已經證明 benchmark-aware / family-aware 是有用的
- 但距離真正的 intervention policy 還有一段

### 11.4 raw representation 還沒有對齊 intervention state

- 目前 raw prefix embeddings 主要學到 benchmark 和 family style
- 還沒有自然長出 takeover geometry

---

## 12. 下一步建議

### 短期最值得做

1. **把 fragility probe 做得更乾淨**
- 把現在的 `normalize` 明確分成：
  - rerun-control
  - true surface-rewrite
- 再加 equivalence filter，排除明顯改到內容的 rewrite

2. **把 fragility 特徵接回 predictor**
- 比較：
  - structure only
  - confidence-related
  - fragility-related
  - benchmark/family-aware

3. **補 case study**
- 從穩定可救題與高 variance 題中，挑 2–3 題做深入 qualitative analysis。

### 中期方向

4. **建 benchmark-aware / family-aware intervention policy**
- 從 oracle analysis 走向真正可部署的 prefix-level routing policy。

5. **做 task-conditioned representation learning**
- 不是直接 embed raw prefix text
- 而是讓 representation 對齊 takeover-relevant supervision，例如 `Δ_t`、fragility、risk

---

## 13. 一句話總結

如果要用一句話總結目前這條研究線，我會這樣說：

> 我們目前已經證明：prefix-based takeover gain 不是單一模型或單一 benchmark 的特例，而是跨 family、跨 benchmark 都存在的 reasoning-state phenomenon；但不同 domain 的 intervention geometry 不同，因此要從 oracle analysis 走向可用 policy，還需要 benchmark-aware 與更接近 state stability 的訊號，例如 fragility。

---

## 14. 表格欄位補充說明

這一節補充前面總表裡幾個比較容易誤解的欄位。

### 14.1 `Full-trace 正確`

這表示：

- small model 從零開始完整解一題
- 最後是否答對

這是 **task-level baseline accuracy**，用來描述 small model 本身完整解題時的表現。

### 14.2 `Prefix steps`

這表示：

> 這一組 run 裡，總共抽取了多少個 prefix state，並在這些 prefix 上做 `small continue` 與 `large takeover` 的比較。

它不是題數，而是所有題加總後的 prefix-level rollout 數量。

例如：

- 如果某題被切成 5 個 reasoning segments
- 那這題通常就會產生 5 個 prefix steps
- 所有題加總之後，就是表裡的 `Prefix steps`

因此，`Prefix steps` 越多通常表示：

- 這組 run 的 trace 較長
- segmentation 較細
- 我們對中間狀態的 probing 更密

### 14.3 `Positive / Zero / Negative`

這三個數字都是 **prefix-level** 統計，不是題目數。

- `Positive`：large takeover 對、small continue 錯
- `Zero`：兩邊都對或兩邊都錯
- `Negative`：large takeover 錯、small continue 對

所以它們回答的是：

> 在所有被測試的 prefix state 中，large 接手相對於 small 繼續，有多少次帶來正收益、沒差、或反而變差。

### 14.4 `至少有一個 positive 的題數`

這個欄位是 **task-level** 統計。

它表示：

- 一題只要有任意一個 prefix 出現 positive gain
- 這題就算進來

因此這欄可以回答：

> 有多少題至少在某個中間狀態上，是值得 large model 接手的。

這個數字通常比 raw `Positive` count 更容易直觀理解。

### 14.5 `Small success rate`

這表示：

> 在所有 prefix steps 上，讓 small model 從該 prefix 自己繼續做完時，最終答對的比例。

它是 **prefix-level small-continue accuracy**。

### 14.6 `Large success rate`

這表示：

> 在所有 prefix steps 上，讓 large model 從該 prefix 接手做完時，最終答對的比例。

它是 **prefix-level large-takeover accuracy**。

因此：

- `Small success rate` 與 `Large success rate` 都不是 full-trace accuracy
- 而是針對所有 prefix rollout 的平均表現

這兩個欄位可以直接反映：

> 在中間狀態接手這件事上，large model 整體上是否比 small model 更穩。

### 14.7 `平均 segments` / `最大 segments`

這兩個欄位描述的是每題完整 trace 被切分後的長度。

- `平均 segments`：每題平均被切成幾段
- `最大 segments`：最長的一題被切成幾段

它們能幫助判斷：

- 某個 family 的 reasoning trace 是不是更長
- 某個 benchmark 的 segmentation 是不是更細
- 為什麼某些 run 的 `Prefix steps` 特別多

### 14.8 為什麼這些欄位要一起看

單看某一欄通常不夠。

例如：

- `Positive` 很高，不一定代表這組一定最好，因為它也可能只是 `Prefix steps` 比較多。
- `Full-trace 正確` 很高，也不代表 prefix takeover 就一定沒價值，因為某些題仍然可能存在局部可救 prefix。
- `Large success rate` 高，也要搭配 `Negative` 一起看，才能判斷接手是否同時穩定且安全。

因此，比較穩的讀法通常是一起看：

- `Full-trace 正確`
- `Prefix steps`
- `Positive / Zero / Negative`
- `至少有一個 positive 的題數`
- `Small success rate / Large success rate`
- `平均 segments`

---

## 15. 補充 Insight

除了前面主結果之外，還有幾個補充 insight 值得一起報告。

### 15.1 `Zero` 不是單一類別

`Zero` 其實可以再拆成兩種：

- `both_correct`：small 本來就夠好了，不需要介入
- `both_wrong`：在這個 prefix state 下，換 large 接手也救不起來

這兩種 zero 的意義完全不同。

因此，當我們看到「沒有 gain」時，不能直接解讀成同一件事，而應區分：

- 不需要介入
- 還是介入也救不了

### 15.2 有些題是跨 family 穩定可救的

在 OlympiadBench 與 LiveBench 中，我們都看到一些題目會在多個 family pairing 中反覆出現 positive gain。

這支持一個比較強的說法：

> 某些 takeover opportunity 不是單一模型 pairing 的偶發噪音，而是 benchmark 結構下可重複觀察的 reasoning-state phenomenon。

### 15.3 也有一些題是 `pairing-sensitive`

有些題並不是穩定可救或穩定不可救，而是在不同 pairing 下變異很大。

這表示：

> intervention value 不只依賴題目本身，也依賴 small-family 與 large-family 的組合。

### 15.4 predictor 弱，不代表沒訊號

目前 predictor baseline 雖然還不算很強，但這不表示問題沒有結構。

相反地，這比較像在說：

- class imbalance 存在
- family heterogeneity 很強
- benchmark heterogeneity 也很強
- 單一表面特徵的分離度有限

因此，這個問題本身不是用簡單 heuristic 就能吃掉的，這也正好支持要往更本質的 state signal 走。

### 15.5 raw embedding 先學到的是 benchmark 與 family style

representation pilot 顯示：

- raw prefix embedding 很容易分出 benchmark
- 也容易分出 small family
- 但還抓不到 takeover value

這表示如果不做額外設計，representation 會先學到 domain/style geometry，而不是 intervention-relevant geometry。

### 15.6 hidden states 也支持同樣結論

我們後來又補做了第一版 hidden-state pilot，直接在 `Qwen/Qwen3-14B` 與 `mistralai/Ministral-8B-Instruct-2410` 的 prefix hidden states 上做 task-level linear probe。

主要結果是：

- `Qwen` 與 `Mistral` 的 raw hidden states 幾乎都可以完美分出 `LiveBench` 與 `OlympiadBench`
- benchmark probe 的 AUROC 幾乎接近 `1.0`
- 但對 `small full-trace success` 的 probe 雖然有訊號，仍明顯弱於 benchmark signal

更具體地說：

- `Qwen` 的 success probe 最佳 AUROC 約為 `0.827`
- `Mistral` 的 success probe 最佳 AUROC 約為 `0.843`
- 兩者最佳表現都主要落在中層附近，而不是只出現在最後一層

這表示：

> 即使直接看 model hidden states，最先浮現的仍然是 benchmark / domain geometry，而不是最乾淨的 outcome-relevant geometry。

不過 hidden states 也不是完全沒有 outcome information。

相反地，這批結果比較像是在說：

- outcome-relevant signal 的確已經存在於 hidden states 中
- 只是它和 benchmark/style signal 糾纏得很深
- 如果不做 benchmark control、family control 或 step control，probe 很容易先學到 domain 差異

此外，兩個 family 的 outcome geometry 也不完全一樣：

- `Qwen` 在 `LiveBench` 上的 success signal 特別強
- `Mistral` 在 `OlympiadBench` 上相對更穩

這支持一個更細的說法：

> raw state space 裡確實有 outcome signal，但它如何和 benchmark geometry 纏在一起，會隨 small-family 改變。

因此，hidden-state pilot 並沒有推翻前面的 representation story，反而是用更直接的 state representation 再重複驗證了一次：

- raw representation 的第一主訊號是 benchmark/style
- intervention-relevant signal 存在，但不是最先、也不是最乾淨浮現的幾何

### 15.7 fragility 的真正重點是：很小的擾動也可能改變 correctness

雖然 fragility pilot 還很 early，但現在已經能保守地說：

> 某些 prefix 在很小的表面或語義擾動下，small continuation 的 correctness 會翻轉。

這是目前最接近「reasoning state 穩定性」的初步行為證據。

---

## 16. 下一階段研究 Roadmap

在目前結果基礎上，下一階段最值得推進的方向，可以收斂成三條互相銜接的研究線。

### 16.1 方向一：Early Decision

核心問題是：

> 只看前 `N` 個 prefix steps，能不能已經判斷 small model 這條推理最後會不會成功？

這條線的價值在於：

- 它直接檢查 early information 是否足夠。
- 它比直接做 intervention policy 更乾淨，因為先不碰 cost / routing，只先問 early diagnosis 是否可能。
- 它可以把目前看到的 `early > late` pattern 收斂成更明確的可測問題。

### 16.2 方向二：Minimal Sufficient Prefix

如果 Early Decision 能成立，下一步就不是只問「早期訊息有沒有用」，而是問：

> 最少要看到多少 prefix，才足以穩定預測這條推理的 outcome？

這條線的重點是為每題定義：

- 一個最小 sufficient prefix
- 或一個 critical step

這個概念有幾個好處：

- 很有 novelty
- 很適合做分布分析
- 能自然連到 information bottleneck 與 intervention timing

### 16.3 方向三：重要 step / information bottleneck

第三條線是更細地問：

> 哪些 step 或 segment 才是真正承載 outcome 判斷資訊的瓶頸？

這條線不一定要先做到 token-level attribution。比較穩的做法是先從：

- step-level truncate test
- remove-segment test

開始。這樣比較不容易混進 token-level prompt artifact。

### 16.4 推進順序

如果依照風險與產出效率排序，我會建議：

1. 先做 `Early Decision`
2. 再升級成 `Minimal Sufficient Prefix`
3. 最後再做 `important step / information bottleneck`

也就是：

- 先確認早期資訊到底能不能預測 outcome
- 再定義「最早何時已經足夠」
- 最後再問「真正重要的是哪一步」

---

## 17. Early Decision：具體實驗設計

### 17.1 核心研究問題

Early Decision 要回答的是：

> 給定 small model 的前 `t` 個 prefix steps，我們能不能預測 small model 最後 full-trace 會成功還是失敗？

這個版本和直接問 large takeover value 不同。它更像在問：

- small model 的 failure signal 是否能在早期被看出來
- reasoning outcome 是否存在 early diagnosability

### 17.2 第一版最乾淨的 label

我建議第一版先用這個 label：

- `y = 1`：small model 的 full-trace 最後答對
- `y = 0`：small model 的 full-trace 最後答錯

原因是：

- label 乾淨
- 不會混進 judge quality
- 也不會混進 intervention cost

因此第一版問題是：

> prefix `h_t` 能不能預測 small full-trace outcome？

### 17.3 實驗資料怎麼來

對每題：

1. 取 small model 的 full trace
2. 做 segmentation
3. 對每個 prefix `t = 1, 2, 3, ...` 建一個 observation
4. observation 的輸入是：
   - prefix text
   - step index
   - prefix length
   - family / benchmark metadata
5. label 是：
   - 這題的 small full-trace 最終是否正確

這樣就能得到一組 early-decision dataset。

### 17.4 可以怎麼做 predictor

第一版不一定要很複雜，可以做三層 baseline：

1. `structure only`
- step index
- prefix tokens
- current segment tokens

2. `structure + confidence-related`
- hedge
- self-correction
- backtracking

3. `structure + state + family + benchmark`
- 對齊目前我們已有的 predictor pipeline

如果要加 LLM judge 版，建議放第二版：

- 給大模型 prefix
- 問它：
  - 「根據這段 prefix，你認為這條 small reasoning 最後會成功嗎？」

但第一版最好先做非 judge 版本，這樣比較乾淨。

### 17.5 評估方式

最基本先看：

- AUROC
- accuracy
- F1

但更重要的是看「隨 prefix 深度變化的可預測性」：

- `t = 1` 時能不能預測
- `t = 2` 時能不能預測
- `t = 3` 時能不能預測

所以比較值得畫的是：

- x 軸：prefix step `t`
- y 軸：prediction quality（例如 AUROC）

這樣就能直接回答：

> 到第幾步時，資訊已經足夠？

### 17.6 這條線最可能得到的 insight

如果實驗結果顯示：

- `t = 1` 或 `t = 2` 就已經有不錯的 AUROC

那可以支持：

> early information is already sufficient to diagnose reasoning success or failure.

如果不同 benchmark / family 的 critical step 不同，則可以進一步得到：

- `LiveBench` 可能更早可判斷
- `OlympiadBench` 可能需要更深 prefix
- `Llama` 可能 failure signal 會提早暴露
- `Qwen` 可能更晚才分化

### 17.7 和 Minimal Sufficient Prefix 的連接

Early Decision 做完之後，就可以自然接到下一步：

- 對每題找最小 `t`
- 使得從該步開始，prediction 已經穩定

這就是 Minimal Sufficient Prefix。

所以 Early Decision 不只是獨立分析，也是一個通往：

- critical step
- information bottleneck
- intervention timing

的基礎。

### 17.8 為什麼這條線值得先做

我認為這是三條線裡最該先做的，因為：

1. 它和目前 prefix dataset 直接對齊
2. 不需要新增太多基礎設施
3. 問題直觀、好講、好畫圖
4. 做完後很自然就能接 Minimal Sufficient Prefix

一句話說，這條線是在問：

> 在 reasoning 剛開始不久時，我們是否就能預見這條 small reasoning 會不會走向成功？

---

## 18. Minimal Sufficient Prefix：具體實驗設計

### 18.1 核心研究問題

Minimal Sufficient Prefix 要回答的是：

> 對於一條 small-model reasoning trace，最少要看到多少 prefix，才足以穩定預測它最後的 outcome？

這條線比 Early Decision 更進一步。

- Early Decision 問的是：前幾步能不能預測？
- Minimal Sufficient Prefix 問的是：**最早從哪一步開始，資訊已經足夠？**

因此，這條線的目標不是只得到一條平均 AUROC 曲線，而是：

- 為每一題找到一個最小 sufficient `t`
- 研究這個 `t` 的分布與結構

### 18.2 最小 sufficient prefix 的直覺

假設某題有 prefix：

- `t = 1`
- `t = 2`
- `t = 3`
- `t = 4`
- `t = 5`

如果從 `t = 2` 開始，模型對「這題最後會成功 / 失敗」的預測已經穩定，而且之後再看到更多 prefix，判斷也不再改變，
那我們就可以說：

- 這題的 minimal sufficient prefix 是 `t = 2`

這個 `t` 可以理解成：

- critical step
- earliest decisive prefix
- outcome information first becomes sufficient here

### 18.3 第一版如何定義「sufficient」

第一版不需要太複雜，我建議先用一個保守定義：

對某題的 prefix 預測序列 `\hat{y}_1, \hat{y}_2, ..., \hat{y}_T`，找最小的 `t*`，滿足：

1. `\hat{y}_{t*}` 已經等於最終正確 label
2. 從 `t*` 到 `T` 的預測不再改變

也就是：

> 第一個「之後都不再翻轉，且最後是對的」的 prefix。

這個版本最乾淨，因為它不需要先引入 confidence threshold。

### 18.4 第二版如何定義「穩定」

如果第一版太嚴格，可以做第二版較柔性的定義，例如：

- 從 `t*` 開始，預測機率始終高於某門檻
- 從 `t*` 開始，後面最多允許一次小幅波動
- 或用 moving average / smoothed confidence 定義穩定區

但我建議第一輪先做離散版，因為比較好解釋。

### 18.5 需要什麼資料

這條線可以直接建立在 Early Decision 資料上。

對每題，我們已經有：

- 一串 prefix `h_1, h_2, ..., h_T`
- 每個 prefix 的 predictor output
- 該題的最終 outcome label

所以不需要另外蒐集新資料。

我們只需要：

1. 先把 Early Decision predictor 跑完
2. 對每題整理 prefix-level prediction trajectory
3. 再在每題上找最小 sufficient `t*`

### 18.6 可以做哪些分析

#### A. `t*` 的整體分布

最基本先看：

- 有多少題在 `t = 1` 就足夠
- 有多少題在 `t = 2`
- 有多少題要到很後面才足夠
- 有多少題根本不存在穩定 sufficient prefix

這樣會得到一個：

- minimal sufficient prefix distribution

#### B. 按 benchmark 比較

例如：

- `LiveBench` 的 `t*` 是否普遍更早
- `OlympiadBench` 的 `t*` 是否更晚、分布更長尾

這可以直接對應我們現在看到的：

- `LiveBench` 結構更規整
- `OlympiadBench` 更 heterogeneous

#### C. 按 small family 比較

例如：

- `Llama` 的 failure signal 是否更早暴露
- `Qwen` 是否需要更長 prefix 才看得清楚
- `Mistral` 是否最乾淨地呈現 early-decision pattern

#### D. 按 task regime 比較

把我們前面 clustering / task structure 的結果接回來，可以看：

- stable-recoverable 題是否有更早的 `t*`
- negative-risk 題是否更晚才穩定
- pairing-sensitive 題是否根本沒有穩定的 sufficient prefix

### 18.7 這條線最可能帶來的 insight

如果這條線成立，我們就可以講：

> outcome-relevant information is often concentrated in an early minimal prefix, rather than being uniformly distributed across the whole reasoning trace.

這個 insight 很重要，因為它表示：

- 不是整條 trace 每一段都同樣重要
- 有些題很早就已經暴露成敗訊號
- 後面很多 token 可能只是展開，不是新增關鍵資訊

這就自然連到：

- information bottleneck
- critical step
- early intervention timing

### 18.8 為什麼這條線有 novelty

這條線的 novelty 在於，它不是只說：

- early prefix 有用

而是說：

- 每題可能有一個 **minimal sufficient prefix**
- 從那一步開始，outcome 已經可被穩定預測

這比單純畫一條 AUROC vs t 曲線更有概念，也更適合形成 paper-style statement。

### 18.9 實作上的最小版本

如果要快速做第一版，我建議：

1. 先完成 Early Decision predictor
2. 對每題收集 `t = 1..T` 的 prediction trajectory
3. 用離散定義找 `t*`
4. 畫三張圖：
   - 全體 `t*` 分布
   - benchmark-by-benchmark `t*` 分布
   - family-by-family `t*` 分布

這樣就已經足夠回答：

- minimal sufficient prefix 是否存在
- 它在不同 domain / family 上如何分布

### 18.10 和後續方向的關係

Minimal Sufficient Prefix 做完後，可以自然接到：

- important step / bottleneck analysis
- intervention timing policy
- budgeted routing

因為一旦知道某題在 `t*` 之後資訊就夠了，就可以問：

- 是否真的有必要看到更後面的 trace？
- 是否可以在 `t*` 就做 decision？
- 是否能據此設計更便宜的 early intervention policy？

一句話說，這條線是在問：

> 一條 reasoning trace 裡，最早從哪裡開始，Outcome 已經可知？

---

## 19. Early Decision 的第一版結果

我們已經不只寫出 Early Decision 的實驗設計，也已經跑出第一版 baseline。

### 19.1 設定

這一版的 target 是：

- `y = 1`：small model 的 full-trace 最後答對
- `y = 0`：small model 的 full-trace 最後答錯

也就是：

> 只看某個 prefix，預測這條 small reasoning 最後會不會成功。

### 19.2 主要結果

| Experiment | Test AUROC | Test F1 | 解讀 |
| --- | ---: | ---: | --- |
| `structural_only` | 0.446 | 0.571 | 只靠步數與長度不夠 |
| `state_signals` | 0.559 | 0.607 | 加入 prefix-level state signal 後有提升 |
| `state_plus_family` | **0.592** | **0.587** | pooled setting 裡目前最好 |
| `state_plus_family_plus_benchmark` | 0.576 | 0.580 | benchmark one-hot 沒再明顯提升 |
| `olympiad_only_state_plus_family` | 0.504 | 0.595 | OlympiadBench 幾乎接近難以預測 |
| `livebench_only_state_plus_family` | **0.880** | **0.739** | LiveBench 的 early diagnosability 很強 |
| `train Olympiad -> test LiveBench` | 0.804 | 0.643 | Olympiad 訓練的 signal 有部分能轉到 LiveBench |
| `train LiveBench -> test Olympiad` | 0.545 | 0.568 | 轉回 OlympiadBench 明顯較弱 |

### 19.3 step-by-step 看

用 `state_plus_family` 這個 pooled baseline 看：

- step 1：AUROC `0.596`
- step 2：AUROC `0.615`
- step 3：AUROC `0.676`
- step 4+：AUROC `0.533`

用 `livebench_only_state_plus_family` 看：

- step 1：AUROC `1.000`
- step 2：AUROC `0.927`
- step 3：AUROC `0.855`
- step 4+：AUROC `0.814`

### 19.4 解讀

這個結果支持：

> outcome diagnostability 的確存在，但它高度 benchmark-dependent。

更具體地說：

- `LiveBench` 上，前 `1–2` 步就有很強的 outcome signal。
- `OlympiadBench` 上，early diagnosability 弱很多。
- 因此 early diagnosis 不是不存在，而是不同 benchmark 的可判斷性差很多。

---

## 20. Minimal Sufficient Prefix 的第一版結果

在 Early Decision 基礎上，我們進一步分析：

> 對每一題來說，最早從哪一步開始，prediction 已經穩定到足以判斷最終 outcome？

### 20.1 離散版 MSP

第一版 MSP 使用一個保守定義：

- 找最小的 `t*`
- 使得從 `t*` 開始 prediction 不再翻轉
- 且 `\hat{y}_{t*}` 已經等於正確 label

### 20.2 主要結果

- test tasks：`96`
- 有找到 MSP 的 tasks：`45`
- coverage：`0.469`
- mean MSP：`1.96`

分布：

- step 1：`27`
- step 2：`7`
- step 3：`3`
- step 4：`3`
- step 5：`4`
- step 6：`1`

### 20.3 benchmark 差異

`LiveBench`：

- coverage：`0.556`
- mean MSP：`1.1`

`OlympiadBench`：

- coverage：`0.449`
- mean MSP：`2.2`

### 20.4 family 差異

`Qwen`：

- coverage：`0.594`
- mean MSP：`1.21`

`Llama`：

- coverage：`0.500`
- mean MSP：`2.69`

`Mistral`：

- coverage：`0.313`
- mean MSP：`2.2`

### 20.5 解讀

這表示：

> Minimal Sufficient Prefix 不只是概念，它在資料裡真的長出來了，而且其位置會隨 benchmark 與 small family 改變。

更具體地說：

- `LiveBench` 上，outcome-relevant information 通常很早就夠了。
- `OlympiadBench` 上，MSP 更晚、也更分散。
- `Qwen` 較早定型。
- `Llama / Mistral` 通常需要更深 prefix。

---

## 21. Probability-Stable MSP

為了讓 MSP 的說法更保守，我們又做了更嚴格的版本：

> 找最早的 `t*`，使得對正確 label 的機率從該步開始一直高於固定門檻。

### 21.1 主要結果

當 threshold = `0.6`：

- overall coverage：`31 / 96 = 0.323`
- mean MSP：`1.58`

其中：

- `LiveBench` coverage：`0.500`
- `OlympiadBench` coverage：`0.282`

當 threshold = `0.7`：

- overall coverage：`9 / 96 = 0.094`

當 threshold = `0.8`：

- overall coverage：`3 / 96 = 0.031`

### 21.2 解讀

這表示：

> 離散版 MSP 在不少題上存在；但若要求高信心且穩定的 sufficiency，coverage 會快速下降。

也就是：

- early sufficiency 是存在的
- 但 high-confidence early sufficiency 比較稀少
- 而且主要集中在較規整的 benchmark，例如 `LiveBench`

---

## 22. Bottleneck Proxy 與 Alignment

在 MSP 之後，我們又往前推兩步：

1. 看 decisive step 大約落在哪裡
2. 看 early diagnosis 與 early takeover 是否真的對齊

### 22.1 `cross70` 的直覺

這裡我們用：

- `MSP`：最早穩定可判斷的步驟
- `cross70`：最早對正確 outcome 有 `>= 0.7` 信心的步驟

其中 `cross70` 可以當成：

> 較高信心 decisive step 的 proxy

### 22.2 Bottleneck proxy 的主要結果

`first_cross_70` 整體分布：

- step 1：`8`
- step 2：`17`
- step 3：`4`
- step 4：`1`

平均：

- overall：`1.93`
- `LiveBench`：`1.6`
- `OlympiadBench`：`2.1`

按 family 看：

- `Llama`：`1.79`
- `Mistral`：`1.56`
- `Qwen`：`2.71`

### 22.3 解讀

這表示：

> 中度到高信心的 decisive step 通常落在 step `1–2`，但其位置會隨 benchmark 和 small family 改變。

其中：

- `LiveBench` 更早形成穩定判斷
- `OlympiadBench` 更晚
- `Qwen` 雖然常有早期方向，但高信心 crossing 反而更晚

### 22.4 Early diagnosis 與 takeover 對齊

如果只看那些至少有一個 positive takeover 的題，則：

- `MSP <= earliest positive` rate：`0.667`
- `cross60 <= earliest positive` rate：`0.895`
- `cross70 <= earliest positive` rate：`0.692`

按 benchmark 看：

`LiveBench`：

- `MSP <= positive`：`0.800`
- `cross70 <= positive`：`0.875`

`OlympiadBench`：

- `MSP <= positive`：`0.571`
- `cross60 <= positive`：`1.000`
- `cross70 <= positive`：`0.400`

### 22.5 解讀

這支持一個很重要的說法：

> 在相當多有正 takeover 的題上，early outcome diagnostics 會早於或同步於 takeover opportunity。

但不同 benchmark 的對齊型態不同：

- `LiveBench` 上，診斷和 takeover 更同步、也更早。
- `OlympiadBench` 上，較弱的診斷 signal 可能很早出現，但高信心、穩定的診斷通常更晚成熟。

---

## 23. Case Study 與 Segment Removal

### 23.1 Case study

我們已經補了四種代表 case：

1. `aligned`：早診斷與早 takeover 同步
2. `misaligned`：positive takeover 比穩定診斷更早
3. `no stable MSP`：一路沒有穩定可知點
4. `late bottleneck`：decisive information 很晚才出現

這表示：

> aggregate pattern 不只是統計假象，而是真能回到題目層級看到不同 regime。

### 23.2 Segment removal bottleneck

我們也做了第一版 segment removal：

- 對有 `cross70` 的題
- 刪掉 decisive segment
- 重算 Early Decision probe 的 correct-label probability
- 並和一個 control removal 比較

主要結果：

- evaluated tasks：`22`
- mean decisive drop：`0.098`
- mean control drop：`0.009`
- mean drop gap：`0.089`

其中：

- `OlympiadBench` mean drop gap：`0.073`
- `LiveBench` mean drop gap：`0.189`

### 23.3 解讀

這版雖然還是 first-pass probe，但結果支持：

> `cross70` 對應的 step 並不是任意的，它更像真的承載了 outcome-relevant information 的 bottleneck。

因此，這條線目前已經形成一個相對完整的故事：

- 先看到 early diagnosability
- 再找到 MSP
- 再用 `cross70` 逼近 decisive step
- 最後用 removal test 提供初步 bottleneck evidence

---

## 24. Routing Simulation Baseline

在完成 Early Decision、MSP、alignment 與 bottleneck proxy 之後，我們已經把這些診斷訊號接到第一版 prefix-level routing policy。

### 24.1 問題設定

這一版 policy 的想法很直接：

- 用 Early Decision predictor 輸出 `P(small failure)`
- 在前幾個 prefix 中檢查這個 failure probability
- 一旦超過門檻，就在該 prefix 呼叫 `large takeover`

也就是：

> 不是等 oracle 事後分析，而是用 prefix-level diagnosis signal 直接驅動 takeover decision。

### 24.2 模擬方式

第一版 routing simulation 做的是：

1. 在 test tasks 上使用 `state_plus_family` Early Decision predictor
2. 對每題依序看 step `1, 2, 3, ...`
3. 若 `P(small failure)` 超過門檻，則在最早可用步驟 route 到 large
4. 若沒有超過門檻，則維持 small full-trace 結果
5. 同時用同樣 `max_step` 預算下的 oracle routing 當 upper bound 比較

### 24.3 最佳設定

目前最好的設定是：

- `max_step = 2`
- `failure_threshold = 0.6`

在 `96` 個 test tasks 上：

- small baseline accuracy：`0.438`
- routing policy accuracy：`0.583`
- gain over small：`+0.146`
- oracle accuracy under same budget：`0.656`
- gap to oracle：`0.073`
- route rate：`0.625`
- mean trigger step：`1.383`

### 24.4 benchmark breakdown

`OlympiadBench`：

- small baseline：`0.449`
- policy accuracy：`0.500`
- gain：`+0.051`
- route rate：`0.603`

`LiveBench`：

- small baseline：`0.389`
- policy accuracy：`0.944`
- gain：`+0.556`
- route rate：`0.722`

### 24.5 解讀

這個結果非常重要，因為它表示：

> Early diagnosis 不只是分析上的旁支，它已經足以支撐一個有實際收益的 prefix-level routing baseline。

而且這和前面所有分析是連續一致的：

- `LiveBench` 上，早期訊號更乾淨，因此 routing 效果非常強。
- `OlympiadBench` 上，雖然也有 gain，但提升幅度明顯較小，反映了它更 heterogeneous、也更難早期判斷。

### 24.6 目前能怎麼定位這一步

因此，如果用整條研究線來看，目前已經不只是：

- 驗證 oracle gain 現象
- 建 baseline predictor
- 做 state analysis

而是已經走到：

> **第一版 policy prototype：用 early diagnosis signal 做 prefix-level routing simulation。**

這也表示接下來最自然的下一步，不是再驗證「有沒有訊號」，而是：

- 如何把 routing policy 做得更強
- 如何把 risk 也納進 decision
- 如何把這條線整理成更完整的 deployment story

### 24.7 hidden-state routing 與 error overlap

我們後來也把 routing policy 直接接到 raw hidden states 上，想問：

> 如果不用 prefix text / state features，而是直接用 small model 的 internal state，能不能做出更好的 routing？

第一版 hidden-state routing 主要做了三件事：

1. `layer sweep`
2. `mean pooled` vs `last token` pooling comparison
3. 直接預測 `positive takeover`，而不只預測 `small failure`

主要結果是：

- 在 `small failure` routing 上，hidden-state policy 可以追平 text-based policy
- 但即使做了 layer sweep，最佳 hidden-state policy 仍然沒有超過 text baseline
- `mean pooled` 明顯比 `last token` 更好
- 若直接預測 `positive takeover`，則 text/state-feature policy 反而明顯優於 hidden-state policy

更具體地說，在 `qwen + mistral` 子集上的最佳結果為：

- text failure-routing accuracy：`0.562`
- hidden-state failure-routing accuracy：`0.562`
- text positive-routing accuracy：`0.562`
- hidden-state positive-routing accuracy：`0.547`

因此，這條線目前支持的比較不是：

> raw hidden states 是更強的 routing signal

而是：

> raw hidden states 確實含有可用的 routing information，但在目前這個 setup 下，它們還不是比 text/state features 更好的 policy signal。

另外一個值得注意的發現是，兩種 policy 的錯誤模式非常一致。

- 在 failure-based routing 下，text policy 與 hidden-state policy 錯的是完全同一批 task
- task-level error overlap 是 `9 / 9`
- Jaccard overlap = `1.0`

在 direct `positive takeover` routing 下，兩者也高度重疊：

- text policy 錯 `9` 題
- hidden-state policy 錯 `10` 題
- 其中有 `9` 題是共同錯誤
- Jaccard overlap = `0.9`

這表示：

> hidden-state policy 目前並不是在修正 text policy 的另一組盲點；它更像是在重複捕捉同一批困難 task 上的訊號，只是有時會再多帶入一些噪音。

因此，比較穩的定位方式是：

- hidden-state 線目前更適合放在 representation / state-geometry analysis
- text/state-feature 線目前仍是比較強、也比較實用的 routing baseline
