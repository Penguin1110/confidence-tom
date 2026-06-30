# LiveBench Re-entry 完整研究與解讀（以「為什麼 full correct 但 re-entry 不行」為核心）

## 核心研究問題

這次真正想回答的，不是「哪個步數最好」，而是下面這個更根本的問題：

> **如果同一條 full reasoning trace 最後已經能答對，為什麼把它中途的 prefix 拿出來做 re-entry，模型卻常常接不回正解？**

直覺上，如果 full trace correct，代表這條推理路徑裡最終確實累積到了足夠資訊。那麼理論上，某些 prefix 應該也已經包含了「可重用的正確狀態」，re-entry 應該可以從那裡接回去。

但結果不是這樣。

所以這份分析要解釋的其實是三件事之間為什麼會不一致：

1. `full-trace correctness`
2. `prefix-level reusability`
3. `correctness-state persistence`

---

## 一句話主結論

目前最穩的總結是：

> **full trace 能答對，不代表中途 prefix 已經形成穩定、可重用的正確 state；而 prefix 一度可重用，也不代表完整 trace 會把這個 state 保留到最後。**

也就是說，問題不只是「資訊有沒有出現」，而是：

- 它是不是以可重用的形式存在
- 它能不能脫離原始 trajectory 被重新啟動
- 它出現之後能不能被後續 continuation 保住

---

## 分析範圍

- 資料來源：`outputs/results/imported/results-livebench-reentry-a07127a`
- 使用範圍：`reentry/*_no_outliers`
- hidden-state 來源：`probe/*/reentry_probe_rows.jsonl`
- family：
  - `gemma4`
  - `ministral`
  - `mistral7`
  - `qwen25`
  - `qwen3`
- task 數：`141`

---

## 一、先看最直接的 anomaly：full correct 為什麼沒有自然分解成 reusable prefixes？

### 1.1 Full-correct task 裡，超過一半不是 stable-success

這批資料裡：

- full-correct task：`44`
- 其中 `stable-success`：`21`
- 其中 `late-success`：`23`

也就是說：

> **有 `52.3%` 的 full-correct task，並沒有表現成「早期、穩定、可重用」的 prefix 結構。**

這很關鍵。因為如果 full success 只是單純靠資訊逐步累積，那應該會比較常看到一個乾淨的 sufficiency point，或至少一旦接近正解，prefix re-entry 也會變穩。

但現在不是。這表示：

- full-trace success 不等於 early reusable state
- 原始完整 trajectory 很可能提供了額外的狀態延續條件
- 把 prefix 抽離原始軌跡後，那個條件可能被破壞了

### 1.2 很多 full-correct task，到很後面才 first-correct

在 `44` 個 full-correct task 裡：

- `17` 題的 `first_correct_frac > 0.5`
- 也就是有 `38.6%` 的 full-correct 題目，要到 trace 後半段才第一次出現可重用的 correct prefix

這直接否定了「full correct 就代表前面早就有足夠資訊」這個強版本直覺。

比較貼近資料的說法是：

> full trace 裡的正確性，常常不是早早就以可重用的形式存在，而是要沿著原始 trajectory 走到夠後面，才慢慢形成可被 re-entry 啟動的 state。

### 1.3 還有不少 full-correct task，雖然最後答對，但中途很不穩

在 full-correct task 裡：

- `16/44` 的 `local_correct_rate < 0.5`
- 也就是有 `36.4%` 的 full-correct 題目，即使最終答對，中途大多數 prefix 其實都還不能穩定接回正解

這一點非常重要，因為它說明：

> **full-trace success 可以建立在一條對 prefix re-entry 很不友善的 dynamics 上。**

模型也許只有在原始連續 rollout 的內部狀態下，才能順利維持那條成功路徑；一旦把 prefix 拔出來重啟，就不一定還在同一個 basin 裡。

### 1.4 但最末端通常又能接回來

同時，full-correct task 裡：

- `last_small_correct = 0.977`
- 也就是 `44` 題裡只有 `1` 題最後一個 prefix 還接不回正解

這代表：

> full-trace success 並不是完全不具可重用性，而是它的可重用性往往出現得很晚，而且常常要等到已經非常接近原始成功終點時才變穩。

所以真正的 anomaly 不是「永遠接不回來」，而是：

> **為什麼 full success 的可重用性要那麼晚才顯現？**

---

## 二、另一個方向的 anomaly：為什麼 prefix 一度可重用，full trace 最後卻還是錯？

這就是 `late-failure`。

在 `97` 個 full-wrong task 裡：

- `26` 題屬於 `late-failure`
- 佔 full-wrong 的 `26.8%`

也就是：

> **超過四分之一的 full-wrong 題目，中途其實曾出現過可重用的 correct prefix。**

這代表另一種不一致：

- `prefix reusability`
- 不等於 `final preservation`

也就是說，模型不是單純「知道/不知道」：

- 有時它中途其實已經進過可用 state
- 但完整原始 trace 沒把這個 state 保到最後

這正是 `late-failure` 的研究價值。

所以現在其實有兩種不同 violation：

1. `full-correct but prefix-not-reusable-early`
2. `prefix-reusable but full-not-preserved`

前者對應 `late-success`，後者對應 `late-failure`。

這兩者一起說明：

> **full correctness、prefix reusability、state persistence 是三件相關但不等價的事。**

---

## 三、taxonomy 在這個 framing 下要怎麼重講

這四類 taxonomy 不是只是描述表面行為，而是在拆不同類型的 mismatch。

| Category | Count | Share | Mean first-correct frac | Mean local correct rate | Last-small-correct rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| stable-success | 21 | 0.149 | 0.153 | 0.884 | 1.000 |
| late-success | 23 | 0.163 | 0.710 | 0.326 | 0.957 |
| late-failure | 26 | 0.184 | 0.467 | 0.332 | 0.269 |
| persistent-failure | 71 | 0.504 | 1.000 | 0.000 | 0.000 |

### `stable-success`

這是最符合直覺的正常情況：

- full trace correct
- correct prefix 很早出現
- 後續也穩

這類代表：

> full success 和 prefix reusability 是一致的。

### `late-success`

這是這次最值得重視的 anomaly 之一：

- full trace correct
- 但 correct prefix 很晚才出現，或中途其實不穩

在這批資料中：

- `late-success` 的 `mean first-correct frac = 0.710`
- `mean local correct rate = 0.326`
- 但 `last-small-correct = 0.957`

這表示：

> 這些題目最終能成功，但中途多數時候並沒有形成穩定、可重用的 correct state；通常要到非常後面，re-entry 才終於能接得回去。

所以 `late-success` 不是「比較晚成功」而已，而是：

> **full success 並沒有早期顯化成 reusable partial state。**

### `late-failure`

這是另一種反方向 anomaly：

- 中途一度有可重用 prefix
- 但 full trace 最終沒保住

在這批資料中：

- `mean first-correct frac = 0.467`
- `mean local correct rate = 0.332`
- `last-small-correct = 0.269`

也就是：

> 它常常比 `late-success` 更早 first-correct，但最後反而比較保不住。

所以 `late-failure` 的核心不是能力不足，而是：

> **state preservation failure。**

### `persistent-failure`

這類比較像 base competence 不足：

- full trace 不行
- prefix re-entry 也不行

這類不是最反常的，但提供了對照組。

---

## 四、最重要的觀察：`late-success` 和 `late-failure` 的差別，不是誰比較早

這是整份分析的核心。

如果只看表面直覺，你可能會以為：

- `late-success` 應該是「比較早就接近正解」
- `late-failure` 應該是「一直都比較差」

但資料不是這樣。

aggregate 上：

- `late-success first-correct frac = 0.710`
- `late-failure first-correct frac = 0.467`

也就是：

> **`late-success` 平均其實比 `late-failure` 更晚才 first-correct。**

但同時：

- `late-success last-small-correct = 0.957`
- `late-failure last-small-correct = 0.269`

所以真正差別是：

> `late-success` 雖然常常更晚才出現 correct prefix，但一旦形成，就比較能保到最後；`late-failure` 則常常較早碰到局部正確，卻沒有把它維持住。

這對研究問題的意義非常大：

> 問題不是「資訊夠不夠早出現」，而是「出現後是不是以可持續、可重啟的形式存在」。

---

## 五、Direct competence：re-entry 比較像 rescue，不像創造能力

如果把 full trace correctness 和 re-entry 對照：

| Full trace correct? | Tasks | Any small correct | Last small correct | Mean first-correct frac | Mean local correct rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| True | 44 | 1.000 | 0.977 | 0.444 | 0.593 |
| False | 97 | 0.268 | 0.072 | 0.857 | 0.089 |

這代表：

- full trace 本來就會的題，幾乎一定至少能找到某個可用 prefix
- full trace 本來不會的題，re-entry 很少真的把它救回來

因此，re-entry 比較像：

> **competence-conditioned rescue**

而不是：

> capability creation

這個觀察會影響你對 anomaly 的解讀：

- `late-success` 不是「模型突然學會了」
- `late-failure` 也不是「模型完全不會」

而是模型已經有某種 competence-related state，只是它：

- 不夠早可重用
- 或不夠穩能保留

---

## 六、Hidden-state probe：hidden state 比較像在讀「當下 prefix 是否可局部接對」

用 hidden state 做 task-level probe，aggregate 結果如下：

| Pooling | Target | Weighted AUROC |
| --- | --- | ---: |
| mean_hidden | full_trace_correct | 0.370 |
| mean_hidden | reentry_exact_correct | 0.524 |
| mean_hidden | late_failure_task | 0.627 |
| last_token_hidden | full_trace_correct | 0.463 |
| last_token_hidden | reentry_exact_correct | 0.757 |
| last_token_hidden | late_failure_task | 0.566 |

這裡最重要的訊息是：

1. `last_token_hidden` 對 `reentry_exact_correct` 的訊號最強，weighted AUROC `0.757`
2. hidden state 對 `full_trace_correct` 的整體訊號反而比較弱

這表示 hidden state 目前最像在反映：

> **這個 prefix 當下有沒有處在一個可局部接對的 state**

而不是穩定反映：

> 這整條 full trace 最後會不會成功

這和主研究問題很一致。因為我們要解釋的正是：

- full success
- 為什麼沒有自然對應到 prefix reusability

而 probe 結果剛好顯示：

> hidden state 對 local re-entry state 比對 full success 更敏感。

這也暗示：

> full-trace success 所依賴的，不只是某個靜態 hidden-state snapshot，而更像是 rollout dynamics 本身。

---

## 七、Late-success vs Late-failure 的 hidden-state 診斷：可分，但不是靠「誰比較早」

如果只看已經出現某種 local correctness 的題，也就是 `late-success` vs `late-failure`：

| Pooling | Weighted AUROC |
| --- | ---: |
| mean_hidden | 0.490 |
| last_token_hidden | 0.564 |

aggregate 訊號中等，但 `qwen3` 很強：

| Family | Pooling | AUROC |
| --- | --- | ---: |
| qwen3 | mean_hidden | 0.764 |
| qwen3 | last_token_hidden | 0.880 |

更重要的是 `qwen3` 的 category stats：

| Category | Mean first-correct frac | Mean local correct rate | Mean last-small-correct |
| --- | ---: | ---: | ---: |
| late-success | 0.900 | 0.246 | 1.000 |
| late-failure | 0.423 | 0.511 | 0.600 |

這再次說明：

- `late-success` 更晚才形成 correct prefix
- 但它一旦形成，比較能保到最後

所以 hidden-state 診斷支持的不是「早 enough」故事，而是：

> **state persistence / state survivability**

---

## 八、Trajectory geometry：為什麼會出現「full correct 但 re-entry 不穩」？

這部分最接近你真正想問的 mechanistic 問題。

我們把 hidden states 當成 prefix trajectory，而不是單點，去看：

- consecutive cosine
- cosine-to-final
- mean displacement
- max displacement

aggregate 的 `late-success - late-failure` 差值如下：

### `mean_hidden`

- first-correct frac: `+0.279`
- local correct rate: `-0.029`
- last-small-correct: `+0.651`
- consecutive cosine: `+0.003`
- cosine-to-final: `+0.021`
- mean displacement: `-1.205`
- max displacement: `-7.258`

### 這些數字怎麼讀

它們的含義是：

- `late-success` 並沒有比較早形成正確 prefix
- 但它比較接近自己的 final successful state
- 而且 trajectory 漂移更小、最大跳動也更小

也就是說：

> **`late-success` 比較像雖然晚，但一旦進到正確 basin，就會穩定收斂。**

相對地：

> **`late-failure` 比較像中途碰到過可用 state，但 trajectory 後來漂走了。**

### `qwen3` 是最清楚的例子

`qwen3` 的 `mean_hidden`：

- `late-success mean cosine-to-final = 0.988`
- `late-failure mean cosine-to-final = 0.902`
- `late-success mean displacement = 5.652`
- `late-failure mean displacement = 8.538`

`qwen3` 的 `last_token_hidden`：

- `late-success mean cosine-to-final = 0.954`
- `late-failure mean cosine-to-final = 0.926`
- `late-success mean displacement = 29.547`
- `late-failure mean displacement = 33.334`

這些數字幾乎就是把研究問題翻成幾何語言：

> full correct 但 re-entry 不行，常常不是因為正確 state 完全不存在，而是因為這個 state 要到很後面才形成，或者雖然形成了，但只有沿著原始 trajectory 才能穩定維持。

---

## 九、Success / failure manifold：為什麼有些 full-correct prefixes 還是不夠可重用？

這部分把 final hidden states 聚成 success / failure centroids，再看每個 prefix 靠哪一邊。

aggregate 的 `late-success - late-failure` 差值：

### `mean_hidden`

- `sim(success) = +0.065`
- `sim(failure) = -0.034`
- `success-failure margin = +0.099`

### `last_token_hidden`

- `sim(success) = +0.028`
- `sim(failure) = +0.003`
- `success-failure margin = +0.025`

這表示：

> `late-success` 的 prefix hidden states，平均比 `late-failure` 更靠近 success manifold。

但關鍵是：

> 它雖然更 success-like，卻常常還是要到後段才真正變得可重用。

這說明 success manifold 的靠近程度本身還不夠；更重要的是：

- 是否已經夠接近可重啟的 basin
- 是否能在之後維持不掉出來

一個很乾淨的例子是 `ministral` 的 `mean_hidden`：

| Category | sim(success) | sim(failure) | success-failure margin |
| --- | ---: | ---: | ---: |
| late-success | 0.902 | 0.872 | +0.029 |
| late-failure | 0.872 | 0.890 | -0.018 |

這代表：

- `late-success` 的 prefix 確實更 success-like
- `late-failure` 的 prefix 更 failure-like

但即使如此，`late-success` 仍然常常不是 early reusable。這又回到同一個核心點：

> **success-likeness 不等於 immediate reusability。**

---

## 十、把所有證據串起來：目前最合理的解釋是什麼？

目前這批資料最支持的整體圖像是：

### 10.1 Full-trace success 常常依賴 trajectory，而不是單一 prefix 中已經封裝好的資訊

也就是：

- 成功不一定已經在中途以「可抽取、可重啟」的形式存在
- 它可能需要原始 rollout 的連續狀態更新，才逐步穩住

所以：

> full correct 並不自然分解成 prefix-level reusable states。

### 10.2 Prefix correctness 的出現，不是最難的部分；最難的是「保存」

`late-failure` 顯示：

- 模型有時中途其實能到局部正解
- 但完整 trace 沒保住

所以瓶頸不只是：

- state formation

也包括：

- state preservation

### 10.3 Re-entry 更像在 probe「correct state 是否已形成且可獨立存活」

這句很重要。因為 re-entry 不是只在問：

- 中途有沒有某些正確線索

它其實在問更強的問題：

- 這些線索是否已經組成一個可被重新啟動、可脫離原始軌跡獨立存活的 state

這也是為什麼：

- full correct 不保證 re-entry correct
- prefix correct 也不保證 full trace correct

---

## 十一、這樣定位後，taxonomy 的意義就更清楚了

taxonomy 不只是分類現象，而是在拆三種不一致：

### A. 完全一致

- `stable-success`
- full correct
- prefix 也早早可重用

### B. Full success 沒有早期顯化成 reusable state

- `late-success`

這正是你現在真正要研究的 anomaly。

### C. Reusable state 出現過，但沒有被保住

- `late-failure`

這是另一個對稱但不同方向的 anomaly。

### D. 兩邊都沒有

- `persistent-failure`

這類主要是 base competence 不足。

所以如果要把研究定位得更準，可以說：

> **我們不是在研究 partial reasoning 何時夠長，而是在研究 full success、prefix reusability、state persistence 為什麼彼此脫鉤。**

---

## 十二、跨 family 觀察

不同 family 對這個問題的表現不太一樣。

### `qwen3`

- 最適合講 mechanistic story
- `late-success vs late-failure` hidden-state 可分性最強
- geometry 上的 persistence / drift 差異也最清楚

### `ministral`

- manifold 結果很乾淨
- 可用來講 success-like vs failure-like state
- 但樣本小，probe 分數要保守解讀

### `gemma4`

- local correctness 的 last-token hidden 訊號很強
- 代表 prefix 當下的局部可用性很容易被讀到

### `mistral7`

- `persistent-failure` 比例高
- 很多 family-level probe 幾乎缺正例

### `qwen25`

- 有局部訊號
- 但沒有 `qwen3` 那麼完整地支撐 state-persistence 故事

---

## 十三、最重要的限制

這些結論雖然方向很一致，但還有幾個要保守的地方：

1. family 的 task 數小
   多數只有 `27-29` 題，某些 category 在 test split 只剩 `1-3` 題。

2. 只有單一 selected layer
   目前還不能做真正完整的 layer-wise trajectory 或 layer-specific causal tracing。

3. 目前還是關聯性證據
   我們能說 state persistence 很像關鍵瓶頸，但還沒做到 activation patching 這種因果測試。

所以目前最穩的結論應該是：

> **這批資料強烈支持 full-trace success 並不等於 prefix-level reusability，而 state persistence 很可能是兩者脫鉤的關鍵因素；但要把它升級成因果主張，還需要進一步 intervention。**

---

## 十四、最適合直接拿去報告的一段話

> 我們原本直覺上會以為，如果一條 full reasoning trace 最後能答對，那它中途某些 prefix 應該也已經包含足夠資訊，re-entry 應該能接回正解。但這批 LiveBench re-entry 結果顯示，事情不是這樣。
> 在 `44` 個 full-correct task 裡，有 `23` 個不是 stable-success，也就是超過一半的 full success 並沒有自然對應到早期、穩定、可重用的 prefix。相反地，在 `97` 個 full-wrong task 裡，又有 `26` 個曾經出現過可重用的 correct prefix，卻沒有被完整 trace 保留到最後。
> 這代表 full correctness、prefix reusability、state persistence 是三件相關但不等價的事。hidden-state 分析也支持這個解讀：模型內部 state 對 local re-entry correctness 比對 full-trace success 更敏感，而 `late-success` 與 `late-failure` 的主要差別，不是誰比較早 first-correct，而是 correct state 出現之後能不能穩定收斂並被保住。
> 因此這條研究的重點不應該是找某個固定最佳 `T`，而是理解為什麼 full-trace success 沒有自然分解成 reusable partial states，以及為什麼有些可用的局部正確 state 最後會流失。

---

## 十五、下一步最值得做的事

如果要真正往 mechanistic explanation 走，最值得做的是：

1. 重新抽 multi-layer hidden states
   直接做 layer-wise trajectory，看哪一層開始出現 `late-success` / `late-failure` 分歧。

2. activation patching
   對 `qwen3` 的 `late-success` / `late-failure` 做 prefix-state patching，測試 correct state 是否真的是 causal bottleneck。

3. 拆出更細的 anomaly taxonomy
   例如把 `late-success` 再拆成：
   - full correct but late reusable
   - full correct but unstable reusable

---

## 對應檔案

- hidden-state probe：[hidden_state_livebench_reentry.md](/Users/powerarena/Documents/GitHub/confidence-tom/docs/mainline/generated/analysis/trace/hidden_state_livebench_reentry.md)
- late-success vs late-failure：[hidden_state_late_success_vs_failure.md](/Users/powerarena/Documents/GitHub/confidence-tom/docs/mainline/generated/analysis/trace/hidden_state_late_success_vs_failure.md)
- geometry：[hidden_state_geometry.md](/Users/powerarena/Documents/GitHub/confidence-tom/docs/mainline/generated/analysis/trace/hidden_state_geometry.md)
- manifolds：[hidden_state_manifolds.md](/Users/powerarena/Documents/GitHub/confidence-tom/docs/mainline/generated/analysis/trace/hidden_state_manifolds.md)
- taxonomy 中文稿：[trace_taxonomy_report_zh.md](/Users/powerarena/Documents/GitHub/confidence-tom/docs/mainline/generated/analysis/trace/trace_taxonomy_report_zh.md)
