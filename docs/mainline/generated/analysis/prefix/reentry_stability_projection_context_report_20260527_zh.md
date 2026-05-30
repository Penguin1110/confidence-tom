# Prefix Stability Projection 分析報告

日期：2026-05-27

這份報告接續 hidden-state analysis，專門分析：

> 不同模型、不同 prefix bucket、不同 trace category 的 hidden-state stability projection 是否呈現一致結構？

原始表格在：

```text
docs/mainline/generated/analysis/prefix/reentry_stability_projection_by_prefix_category_20260527.tsv
```

這份報告不是只列數字，而是把那些數字整理成可解釋的實驗結論。

## 分析問題

前面的實驗已經發現：

1. Math500 中 re-entry 有明顯 rescue 效果。
2. LiveBench reasoning 中 re-entry 比較容易造成 damage。
3. hidden state 中存在一條 model-specific 的 stability axis。

這裡進一步問：

> 如果把 prefix 按照模型、prefix 位置、trace category 分開看，穩定和不穩定 prefix 的 hidden state 是否仍然有可辨識的結構？

換句話說，不只看整體平均，而是看：

```text
benchmark / model / prefix bucket / category -> mean stability projection
```

## Category 定義

本報告使用 trace-based category：

```text
preserve = prepare 對，reentry 也對
trace_rescue = prepare 錯，reentry 救回來
trace_damage = prepare 對，reentry 反而錯
persistent_fail = prepare 錯，reentry 仍然錯
```

其中：

```text
stable = preserve + trace_rescue
unstable = trace_damage + persistent_fail
```

## Stability Projection 是什麼

對每個模型各自計算：

```text
stable_centroid = mean(hidden states of stable prefixes)
unstable_centroid = mean(hidden states of unstable prefixes)
stability_axis = stable_centroid - unstable_centroid
```

然後每個 prefix 的 hidden state 投影到這條 axis：

```text
projection(h) = dot(h, unit(stability_axis))
```

讀法：

```text
mean_projection 越高 -> 越 stable-like
mean_projection 越低 -> 越 unstable-like
```

這個數字不能跨模型直接比大小。它的意義是在同一個模型內比較：

```text
同一模型裡 preserve 是否比 damage 高？
同一模型裡 rescue 是否比 persistent_fail 高？
同一個 prefix bucket 裡，四種 category 是否有穩定排序？
```

## 整體發現

整體來看，stability projection 支持一個清楚但有條件的結論：

> Prefix 的穩定性不只反映在最後答案，也反映在模型讀完 prefix 當下的 hidden state。成功 re-entry 的 prefix 通常投影較高，damage 或 fail 的 prefix 通常投影較低。

但這個結構是 model-dependent：

```text
最清楚：Math500 Gemma、LiveBench Qwen3
中等清楚：Math500 OLMo、Math500 Mistral、LiveBench Gemma4
較弱或樣本限制大：Math500 Qwen、LiveBench Qwen25、LiveBench Mistral7
```

## Math500 Gemma：最乾淨的 Stability Axis

Math500 Gemma 是目前最有說服力的 case。它在多數 prefix bucket 都呈現：

```text
preserve / trace_rescue  >>  trace_damage / persistent_fail
```

代表成功 re-entry 的 prefix 和失敗 re-entry 的 prefix 在 hidden-state stability axis 上分得很開。

### p01

```text
preserve        +55.3
trace_rescue    +61.4
trace_damage     +8.3
persistent_fail +12.5
```

### p02

```text
preserve        +53.3
trace_rescue    +57.0
trace_damage    -21.3
persistent_fail +14.5
```

### p03

```text
preserve        +41.1
trace_rescue    +53.1
trace_damage    -62.5
persistent_fail -13.5
```

### p04-p05

```text
preserve        +25.6
trace_rescue    +49.7
trace_damage    -37.8
persistent_fail -19.1
```

### 解讀

Math500 Gemma 的 hidden state 很像有一條清楚的「可接續性方向」：

```text
stable side:   preserve, trace_rescue
unstable side: trace_damage, persistent_fail
```

這支持 Math500 中 prefix scaffold 的說法：

> 即使完整 prepare trace 最後答錯，中間某些 prefix 仍然會把模型帶到一個 stable-like hidden state，使 re-entry 有機會救回答案。

在這個模型上，hidden-state projection 不只是事後描述，而是很像一個 prefix reliability signal。

## LiveBench Qwen3：明顯的 Reliability Gradient

LiveBench Qwen3 不是把四類乾淨切成正負兩邊，而是形成一個穩定性排序：

```text
preserve / trace_rescue > trace_damage > persistent_fail
```

### p01

```text
preserve        +39.8
trace_rescue    +44.9
trace_damage    +17.2
persistent_fail  +6.2
```

### p02

```text
preserve        +45.1
trace_rescue    +33.2
trace_damage    +16.0
persistent_fail  +6.0
```

### p03

```text
preserve        +44.4
trace_rescue    +34.4
trace_damage    +15.1
persistent_fail +11.5
```

### p06-p10

```text
preserve        +47.2
trace_rescue    +23.4
trace_damage    +11.9
persistent_fail +10.3
```

### 解讀

Qwen3 的 hidden space 比較像「連續穩定性梯度」：

```text
最穩：preserve
次穩：trace_rescue
偏不穩：trace_damage
最不穩：persistent_fail
```

這和前面 hidden norm 的結果一致：LiveBench Qwen3 是 hidden-state 訊號最強的模型之一。

但也要注意，LiveBench Qwen3 同時有 rescue 和 damage。因此它不是「re-entry 一定好」，而是：

> hidden state 可以幫助區分哪些 prefix 比較可能成功接續，哪些 prefix 可能破壞原本正確 trace。

## Math500 OLMo：Stable vs Fail 分離清楚

Math500 OLMo 幾乎沒有 trace_damage，因此不能完整比較 rescue 和 damage，但可以看 stable 和 persistent_fail。

### p01

```text
preserve        +12.7
trace_rescue    +11.0
persistent_fail  -3.9
```

### p02

```text
preserve        +13.9
trace_rescue    +12.4
persistent_fail  -2.1
```

### p04-p05

```text
preserve        +13.8
trace_rescue    +16.2
persistent_fail  +0.5
```

### 解讀

OLMo 的結構是：

```text
preserve / trace_rescue > persistent_fail
```

雖然缺少 damage 類別，但它仍支持：

> 成功 re-entry 的 prefix hidden state 比 persistent failure 更 stable-like。

這也解釋了 Math500 OLMo 的行為：prepare 本身弱，但 re-entry 能救一部分題目，而且幾乎不破壞原本正確的 trace。

## Math500 Mistral：Preserve 高，Rescue 介於中間

Math500 Mistral 的 pattern 比 Gemma 弱，但仍有結構：

```text
preserve > trace_rescue > persistent_fail
```

### p01

```text
preserve        +10.7
trace_rescue     -8.7
persistent_fail -18.9
```

### p02

```text
preserve        +14.4
trace_rescue     -2.9
persistent_fail -21.6
```

### p04-p05

```text
preserve         -1.7
trace_rescue    -15.1
persistent_fail -33.0
```

### 解讀

Mistral 的 rescue prefix 不一定已經非常 stable-like。它更像：

> rescue 從 unstable region 往 stable region 移動，但沒有像 preserve 那麼高。

這表示 Mistral 的 rescue 可能不是因為 prefix 本身非常穩，而是模型仍能從較不穩的內部狀態中修正回來。

## LiveBench Gemma4：早期 Fragility，後期整體上升

LiveBench Gemma4 是比較混雜但很有意思的模型。早期 projection 整體偏低，代表早期 prefix 很容易落在 unstable-like 狀態。

### p01

```text
preserve          +2.2
trace_rescue      -9.2
trace_damage     -17.4
persistent_fail  -39.4
```

### p02

```text
preserve          +4.6
trace_rescue      -8.4
trace_damage     -15.2
persistent_fail  -31.6
```

### p06-p10

```text
preserve        +56.8
trace_rescue    +46.0
trace_damage     -1.2
persistent_fail  +2.3
```

### p11-p20

```text
preserve        +66.2
trace_rescue   +105.4  # n=1
trace_damage    +38.0
persistent_fail +51.3
```

### 解讀

Gemma4 的早期 prefix 很脆弱：

```text
p01/p02 中 preserve 也不高，damage/fail 更低
```

到了中後段，整體 projection 上升，代表模型內部狀態變得更 stable-like。但 damage 和 persistent_fail 也會被拉高，因此 category separation 沒有 Math500 Gemma 那麼乾淨。

這支持 LiveBench 的整體結論：

> LiveBench prefix 的問題不是完全沒有訊號，而是訊號混雜。即使 hidden state 變得比較 stable-like，也不保證 re-entry 不會 damage。

## LiveBench Qwen25：訊號弱且混雜

Qwen25 的結果比較弱，和前面 rescue/damage centroid 不顯著一致。

### p01

```text
preserve        +16.8
trace_rescue    +12.0
trace_damage     -4.0
persistent_fail  -9.4
```

### p02

```text
preserve         +7.2
trace_rescue     -3.5
trace_damage     -6.5
persistent_fail -20.4
```

### p04-p05

```text
preserve        +11.2
trace_rescue    -13.4
trace_damage     -2.1
persistent_fail -17.8
```

### 解讀

Qwen25 大致仍有：

```text
preserve > persistent_fail
```

但 rescue 和 damage 的排序不穩定。這代表 stability axis 對 Qwen25 有一點訊號，但不足以穩定區分 rescue/damage。

## Math500 Qwen：整體訊號較弱

Math500 Qwen 的 prepare 本來就很強，所以 rescue 樣本很少，這限制了 hidden-state 分析。

一些 bucket 中 damage 確實比 preserve 低：

```text
p03:
preserve       +6.9
trace_damage  -29.1

p11-p20:
preserve       -0.2
trace_damage  -16.3

p21-p50:
preserve       +7.8
trace_damage   -1.4
```

但整體 stable/unstable centroid 在 Math500 Qwen 中不顯著：

```text
centroid p ~= 0.163
```

### 解讀

Math500 Qwen 的主要問題是資料分布：

```text
prepare 已經很常答對 -> rescue denominator 很小
```

因此這裡不能強推 hidden-state conclusion。比較穩的說法是：

> Math500 Qwen 有局部 preserve > damage 的趨勢，但整體 hidden-state stability axis 不夠顯著。

## 跨模型整理

可以把模型分成三組：

### 1. 清楚穩定軸

```text
Math500 Gemma
LiveBench Qwen3
Math500 OLMo
```

特徵：

```text
preserve / trace_rescue 明顯高於 trace_damage / persistent_fail
```

這些模型最支持 hidden-state stability signal。

### 2. 有穩定軸，但類別混合較多

```text
Math500 Mistral
LiveBench Gemma4
```

特徵：

```text
preserve 通常高
fail 通常低
rescue 或 damage 可能混在中間
```

這些模型說明 hidden-state signal 存在，但不是乾淨分類器。

### 3. 訊號弱或受樣本限制

```text
Math500 Qwen
LiveBench Qwen25
LiveBench Mistral7
```

特徵：

```text
rescue/damage 樣本不平衡
或 category projection 排序不穩定
```

這些模型不適合拿來做強結論。

## 研究結論

這份 prefix/category projection 分析支持以下結論：

> Prefix reliability is reflected not only in the final answer, but also in the hidden state after reading the prefix.

中文：

> prefix 是否可靠，不只體現在最後答案，也體現在模型讀完 prefix 後的 hidden state。

更具體地說：

1. **Math500 Gemma** 顯示最清楚的 stable/unstable hidden-state 分離。
2. **LiveBench Qwen3** 顯示穩定性梯度，category 排序很清楚。
3. **Math500 OLMo** 顯示 stable outcomes 和 persistent failure 的分離。
4. **LiveBench Gemma4** 顯示早期 prefix fragility，但中後段 hidden state 變得更 stable-like。
5. **Qwen25 / Math500 Qwen** 的訊號較弱，需要保守解讀。

## 最適合放進論文或報告的說法

可以寫：

> We stratify hidden-state stability projections by benchmark, model, prefix bucket, and trace category. In Math500 Gemma and LiveBench Qwen3, successful re-entry outcomes consistently occupy higher stability-axis projections than damage or persistent failures. This suggests that prefix reliability is reflected in the model's internal state before continuation. However, the effect is model-dependent: some models show a clean stability direction, while others exhibit weaker or more mixed category separation.

中文版本：

> 我們將 hidden-state stability projection 依照 benchmark、模型、prefix bucket 與 trace category 分層分析。結果顯示，在 Math500 Gemma 和 LiveBench Qwen3 中，成功 re-entry 的 prefix 通常具有較高的 stability-axis projection，而 damage 或 persistent failure 的 prefix 較低。這表示 prefix 是否可靠，在模型繼續生成之前已經反映於其內部 hidden state。不過，這個效果具有模型依賴性；有些模型呈現清楚的穩定性方向，有些模型則只有較弱或混雜的類別分離。

## 使用限制

1. Projection 數字只能在同一模型內比較，不能跨模型比絕對大小。
2. 有些 category 的樣本數很小，例如 `n=1`，只能當作提示，不能當作強結論。
3. Stability axis 是高維方向，不是單一可命名特徵。
4. Hidden-state separation 不是完美分類線，點雲仍然重疊。
5. 目前使用的是 last-layer last-token hidden state；若要更完整，可再比較 mean pooling、不同 layer、或 token-level trajectory。

