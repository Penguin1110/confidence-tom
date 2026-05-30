# Prefix Re-entry 實驗：Hidden-State 分析進度報告

日期：2026-05-28
資料截止：2026-05-27

---

## 一、實驗概覽

本實驗研究「prefix re-entry」現象：給模型一段 reasoning prefix（完整 trace 中途截斷），觀察模型接續後是否比從頭重跑更好或更差。核心問題是：

> **Prefix 對後續推理的角色是 scaffold（鷹架）還是 fragility（脆弱點）？**

目前完成兩個資料集：

| 資料集 | 題型 | 模型數 | Prefix 行數（非 outlier） |
|--------|------|--------|---------------------------|
| **Math500** | 數學推理 | 4 | 949 rows |
| **LiveBench Reasoning** | 綜合推理 | 5 | 1,167 rows |

---

## 二、分析框架

### 2.1 Trace-Based Category 定義

所有分析以 `full_trace_correct`（不是 `full_rerun_correct`）作為 baseline：

```
preserve        = prepare 對，re-entry 也對
trace_rescue    = prepare 錯，re-entry 救回來
trace_damage    = prepare 對，re-entry 反而錯
persistent_fail = prepare 錯，re-entry 仍然錯

stable   = preserve + trace_rescue
unstable = trace_damage + persistent_fail
```

### 2.2 Hidden-State 是什麼

每個 prefix 的 hidden state 是：

```
last-layer, last-token hidden state（4096 dim）
```

即模型讀完整個 re-entry prompt 後、準備繼續生成前的內部狀態向量。

### 2.3 三種分析方法

**方法 A：Hidden Norm**
```
norm(h) = sqrt(Σ hᵢ²)
```
向量長度，粗略代表「內部表示強度」。只能同模型內比較。

**方法 B：Rescue-Damage Centroid + Permutation Test**
```
rescue_centroid = mean(h | category == trace_rescue)
damage_centroid = mean(h | category == trace_damage)
rescue_damage_axis = rescue_centroid - damage_centroid
projection(h) = dot(h, unit(rescue_damage_axis))
```
測試 rescue 和 damage 的高維中心是否可靠不同。

**方法 C：Stable-Unstable Centroid + Projection（主要方法）**
```
stable_centroid   = mean(h | stable)
unstable_centroid = mean(h | unstable)
stability_axis    = stable_centroid - unstable_centroid
projection(h)     = dot(h, unit(stability_axis))
```
投影越高 → 越 stable-like；越低 → 越 unstable-like。

> **重要**：projection 數字只能在同一模型內比較，不能跨模型比絕對大小。

---

## 三、整體行為結果（Trace-Based）

### 3.1 Math500：Prefix 傾向 Scaffold

| 類別 | 數量 | 比率 |
|------|------|------|
| preserve | 441 | — |
| **trace_rescue** | **104** | **22.3%（of prepare-wrong）** |
| trace_damage | 41 | 8.5%（of prepare-correct） |
| persistent_fail | 363 | — |
| **淨效果** | **+63** | scaffold 方向 |

### 3.2 LiveBench Reasoning：Prefix 傾向 Fragility

| 類別 | 數量 | 比率 |
|------|------|------|
| preserve | 275 | — |
| trace_rescue | 61 | 8.5%（of prepare-wrong） |
| **trace_damage** | **177** | **39.2%（of prepare-correct）** |
| persistent_fail | 654 | — |
| **淨效果** | **−116** | fragility 方向 |

---

## 四、分模型行為結果

### 4.1 Math500（非 outlier）

| 模型 | 行數 | prepare 準確率 | re-entry 準確率 | trace_rescue | trace_damage | 解讀 |
|------|------|---------------|----------------|-------------|-------------|------|
| **Gemma** | 244 | 70.5% | 79.5% | **34/72 = 47.2%** | 12/172 = 7.0% | Rescue 最強 |
| Mistral | 254 | 23.6% | 34.6% | 29/194 = 14.9% | 1/60 = 1.7% | Damage 極低 |
| **OLMo** | 219 | 11.4% | 29.2% | **39/194 = 20.1%** | **0/25 = 0.0%** | Rescue 明顯，完全不 damage |
| Qwen | 232 | 97.0% | 85.8% | 2/7 = 28.6% | 28/225 = 12.4% | Prepare 太強，rescue 空間極小 |

### 4.2 LiveBench Reasoning（非 outlier）

| 模型 | 行數 | prepare 準確率 | re-entry 準確率 | trace_rescue | trace_damage | 解讀 |
|------|------|---------------|----------------|-------------|-------------|------|
| Gemma4 | 401 | 64.1%（trace） | — | 12/144 = 8.3% | **97/257 = 37.7%** | Damage 很高 |
| **Qwen3** | 236 | 49.6%（trace） | — | **19/119 = 16.0%** | 43/117 = 36.8% | Rescue 和 Damage 都明顯 |
| Qwen25 | 138 | 32.6%（trace） | — | 10/93 = 10.8% | 18/45 = 40.0% | 不穩定 |
| Ministral | 190 | 12.1%（trace） | — | 7/167 = 4.2% | 10/23 = 43.5% | Rescue 弱，Damage 高 |
| Mistral7 | 202 | 5.0%（trace） | — | 13/192 = 6.8% | 9/10 = 90.0%* | Damage 分母極小，保守解讀 |

*Mistral7 prepare-correct 只有 10 筆，damage rate 不穩定。

---

## 五、Hidden-State 主要分析：Stable/Unstable Centroid

### 5.1 Permutation Test P-values

以 1000 次 permutation 測試 stable/unstable centroid 是否顯著不同：

| Dataset | 模型 | Centroid p-value | 顯著？ |
|---------|------|-----------------|--------|
| Math500 | Gemma | **p ≈ 0.003** | ✓ |
| Math500 | Mistral | **p ≈ 0.003** | ✓ |
| Math500 | OLMo | **p ≈ 0.003** | ✓ |
| Math500 | Qwen | p ≈ 0.163 | ✗ |
| LiveBench | Gemma4 | **p ≈ 0.003** | ✓ |
| LiveBench | Ministral | **p ≈ 0.003** | ✓ |
| LiveBench | Mistral7 | **p ≈ 0.003** | ✓ |
| LiveBench | Qwen25 | **p ≈ 0.010** | ✓（較弱） |
| LiveBench | Qwen3 | **p ≈ 0.003** | ✓ |

**結論**：9 個模型/資料集組合中，8 個有統計顯著的 stable/unstable centroid separation。唯一例外是 Math500 Qwen（prepare 準確率 97%，rescue 樣本極少）。

### 5.2 Stable vs Unstable 整體 Projection

（數字為在該模型 stability axis 上的平均投影，只能同模型內比）

| Dataset | 模型 | Stable 平均 | Unstable 平均 | 分離幅度 |
|---------|------|------------|--------------|---------|
| Math500 | Gemma | **+39.44** | −10.36 | **49.8** |
| Math500 | Mistral | +0.70 | −28.51 | 29.2 |
| Math500 | OLMo | +13.90 | −1.42 | 15.3 |
| Math500 | Qwen | +2.05 | −12.32 | 14.4 |
| LiveBench | Gemma4 | **+51.14** | +2.62 | **48.5** |
| LiveBench | Ministral | **+61.57** | +17.66 | **43.9** |
| LiveBench | Mistral7 | +51.24 | −29.82 | **81.1** |
| LiveBench | Qwen25 | +6.38 | −16.67 | 23.1 |
| LiveBench | Qwen3 | **+43.63** | +11.27 | **32.4** |

---

## 六、分 Trace Category 的 Projection（各模型詳細）

### 6.1 Math500 — Gemma（最清楚的 stability axis）

| Category | n | Mean Projection | 說明 |
|----------|---|-----------------|------|
| trace_rescue | 34 | **+55.6** | 最 stable-like |
| preserve | 160 | **+36.0** | 穩定 |
| trace_damage | 12 | −11.3 | 偏 unstable |
| persistent_fail | 38 | −10.1 | 偏 unstable |

**結構**：stable side（rescue, preserve） >> unstable side（damage, fail）。四類清楚分成兩邊。

---

### 6.2 Math500 — OLMo（無 damage，stable vs fail 清楚分離）

| Category | n | Mean Projection | 說明 |
|----------|---|-----------------|------|
| preserve | 25 | **+14.5** | 穩定 |
| trace_rescue | 39 | **+13.5** | 接近 preserve |
| persistent_fail | 155 | −1.4 | 偏低 |

OLMo 沒有 trace_damage 行，代表 prefix 對 prepare-correct 的 trace 完全不造成破壞。

---

### 6.3 Math500 — Mistral（Preserve 高，Rescue 介於中間）

| Category | n | Mean Projection | 說明 |
|----------|---|-----------------|------|
| preserve | ~59 | +0.70（整體） | 中段 |
| trace_rescue | ~29 | 介於中間 | 不如 preserve 穩 |
| persistent_fail | ~165 | −28.51（整體） | 很低 |

Mistral 的 rescue prefix 不一定已非常 stable-like，而是從 unstable region 往 stable region 移動。模型能從稍不穩的狀態中修正，而不是因為 prefix 本身就已很穩。

---

### 6.4 Math500 — Qwen（訊號弱，受 prepare 太強限制）

| Category | n | Mean Projection | 說明 |
|----------|---|-----------------|------|
| preserve | 197 | +2.05（整體） | 稍正 |
| trace_damage | 28 | −12.32（整體） | 偏低 |
| trace_rescue | 2 | — | 樣本過少 |

整體 stable/unstable centroid 不顯著（p≈0.163）。問題在資料分布：prepare 已有 97% 正確率，rescue 分母（prepare-wrong）只有 7 筆，無法做有力分析。

---

### 6.5 LiveBench — Qwen3（最清楚的 reliability gradient）

| Category | n | Mean Projection | 說明 |
|----------|---|-----------------|------|
| preserve | 74 | **+46.9** | 最 stable-like |
| trace_rescue | 19 | **+30.8** | 次穩 |
| trace_damage | 43 | +14.7 | 偏低 |
| persistent_fail | 100 | +9.8 | 最低 |

**結構**：`preserve > rescue > damage > fail`，形成清楚的單調穩定性梯度。

這是全部模型中最精緻的 hidden-state 訊號：四個 category 有乾淨的線性排序，代表 stability axis 對 Qwen3 不只是二分，而是連續的可靠度量。

---

### 6.6 LiveBench — Gemma4（早期 fragility，後期整體上升）

| Category | n | Mean Projection | 說明 |
|----------|---|-----------------|------|
| preserve | 160 | **+52.6** | 高 |
| trace_rescue | 12 | +32.2 | 中段 |
| persistent_fail | 132 | +7.4 | 混雜 |
| trace_damage | 97 | −3.9 | 最低 |

Gemma4 的結構是：preserve, rescue 在 stable side；damage 在 unstable side；但 persistent_fail 混在中間，不像 Gemma（Math500）那樣乾淨。這表示有些 fail 的 prefix 的內部狀態並不完全不穩，failure 可能有其他原因。

---

### 6.7 LiveBench — Qwen25（訊號弱且混雜）

| Category | n | Mean Projection | 說明 |
|----------|---|-----------------|------|
| preserve | ~28 | +6.38（整體） | 稍正 |
| trace_rescue | ~10 | 不穩定 | — |
| trace_damage | 18 | 不穩定 | — |
| persistent_fail | ~82 | −16.67（整體） | 偏低 |

Qwen25 的 rescue/damage centroid 不顯著（p≈0.425），整體訊號較弱。可能原因：樣本數偏小、或 Qwen25 的 hidden-state 對 prefix 穩定性的編碼不如其他模型清楚。

---

### 6.8 LiveBench — Ministral

| Category | 整體 n | Mean Projection | 說明 |
|----------|--------|-----------------|------|
| preserve | ~14 | +61.57（stable 整體） | 高 |
| trace_rescue | 7 | 中段 | |
| trace_damage | 10 | 中段偏高（多在晚段） | 混雜 |
| persistent_fail | ~160 | +17.66（unstable 整體） | 較低 |

Ministral 的特殊性：damage 樣本很少且 n 小，很多 damage 出現在晚段 prefix（p03 之後），此時整體 projection 都偏高，因此 damage 的 projection 也被拉高，不能輕易說 damage-like prefix 的 hidden state 很不穩。

---

### 6.9 LiveBench — Mistral7（強分離，但 preserve 樣本極少）

整體 stable/unstable 分離幅度是所有模型中最大的（81.1），但 preserve 只有 1 行，非常不平衡。Rescue 和 persistent_fail 在 stability axis 上分得很開：

- Rescue（n=13）：多數 bucket 在正側（+54 ~ +83）
- Persistent_fail（n=192）：多數 bucket 在負側（−31 ~ −49）

因此 Mistral7 的 hidden-state 訊號強，但因為 prepare-correct 樣本過少，整體結論需保守解讀。

---

## 七、分 Prefix 步數（Bucket）的 Projection 詳細表

### 7.1 Math500 Gemma：各 Bucket 的四類投影

| Bucket | preserve | trace_rescue | trace_damage | persistent_fail |
|--------|----------|-------------|-------------|----------------|
| p01（n=47） | +55.3（n=27） | **+61.4**（n=8） | +8.3（n=6） | +12.5（n=6） |
| p02（n=47） | +53.3（n=29） | **+57.0**（n=7） | −21.3（n=4） | +14.5（n=7） |
| p03（n=47） | +41.1（n=32） | **+53.1**（n=8） | −62.5（n=1） | −13.5（n=6） |
| p04-p05（n=75） | +25.6（n=52） | **+49.7**（n=10） | −37.8（n=1） | −19.1（n=12） |
| p06-p10（n=28） | +3.9（n=20） | **+77.9**（n=1） | — | −35.5（n=7） |

**觀察**：跨所有 bucket，rescue 的 projection 幾乎都高於 preserve，且兩者都明顯高於 damage/fail。這意味著 Gemma 的 stability axis 在早期（p01）就已清楚，且隨著 prefix 變長，stable/unstable 的分離並未減弱。

**特別注意 p03**：trace_damage（n=1）projection 達 −62.5，是所有 bucket 中最極端的負值，代表在 p03 位置被 damage 的那筆 prefix 的 hidden state 非常 unstable-like。

---

### 7.2 LiveBench Qwen3：各 Bucket 的四類投影（Reliability Gradient）

| Bucket | preserve | trace_rescue | trace_damage | persistent_fail |
|--------|----------|-------------|-------------|----------------|
| p01（n=27） | +39.8（n=6） | +44.9（n=2） | +17.2（n=6） | +6.2（n=13） |
| p02（n=27） | +45.1（n=6） | +33.2（n=3） | +16.0（n=6） | +6.0（n=12） |
| p03（n=27） | +44.4（n=7） | +34.4（n=2） | +15.1（n=5） | +11.5（n=13） |
| p04-p05（n=46） | +47.8（n=13） | +41.6（n=3） | +15.9（n=8） | +12.0（n=22） |
| p06-p10（n=72） | +47.2（n=21） | +23.4（n=8） | +11.9（n=12） | +10.3（n=31） |
| p11-p20（n=37） | +49.5（n=21） | +16.2（n=1） | +14.1（n=6） | +10.5（n=9） |

**觀察**：Qwen3 幾乎在每個 bucket 都維持 `preserve > rescue > damage > fail` 的排序。這個 reliability gradient 在整個 prefix 長度範圍內都很穩定。

**特別注意 p06-p10 之後的 rescue 下降**：rescue 的 projection 從早期的 +44.9 降到 +16.2，但 preserve 維持在 +47-49。這可能表示晚段的 rescue prefix 的 hidden state 稍微偏向不穩方向，但仍然高於 damage/fail。

---

### 7.3 LiveBench Gemma4：各 Bucket 的四類投影（早期 Fragility）

| Bucket | preserve | trace_rescue | trace_damage | persistent_fail |
|--------|----------|-------------|-------------|----------------|
| p01（n=29） | +2.2（n=6） | −9.2（n=1） | −17.4（n=12） | **−39.4**（n=10） |
| p02（n=29） | +4.6（n=5） | −8.4（n=1） | −15.2（n=13） | **−31.6**（n=10） |
| p03（n=29） | +8.1（n=10） | −11.2（n=1） | −30.2（n=8） | −30.4（n=10） |
| p04-p05（n=56） | +26.9（n=17） | −12.9（n=1） | −22.1（n=19） | −29.2（n=19） |
| p06-p10（n=119） | **+56.8**（n=47） | +46.0（n=7） | −1.2（n=27） | +2.3（n=38） |
| p11-p20（n=108） | **+66.2**（n=56） | +105.4（n=1*） | +38.0（n=16） | +51.3（n=35） |
| p21-p50（n=31） | +76.7（n=19） | — | +55.4（n=2） | +66.7（n=10） |

*n=1 的單點不能做統計結論。

**觀察**：Gemma4 有非常清楚的「prefix 長度效應」：
- p01-p04：所有 category 的 projection 都偏低，damage 和 fail 在很深的負值
- p06+：整體 projection 大幅上升，preserve/rescue 明顯高於 damage
- p11+：整體都很高，damage 和 fail 也被拉高，category separation 減弱

這支持 LiveBench 的 fragility hypothesis：**早期 prefix 特別危險**，不只是 damage/fail 很低，連 preserve 的 projection 也很低（+2.2），意味著即使最後 re-entry 成功的那些 prefix，早期的 hidden state 也不特別 stable。

---

### 7.4 Math500 OLMo：各 Bucket（無 Damage）

| Bucket | preserve | trace_rescue | persistent_fail |
|--------|----------|-------------|----------------|
| p01（n=44） | +12.7（n=7） | +11.0（n=6） | −3.9（n=31） |
| p02（n=44） | +13.9（n=7） | +12.4（n=8） | −2.1（n=29） |
| p03（n=40） | +17.1（n=5） | +13.1（n=8） | −0.4（n=27） |
| p04-p05（n=59） | +13.8（n=5） | **+16.2**（n=15） | +0.5（n=39） |
| p06-p10（n=32） | +22.5（n=1） | +6.7（n=2） | −1.6（n=29） |

**觀察**：OLMo 跨所有 bucket 的結構非常穩定：preserve/rescue 在正側，persistent_fail 在負側。分離幅度不如 Gemma 大，但非常一致。

---

### 7.5 Math500 Mistral：各 Bucket（Preserve 高，Rescue 偏低）

| Bucket | preserve | trace_rescue | trace_damage | persistent_fail |
|--------|----------|-------------|-------------|----------------|
| p01（n=47） | +10.7（n=12） | −8.7（n=7） | — | −18.9（n=28） |
| p02（n=47） | +14.4（n=12） | −2.9（n=7） | — | −21.6（n=28） |
| p03（n=47） | +7.9（n=12） | −2.8（n=4） | — | −28.0（n=31） |
| p04-p05（n=76） | −1.7（n=18） | −15.1（n=8） | — | −33.0（n=50） |
| p06-p10（n=37） | +4.6（n=5） | −37.7（n=3） | −13.8（n=1） | −38.0（n=28） |

**觀察**：Mistral 的 rescue projection 比 preserve 低，甚至在晚段（p06+）低到 −37.7。這表示 rescue prefix 不是因為「hidden state 已經很 stable」而成功，而是模型仍然能從不穩的狀態中接回答案。

---

## 八、Rescue-Damage Centroid Permutation Test（補充）

這是針對 trace_rescue 和 trace_damage 這兩個 category 的 centroid 差異：

| Dataset | 模型 | Rescue-Damage p-value | 可信度 |
|---------|------|----------------------|--------|
| LiveBench | Gemma4 | **p ≈ 0.003** | 高 |
| LiveBench | Qwen3 | **p ≈ 0.003** | 高 |
| LiveBench | Ministral | p ≈ 0.007 | 高 |
| LiveBench | Mistral7 | p ≈ 0.040 | 中 |
| Math500 | Gemma | **p ≈ 0.003** | 高 |
| LiveBench | Qwen25 | p ≈ 0.425 | 不顯著 |
| Math500 | Mistral | — | damage 樣本不足 |
| Math500 | OLMo | — | 無 damage 行 |
| Math500 | Qwen | — | rescue 樣本不足 |

---

## 九、特殊發現：Hidden Norm（LiveBench Qwen3）

對多數模型，只看 hidden norm（向量長度）不夠。但 LiveBench Qwen3 的 norm 有非常強的 rescue/damage 差異：

| 統計量 | 數值 |
|--------|------|
| Rescue 平均 norm | 173.0 |
| Damage 平均 norm | 166.7 |
| Cohen's d | **2.026** |
| Welch t-test p | **5.7 × 10⁻⁷** |
| FDR q（BH correction） | **4.0 × 10⁻⁶** |

這表示 Qwen3 在 LiveBench 上的 rescue prefix，不只是高維位置不同，連 hidden vector 的「長度」本身也有顯著差異。這是目前所有模型中最強的 norm-level 訊號。

---

## 十、跨模型整理：三類訊號強度

### Group 1：清楚穩定軸（Strong signal）

| 模型/資料集 | Hidden-state 特徵 |
|------------|-----------------|
| Math500 Gemma | preserve / rescue >> damage / fail，clean binary split |
| LiveBench Qwen3 | preserve > rescue > damage > fail，monotone gradient |
| Math500 OLMo | preserve / rescue >> persistent_fail（無 damage） |

**共同特徵**：four-category ordering 清楚，stability axis p ≈ 0.003，centroid 分離大。

---

### Group 2：有穩定軸，但類別混合較多（Moderate signal）

| 模型/資料集 | Hidden-state 特徵 |
|------------|-----------------|
| Math500 Mistral | preserve 高，rescue 落在中間，fail 很低；rescue 不是因 stable hidden state 而成功 |
| LiveBench Gemma4 | 強烈的 prefix 長度效應：早期脆弱，晚期才穩；persistent_fail 混入 stable region |

**共同特徵**：整體 stable/unstable 顯著，但 rescue 和 damage 的 projection 不如 Group 1 那麼整齊分開。

---

### Group 3：訊號弱或受樣本限制（Weak/limited signal）

| 模型/資料集 | 原因 |
|------------|------|
| Math500 Qwen | prepare 準確率 97%，rescue 分母只有 7 筆；stable/unstable centroid p ≈ 0.163 |
| LiveBench Qwen25 | rescue/damage centroid p ≈ 0.425；樣本偏小 |
| LiveBench Mistral7 | preserve 只有 1 筆；rescue/fail 分離強但基礎極不平衡 |

---

## 十一、綜合解釋

目前 hidden-state 分析支持的敘述是：

> Prefix 的可靠性（是否能讓 re-entry 成功）在一定程度上反映於模型讀完 prefix 後的內部 hidden state。穩定的 prefix（preserve 和 rescue）通常在 model-specific stability axis 上投影較高；不穩定的 prefix（damage 和 fail）通常投影較低。這個分離在 8/9 個模型/資料集組合中具有統計顯著性。

但要注意三個限制：

1. **點雲重疊**：category 的 projection 分布有顯著均值差，但仍高度重疊，不是乾淨的分類線。
2. **Axis 不可命名**：stability axis 是高維方向，不是任何單一可解釋的 feature 或 neuron。
3. **跨模型不可比**：同一個 projection 數字在 Gemma 和 Qwen 之間沒有直接可比性。

---

## 十二、現有限制

| 限制 | 說明 |
|------|------|
| Last-token only | 只用最後一層最後一個 token；沒有 mean pooling 或 token trajectory |
| 高維方向無語意 | Stability axis 沒有單一人類可解讀的含義 |
| 樣本不平衡 | 部分 category（尤其 rescue in Math500 Qwen，damage in Math500 OLMo）行數極少 |
| Projection 只能內部比 | 跨模型不能比絕對大小 |
| Outlier 排除 | segment_count_outlier == 1 的行已排除；這些行有不同的 prefix 分布 |

---

## 十三、下一步建議

| 優先級 | 任務 | 說明 |
|--------|------|------|
| 高 | 跑 GPQA Diamond（v2 pipeline） | 測試 scaffold vs fragility 是否泛化到科學知識推理題 |
| 高 | 分析 GPQA hidden state | GPQA v2 已內建 inline probe，可直接出 projection 結果 |
| 中 | Mean pooling / multi-layer probe | 比較 last-token 和 mean-pooled hidden state 的訊號強度 |
| 中 | Logistic classifier on hidden state | 測試 stability projection 是否能真正預測 re-entry outcome |
| 低 | Token-level trajectory analysis | 看 hidden state 在 prefix token 之間的演化 |

---

## 附：快速重現表格的 Script

（需在 repo root 執行）

```powershell
@'
import json, math
from pathlib import Path
from collections import defaultdict

base = Path("outputs/results")
reentry_files = [base / "reentry/_reentry_math500_local_v1/reentry_rows.jsonl"]
reentry_files += list((base / "reentry").glob("_reentry_livebench_local_v1_*_no_outliers/reentry_rows.jsonl"))
probe_files = [base / "probe/_reentry_math500_local_v1/reentry_probe_rows.jsonl"]
probe_files += list((base / "probe").glob("_reentry_livebench_local_v1_*_no_outliers/reentry_probe_rows.jsonl"))

def key(r): return (r.get("benchmark"), r.get("run_name"), r.get("prefix_id"))
labels = {}
for path in reentry_files:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if int(r.get("segment_count_outlier") or 0): continue
        trace = int(r.get("full_trace_correct") or 0)
        exact = int(r.get("reentry_exact_correct") or r.get("small_continue_correct") or 0)
        cat = "preserve" if trace and exact else "trace_damage" if trace else "trace_rescue" if exact else "persistent_fail"
        labels[key(r)] = {"benchmark": r.get("benchmark"), "family": r.get("small_family",""), "category": cat, "stable": cat in {"preserve","trace_rescue"}}

rows = []
for path in probe_files:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        label = labels.get(key(r))
        if not label: continue
        h = r.get("last_token_hidden")
        if h is None: continue
        rows.append({**label, "h": [float(x) for x in h]})

def centroid(vecs): n=len(vecs); d=len(vecs[0]); return [sum(v[i] for v in vecs)/n for i in range(d)]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def nm(a): return math.sqrt(sum(x*x for x in a))
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

groups = defaultdict(list)
for r in rows: groups[(r["benchmark"], r["family"])].append(r)

print("benchmark\tfamily\tcategory\tn\tmean_projection")
for (bm, fam), grp in sorted(groups.items()):
    s = [r["h"] for r in grp if r["stable"]]
    u = [r["h"] for r in grp if not r["stable"]]
    if len(s)<2 or len(u)<2: continue
    axis = sub(centroid(s), centroid(u))
    unit = [x/nm(axis) for x in axis]
    by_cat = defaultdict(list)
    for r in grp: by_cat[r["category"]].append(dot(r["h"], unit))
    for cat in ["preserve","trace_rescue","trace_damage","persistent_fail"]:
        vals = by_cat.get(cat, [])
        if vals: print(f"{bm}\t{fam}\t{cat}\t{len(vals)}\t{mean(vals):.3f}")
'@ | .\.venv\Scripts\python.exe -
```
