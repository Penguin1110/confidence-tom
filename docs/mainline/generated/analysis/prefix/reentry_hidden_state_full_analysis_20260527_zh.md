# Re-entry Hidden-State Analysis 中文版

日期：2026-05-27

這份筆記整理目前 re-entry 實驗的 hidden-state 分析。目標有三個：

1. 說清楚這個實驗到底在量什麼。
2. 說清楚每個數字是怎麼算出來的。
3. 說明 hidden state 對「穩定 prefix、不穩定 prefix、rescue、damage」透露了什麼。

這份報告最重要的 convention 是：**rescue 使用 trace-based 定義**。

```text
trace_rescue = prepare/full_trace_correct == 0 and reentry_exact_correct == 1
trace_damage = prepare/full_trace_correct == 1 and reentry_exact_correct == 0
preserve = prepare/full_trace_correct == 1 and reentry_exact_correct == 1
persistent_fail = prepare/full_trace_correct == 0 and reentry_exact_correct == 0
```

這和用 `full_rerun_correct` 判斷不同。這裡真正想問的是：

> 一個原本 prepare 完整 trace 答錯的題目，其中某個中間 prefix 是否仍然能把模型帶回正確答案？

也就是：

```text
prepare 錯 -> reentry 對
```

才是這份分析裡的主要 rescue。

## 使用的資料

分析使用非 outlier 的 re-entry rows，以及對應的 probe rows：

```text
outputs/results/reentry/_reentry_math500_local_v1/reentry_rows.jsonl
outputs/results/probe/_reentry_math500_local_v1/reentry_probe_rows.jsonl

outputs/results/reentry/_reentry_livebench_local_v1_*_no_outliers/reentry_rows.jsonl
outputs/results/probe/_reentry_livebench_local_v1_*_no_outliers/reentry_probe_rows.jsonl
```

reentry row 和 probe row 用以下 key 對起來：

```text
(benchmark, run_name, prefix_id)
```

並且排除：

```text
segment_count_outlier == 1
```

## 這裡的 Hidden State 是什麼

probe 拿到的是：

```text
last-layer, last-token hidden state
```

白話說，就是：

> 模型讀完整個 re-entry prompt 和 reasoning prefix 之後，最後一層、最後一個 token 的內部狀態。

可以把它想成：

```text
hidden_state(prefix) = 模型讀完這段 prefix 後，準備繼續生成時的內部狀態
```

每個 hidden state 是一個高維向量：

```text
h = [x1, x2, x3, ..., xd]
```

每個 prefix 都會變成高維空間裡的一個點。然後我們依照 re-entry 結果替這個點上 label：

```text
preserve
trace_rescue
trace_damage
persistent_fail
```

很重要的一點：hidden state 的單一維度通常不能直接解釋成人類語意。不要把某個 dimension 說成「信心」、「數學能力」或「正確性」。比較可靠的分析單位是：

```text
centroid
direction
projection
```

也就是看一群 hidden states 的平均位置、群體之間的方向，以及每個點往那條方向投影多少。

## 實驗主結果

### Math500

Math500 比較支持 prefix scaffold 的假設。

非 outlier、trace-based 結果：

```text
trace_rescue = 104 / 467 = 22.3%
trace_damage = 41 / 482 = 8.5%
preserve = 441
persistent_fail = 363
net = +63
```

分模型：

```text
model     trace_rescue        trace_damage       解讀
gemma     34/72 = 47.2%       12/172 = 7.0%      rescue 最強
mistral   29/194 = 14.9%      1/60 = 1.7%        damage 很低
olmo      39/194 = 20.1%      0/25 = 0.0%        有 rescue，幾乎不 damage
qwen      2/7 = 28.6%         28/225 = 12.4%     prepare 本來就強，rescue 空間小
```

結論：

> 在 Math500 裡，很多 prefix 真的像有用的推理 scaffold。即使原本完整 prepare trace 最後答錯，中間某些 prefix 仍然能讓模型接回正確答案。

### LiveBench Reasoning

LiveBench 則明顯更脆弱。

非 outlier、trace-based 結果：

```text
trace_rescue = 61 / 715 = 8.5%
trace_damage = 177 / 452 = 39.2%
preserve = 275
persistent_fail = 654
net = -116
```

分模型：

```text
model       trace_rescue        trace_damage       解讀
gemma4      12/144 = 8.3%       97/257 = 37.7%     damage 很高
qwen3       19/119 = 16.0%      43/117 = 36.8%     rescue 和 damage 都明顯
qwen25      10/93 = 10.8%       18/45 = 40.0%      不穩定
ministral   7/167 = 4.2%       10/23 = 43.5%      rescue 弱
mistral7    13/192 = 6.8%      9/10 = 90.0%       prepare-correct 分母很小，保守看待
```

結論：

> LiveBench 的 prefix 不太像穩定 scaffold。re-entry 常常會破壞原本正確的 trace，因此這個 benchmark 更像是在暴露 prefix fragility，而不是 robust recoverability。

## Hidden State 的三種看法

### 1. Hidden Norm

hidden norm 是 hidden vector 的長度：

```text
norm(h) = sqrt(sum_i h_i^2)
```

它可以粗略看成內部表示的「強度」，但只能在同一個模型內比較，不能跨模型比較。

錯誤比較：

```text
Qwen norm 170 vs Gemma norm 240
```

正確比較：

```text
Qwen rescue norm vs Qwen damage norm
Gemma stable norm vs Gemma unstable norm
```

目前最清楚的 norm-level 訊號是 LiveBench qwen3：

```text
LiveBench qwen3:
rescue norm = 173.0
damage norm = 166.7
Cohen's d = 2.026
Welch p = 5.7e-07
FDR q = 4.0e-06
```

這表示在 LiveBench qwen3 裡，rescue 和 damage 不只是高維位置不同，hidden norm 本身也有很強差異。

但多數模型裡，單看 norm 不夠。

### 2. Rescue-Damage Centroid

對每個模型，定義：

```text
rescue_centroid = mean(hidden states of trace_rescue rows)
damage_centroid = mean(hidden states of trace_damage rows)

rescue_damage_axis = rescue_centroid - damage_centroid
```

然後把每個 hidden state 投影到這條軸上：

```text
projection(h) = dot(h, unit(rescue_damage_axis))
```

投影越高，代表越 rescue-like；投影越低，代表越 damage-like。

centroid permutation test 顯示，以下模型的 rescue/damage centroid 有顯著差異：

```text
LiveBench gemma4      p ~= 0.003
LiveBench ministral   p ~= 0.007
LiveBench mistral7    p ~= 0.040
LiveBench qwen3       p ~= 0.003
Math500 gemma         p ~= 0.003
```

不顯著：

```text
LiveBench qwen25      p ~= 0.425
```

資料不足，不適合做 rescue/damage centroid 比較：

```text
Math500 mistral       damage rows 太少
Math500 olmo          沒有 damage rows
Math500 qwen          rescue rows 太少
```

解讀：

> rescue 和 damage 大多不能靠 hidden norm 乾淨分開，但在幾個模型中，full hidden vector centroid 有統計上可靠的高維偏移。

也就是說，差異不是「長度差」而已，而是 hidden space 裡的位置和方向差。

### 3. Stable-Unstable Centroid

如果問題改成：

> 什麼 hidden-state 特徵代表 prefix 穩定或不穩定？

那最有用的是 stable/unstable centroid。

定義：

```text
stable = preserve + trace_rescue
unstable = trace_damage + persistent_fail

stable_centroid = mean(hidden states of stable rows)
unstable_centroid = mean(hidden states of unstable rows)

stability_axis = stable_centroid - unstable_centroid
```

然後：

```text
projection(h) = dot(h, unit(stability_axis))
```

投影越高，代表越 stable-like；投影越低，代表越 unstable-like。

這是回答以下問題最直接的方法：

> 那些特別穩定或不穩定的 prefix，在 hidden state 裡有什麼共同特徵？

答案是：

```text
穩定 prefix：在 model-specific stability axis 上投影較高
不穩定 prefix：在 model-specific stability axis 上投影較低
```

## Stable vs Unstable 的 Hidden-State 結果

大多數模型和 benchmark 都有顯著的 stable/unstable centroid separation：

```text
benchmark            model       centroid p
livebench_reasoning  gemma4      0.003
livebench_reasoning  ministral   0.003
livebench_reasoning  mistral7    0.003
livebench_reasoning  qwen25      0.010
livebench_reasoning  qwen3       0.003
math500              gemma       0.003
math500              mistral     0.003
math500              olmo        0.003
math500              qwen        0.163
```

只有 Math500 qwen 在這個分析裡沒有清楚顯著的 stable/unstable centroid separation。

### Stable-Like Projection

各模型在 stability axis 上的平均投影：

```text
benchmark            model       stable mean   unstable mean
livebench_reasoning  gemma4      +51.14        +2.62
livebench_reasoning  ministral   +61.57        +17.66
livebench_reasoning  mistral7    +51.24        -29.82
livebench_reasoning  qwen25      +6.38         -16.67
livebench_reasoning  qwen3       +43.63        +11.27
math500              gemma       +39.44        -10.36
math500              mistral     +0.70         -28.51
math500              olmo        +13.90        -1.42
math500              qwen        +2.05         -12.32
```

這給出一個簡單但有用的 hidden-state 特徵：

```text
stable prefix 的 projection 較高
unstable prefix 的 projection 較低
```

這不是單一可命名 neuron，而是一條模型內部的高維方向。

## 按 Trace Category 看 Projection

以下是各 category 在 stability axis 上的平均投影。

### Math500 Gemma

```text
category          n     mean projection
preserve          160   +36.0
trace_rescue      34    +55.6
trace_damage      12    -11.3
persistent_fail   38    -10.1
```

這是目前最乾淨的結構：

```text
stable side:   preserve, trace_rescue
unstable side: trace_damage, persistent_fail
```

解讀：

> Math500 Gemma 有很強的 internal stability direction。rescue 和 preserve prefixes 位在穩定側；damage 和 persistent failure 位在不穩定側。

### LiveBench Qwen3

```text
category          n     mean projection
preserve          74    +46.9
trace_rescue      19    +30.8
trace_damage      43    +14.7
persistent_fail   100   +9.8
```

這形成一個很漂亮的 reliability gradient：

```text
preserve > trace_rescue > trace_damage > persistent_fail
```

解讀：

> LiveBench Qwen3 沒有把四類切成完全兩邊，但它有很強的單調穩定性排序。越穩定的 category，projection 越高。

### LiveBench Gemma4

```text
category          n     mean projection
preserve          160   +52.6
trace_rescue      12    +32.2
trace_damage      97    -3.9
persistent_fail   132   +7.4
```

解讀：

> preserve 和 rescue 明顯比 damage 更 stable-like。persistent_fail 則比較混雜，可能表示有些錯誤 trace 的 hidden state 仍接近 stable direction，但最後失敗原因不是單純「內部狀態完全不穩」。

### Math500 OLMo

```text
category          n     mean projection
preserve          25    +14.5
trace_rescue      39    +13.5
persistent_fail   155   -1.4
```

Math500 OLMo 沒有 damage rows。

解讀：

> OLMo 的 preserve/rescue 位在較高的 stability region，而 persistent failure 位在較低的 stability region。

## Prefix 穩定性

先前用 step bucket 看 prefix，是為了定位不穩定大概常出現在哪裡。但更深層的結論不是：

```text
第幾步造成不穩定
```

而是：

> 任意步數的 prefix 都可能落在 stable-like 或 unstable-like 的 internal state。step bucket 只是統計哪些區間比較常出現這種狀態。

damage rate 定義：

```text
damage_rate = trace_damage / prepare-correct prefixes
```

rescue rate 定義：

```text
rescue_rate = trace_rescue / prepare-wrong prefixes
```

### Math500 Gemma

```text
bucket   rescue_rate_wrong   damage_rate_ok
p01      57.1%               18.2%
p02      50.0%               12.1%
p03      57.1%               3.0%
p04-05   45.5%               1.9%
p06-10   12.5%               0.0%
```

解讀：

> Math500 Gemma 有很多 rescue-capable prefixes，而且 damage 很快下降。這支持 Math500 prefix scaffold 的說法。

### LiveBench Gemma4

```text
bucket   rescue_rate_wrong   damage_rate_ok
p01      9.1%                66.7%
p02      9.1%                72.2%
p03      9.1%                44.4%
p04-05   5.0%                52.8%
p06-10   15.6%               36.5%
p11-20   2.8%                22.2%
p21-50   0.0%                9.5%
```

解讀：

> LiveBench Gemma4 早期 prefix 很脆弱。許多原本 prepare 正確的 trace，在 re-entry 後被破壞。

### LiveBench Qwen3

```text
bucket   rescue_rate_wrong   damage_rate_ok
p01      13.3%               50.0%
p02      20.0%               50.0%
p03      13.3%               41.7%
p04-05   12.0%               38.1%
p06-10   20.5%               36.4%
p11-20   10.0%               22.2%
```

解讀：

> Qwen3 同時有 rescue 和 damage。它的 hidden-state stability projection 能幫助區分哪些 prefix 比較像會成功接續，哪些比較像會破壞原本正確 trace。

## 如何自己重算

從 repo root 執行：

```powershell
cd C:\Users\admin\Desktop\Experiment\confidence-tom
```

資料 join key：

```python
def key(row):
    return (row.get("benchmark"), row.get("run_name"), row.get("prefix_id"))
```

trace category：

```python
trace = int(row.get("full_trace_correct") or 0)
exact = int(row.get("reentry_exact_correct") or row.get("small_continue_correct") or 0)

if trace and exact:
    category = "preserve"
elif trace and not exact:
    category = "trace_damage"
elif (not trace) and exact:
    category = "trace_rescue"
else:
    category = "persistent_fail"
```

stable/unstable label：

```python
stable = category in {"preserve", "trace_rescue"}
unstable = category in {"trace_damage", "persistent_fail"}
```

hidden norm：

```python
norm = sqrt(sum(x * x for x in last_token_hidden))
```

centroid：

```python
centroid = mean(hidden_vectors)
```

stability axis：

```python
stable_centroid = mean(hidden states where stable)
unstable_centroid = mean(hidden states where unstable)
axis = stable_centroid - unstable_centroid
unit_axis = axis / norm(axis)
projection = dot(hidden_state, unit_axis)
```

rescue/damage axis：

```python
rescue_centroid = mean(hidden states where category == "trace_rescue")
damage_centroid = mean(hidden states where category == "trace_damage")
axis = rescue_centroid - damage_centroid
projection = dot(hidden_state, unit(axis))
```

centroid 顯著性用 permutation test：

```python
observed = distance(mean(group_a), mean(group_b))

for each permutation:
    shuffle group labels
    recompute distance

p = fraction(permuted_distance >= observed_distance)
```

hidden norm 顯著性：

```text
Welch t-test on hidden norms
Mann-Whitney U as a non-parametric check
Benjamini-Hochberg FDR correction across model comparisons
```

## Minimal Recompute Script

以下 script 會重算 stable/unstable projection table。

```powershell
@'
import json, math
from pathlib import Path
from collections import defaultdict

base = Path("outputs/results")
reentry_files = [base / "reentry/_reentry_math500_local_v1/reentry_rows.jsonl"]
reentry_files += [
    d / "reentry_rows.jsonl"
    for d in (base / "reentry").glob("_reentry_livebench_local_v1_*_no_outliers")
    if (d / "reentry_rows.jsonl").exists()
]
probe_files = [base / "probe/_reentry_math500_local_v1/reentry_probe_rows.jsonl"]
probe_files += [
    d / "reentry_probe_rows.jsonl"
    for d in (base / "probe").glob("_reentry_livebench_local_v1_*_no_outliers")
    if (d / "reentry_probe_rows.jsonl").exists()
]

def key(r):
    return (r.get("benchmark"), r.get("run_name"), r.get("prefix_id"))

labels = {}
for path in reentry_files:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("segment_count_outlier") or 0):
            continue
        trace = int(r.get("full_trace_correct") or 0)
        exact = int(r.get("reentry_exact_correct") or r.get("small_continue_correct") or 0)
        if trace and exact:
            cat = "preserve"
        elif trace and not exact:
            cat = "trace_damage"
        elif (not trace) and exact:
            cat = "trace_rescue"
        else:
            cat = "persistent_fail"
        labels[key(r)] = {
            "benchmark": r.get("benchmark"),
            "family": r.get("small_family") or r.get("run_name", "").split("_")[2],
            "category": cat,
            "stable": cat in {"preserve", "trace_rescue"},
        }

rows = []
for path in probe_files:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("segment_count_outlier") or 0):
            continue
        label = labels.get(key(r))
        if not label:
            continue
        h = r.get("last_token_hidden")
        if h is None:
            continue
        h = [float(x) for x in h]
        rows.append({**label, "h": h})

def centroid(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]

def sub(a, b):
    return [x - y for x, y in zip(a, b)]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(sum(x * x for x in a))

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

groups = defaultdict(list)
for r in rows:
    groups[(r["benchmark"], r["family"])].append(r)

print("benchmark\\tfamily\\tcategory\\tn\\tmean_stability_projection")
for (benchmark, family), group in sorted(groups.items()):
    stable = [r["h"] for r in group if r["stable"]]
    unstable = [r["h"] for r in group if not r["stable"]]
    if len(stable) < 2 or len(unstable) < 2:
        continue
    axis = sub(centroid(stable), centroid(unstable))
    axis_norm = norm(axis)
    unit_axis = [x / axis_norm for x in axis]
    by_category = defaultdict(list)
    for r in group:
        by_category[r["category"]].append(dot(r["h"], unit_axis))
    for category in ["preserve", "trace_rescue", "trace_damage", "persistent_fail"]:
        values = by_category.get(category, [])
        if values:
            print(f"{benchmark}\\t{family}\\t{category}\\t{len(values)}\\t{mean(values):.3f}")
'@ | .\.venv\Scripts\python.exe -
```

## 最終解讀

不要把 hidden-state 結果寫成：

> 模型有一個清楚的 rescue neuron。

比較正確的寫法是：

> Prefix reliability is reflected as a model-specific high-dimensional direction in hidden-state space. Stable prefixes tend to project higher along this direction, while unstable prefixes project lower. This separation is statistically significant in most model/benchmark pairs, although the point clouds still overlap.

中文意思：

> prefix 是否穩定，會反映在模型 hidden-state space 裡一條 model-specific 的高維方向上。穩定 prefix 通常在這條方向上投影較高，不穩定 prefix 投影較低。多數模型和 benchmark 中，這個分離具有統計顯著性，但點雲仍然重疊，不能說是完全乾淨分類。

目前最有力的故事是：

1. **Math500** 顯示 prefix 具有有用的 scaffold 效果。
2. **LiveBench** 顯示 prefix fragility，也就是 re-entry 容易破壞原本正確 trace。
3. **Hidden states** 提供弱但可測的證據：stable 和 unstable prefixes 的內部狀態不同。
4. 最可解釋的 hidden-state 特徵不是單一 dimension，而是投影到 model-specific **stability axis**。

