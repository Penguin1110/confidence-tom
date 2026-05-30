# Re-entry Hidden-State Analysis

Date: 2026-05-27

This note summarizes the current re-entry experiment from three angles:

1. What the experiment is measuring.
2. How to compute the metrics again.
3. What the hidden states suggest about stable, unstable, rescue, and damage prefixes.

The key convention in this report is that **rescue is trace-based**:

```text
trace_rescue = prepare/full_trace_correct == 0 and reentry_exact_correct == 1
trace_damage = prepare/full_trace_correct == 1 and reentry_exact_correct == 0
preserve = prepare/full_trace_correct == 1 and reentry_exact_correct == 1
persistent_fail = prepare/full_trace_correct == 0 and reentry_exact_correct == 0
```

This is different from using `full_rerun_correct`. The main question is whether a prefix from an originally wrong prepare trace can still lead the model back to a correct answer.

## Data Used

The analysis uses non-outlier re-entry rows and their corresponding probe rows:

```text
outputs/results/reentry/_reentry_math500_local_v1/reentry_rows.jsonl
outputs/results/probe/_reentry_math500_local_v1/reentry_probe_rows.jsonl

outputs/results/reentry/_reentry_livebench_local_v1_*_no_outliers/reentry_rows.jsonl
outputs/results/probe/_reentry_livebench_local_v1_*_no_outliers/reentry_probe_rows.jsonl
```

Rows are joined by:

```text
(benchmark, run_name, prefix_id)
```

Rows with `segment_count_outlier == 1` are excluded.

## What A Hidden State Means Here

For each prefix, the probe records the model's **last-layer, last-token hidden state** after reading the re-entry prompt.

In plain terms:

```text
hidden_state(prefix) = the model's internal state after it has read the prefix
```

Each hidden state is a high-dimensional vector:

```text
h = [x1, x2, x3, ..., xd]
```

One prefix becomes one point in a high-dimensional space. We then label that point by the re-entry outcome:

```text
preserve
trace_rescue
trace_damage
persistent_fail
```

Important: hidden dimensions are not directly human-readable. A single dimension should not be interpreted as "confidence", "math skill", or "correctness". The more reliable object is a direction or centroid in hidden-state space.

## Main Experiment Result

### Math500

Math500 supports the prefix-scaffold hypothesis.

Non-outlier trace-based results:

```text
trace_rescue = 104 / 467 = 22.3%
trace_damage = 41 / 482 = 8.5%
preserve = 441
persistent_fail = 363
net = +63
```

By model:

```text
model     trace_rescue        trace_damage       interpretation
gemma     34/72 = 47.2%       12/172 = 7.0%      strongest rescue signal
mistral   29/194 = 14.9%      1/60 = 1.7%        low damage
olmo      39/194 = 20.1%      0/25 = 0.0%        rescue without damage
qwen      2/7 = 28.6%         28/225 = 12.4%     prepare already strong; little rescue room
```

Conclusion:

> In Math500, many prefixes act like useful reasoning scaffolds. Even if the original prepare trace ended incorrectly, intermediate prefixes can still guide the model back to the correct answer.

### LiveBench Reasoning

LiveBench is much more fragile.

Non-outlier trace-based results:

```text
trace_rescue = 61 / 715 = 8.5%
trace_damage = 177 / 452 = 39.2%
preserve = 275
persistent_fail = 654
net = -116
```

By model:

```text
model       trace_rescue        trace_damage       interpretation
gemma4      12/144 = 8.3%       97/257 = 37.7%     high damage
qwen3       19/119 = 16.0%      43/117 = 36.8%     both rescue and damage
qwen25      10/93 = 10.8%       18/45 = 40.0%      unstable
ministral   7/167 = 4.2%       10/23 = 43.5%      weak rescue
mistral7    13/192 = 6.8%      9/10 = 90.0%       tiny prepare-correct denominator
```

Conclusion:

> LiveBench prefixes are less reliable as scaffolds. Re-entry often disrupts originally correct traces, so the method exposes prefix fragility rather than robust recoverability.

## Hidden-State View

There are three useful ways to look at hidden states.

### 1. Hidden Norm

The hidden norm is:

```text
norm(h) = sqrt(sum_i h_i^2)
```

It measures vector length. It is only meaningful **within the same model**, not across models.

Wrong comparison:

```text
Qwen norm 170 vs Gemma norm 240
```

Right comparison:

```text
Qwen rescue norm vs Qwen damage norm
Gemma stable norm vs Gemma unstable norm
```

Norm-level finding:

```text
LiveBench qwen3:
rescue norm = 173.0
damage norm = 166.7
Cohen's d = 2.026
Welch p = 5.7e-07
FDR q = 4.0e-06
```

This is the clearest norm-level rescue/damage signal.

For most other model/benchmark pairs, norm alone is not enough.

### 2. Rescue-Damage Centroid

For each model, define:

```text
rescue_centroid = mean(hidden states of trace_rescue rows)
damage_centroid = mean(hidden states of trace_damage rows)

rescue_damage_axis = rescue_centroid - damage_centroid
```

Then project each hidden state onto this axis:

```text
projection(h) = dot(h, unit(rescue_damage_axis))
```

Higher projection means more rescue-like. Lower projection means more damage-like.

Centroid permutation tests showed significant rescue/damage separation for:

```text
LiveBench gemma4      p ~= 0.003
LiveBench ministral   p ~= 0.007
LiveBench mistral7    p ~= 0.040
LiveBench qwen3       p ~= 0.003
Math500 gemma         p ~= 0.003
```

Not significant:

```text
LiveBench qwen25      p ~= 0.425
```

Insufficient for this comparison:

```text
Math500 mistral       too few damage rows
Math500 olmo          no damage rows
Math500 qwen          too few rescue rows
```

Interpretation:

> Rescue and damage are not cleanly separated by hidden norm, but several models show a statistically reliable high-dimensional centroid shift.

### 3. Stable-Unstable Centroid

For the broader reliability question, define:

```text
stable = preserve + trace_rescue
unstable = trace_damage + persistent_fail

stable_centroid = mean(hidden states of stable rows)
unstable_centroid = mean(hidden states of unstable rows)

stability_axis = stable_centroid - unstable_centroid
```

Then:

```text
projection(h) = dot(h, unit(stability_axis))
```

Higher projection means the prefix hidden state is more stable-like. Lower projection means it is more unstable-like.

This is the most useful lens for asking:

> What hidden-state feature do especially stable or unstable prefixes have?

## Stable vs Unstable Hidden-State Result

Most model/benchmark pairs show significant stable/unstable centroid separation:

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

Only Math500 qwen does not show a clear stable/unstable centroid separation in this analysis.

### Stable-Like Projection

Mean projection onto the stability axis:

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

This gives a simple hidden-state feature:

```text
stable prefixes have higher projection onto the model-specific stability axis
unstable prefixes have lower projection onto that axis
```

This feature is high-dimensional. It is not a single human-readable neuron.

## Category Projection

Projection onto the stability axis by trace category:

### Math500 Gemma

```text
category          n     mean projection
preserve          160   +36.0
trace_rescue      34    +55.6
trace_damage      12    -11.3
persistent_fail   38    -10.1
```

This is the cleanest structure:

```text
stable side:   preserve, trace_rescue
unstable side: trace_damage, persistent_fail
```

Interpretation:

> Math500 Gemma has a strong internal stability direction. Rescue and preserve prefixes lie on the stable side; damage and persistent failure lie on the unstable side.

### LiveBench Qwen3

```text
category          n     mean projection
preserve          74    +46.9
trace_rescue      19    +30.8
trace_damage      43    +14.7
persistent_fail   100   +9.8
```

This forms an ordered reliability gradient:

```text
preserve > trace_rescue > trace_damage > persistent_fail
```

Interpretation:

> LiveBench Qwen3 does not split categories into two clean sides, but it has a strong monotonic stability ordering.

### LiveBench Gemma4

```text
category          n     mean projection
preserve          160   +52.6
trace_rescue      12    +32.2
trace_damage      97    -3.9
persistent_fail   132   +7.4
```

Interpretation:

> Preserve and rescue are more stable-like than damage. Persistent failures are mixed, which suggests some wrong traces may still occupy regions near the stable direction but fail for other reasons.

### Math500 OLMo

```text
category          n     mean projection
preserve          25    +14.5
trace_rescue      39    +13.5
persistent_fail   155   -1.4
```

There are no damage rows for Math500 OLMo.

Interpretation:

> OLMo's stable outcomes preserve/rescue occupy a higher-stability region, while persistent failures lie lower on the stability axis.

## Prefix Stability

The earlier step-bucket analysis is useful for locating instability, but the deeper conclusion is not "step number causes instability". The better conclusion is:

> Some prefixes at any step can land in a stable-like or unstable-like internal state. Step buckets only summarize where those states tend to appear more often.

Damage rate is:

```text
damage_rate = trace_damage / prepare-correct prefixes
```

Rescue rate is:

```text
rescue_rate = trace_rescue / prepare-wrong prefixes
```

Examples:

### Math500 Gemma

```text
bucket   rescue_rate_wrong   damage_rate_ok
p01      57.1%               18.2%
p02      50.0%               12.1%
p03      57.1%               3.0%
p04-05   45.5%               1.9%
p06-10   12.5%               0.0%
```

This means Math500 Gemma has many rescue-capable prefixes and little damage after early prefixes.

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

This means LiveBench Gemma4 is fragile early: correct prepare traces are often damaged by re-entry.

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

Qwen3 has both rescue and damage. Its hidden-state stability projection helps separate the two.

## How To Recompute

Run from the repository root:

```powershell
cd C:\Users\admin\Desktop\Experiment\confidence-tom
```

The basic data join is:

```python
def key(row):
    return (row.get("benchmark"), row.get("run_name"), row.get("prefix_id"))
```

The trace category is:

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

The stable/unstable label is:

```python
stable = category in {"preserve", "trace_rescue"}
unstable = category in {"trace_damage", "persistent_fail"}
```

The hidden norm is:

```python
norm = sqrt(sum(x * x for x in last_token_hidden))
```

The centroid is:

```python
centroid = mean(hidden_vectors)
```

The stability axis is:

```python
stable_centroid = mean(hidden states where stable)
unstable_centroid = mean(hidden states where unstable)
axis = stable_centroid - unstable_centroid
unit_axis = axis / norm(axis)
projection = dot(hidden_state, unit_axis)
```

For rescue/damage specifically:

```python
rescue_centroid = mean(hidden states where category == "trace_rescue")
damage_centroid = mean(hidden states where category == "trace_damage")
axis = rescue_centroid - damage_centroid
projection = dot(hidden_state, unit(axis))
```

For centroid significance, use a permutation test:

```python
observed = distance(mean(group_a), mean(group_b))

for each permutation:
    shuffle group labels
    recompute distance

p = fraction(permuted_distance >= observed_distance)
```

For norm significance, use:

```python
Welch t-test on hidden norms
Mann-Whitney U as a non-parametric check
Benjamini-Hochberg FDR correction across model comparisons
```

## Minimal Recompute Script

This script recomputes the stable/unstable projection table.

```powershell
@'
import json, math, statistics
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

## Final Interpretation

The hidden-state result should not be stated as:

> The model has a clear hidden neuron for rescue.

A more accurate statement is:

> Prefix reliability is reflected as a model-specific high-dimensional direction in hidden-state space. Stable prefixes tend to project higher along this direction, while unstable prefixes project lower. This separation is statistically significant in most model/benchmark pairs, although the point clouds still overlap.

The current strongest story is:

1. **Math500** shows useful scaffold behavior.
2. **LiveBench** shows prefix fragility.
3. **Hidden states** provide weak but measurable evidence that stable and unstable prefixes occupy different internal states.
4. The most interpretable hidden-state feature is not a single dimension, but projection onto a model-specific **stability axis**.

