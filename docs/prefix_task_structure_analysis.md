# Prefix Task Structure Analysis

## 方法

這份分析做兩件事：

1. 用 task-level `mean positive fraction` 和 `variance positive fraction` 建立 rank / variance map。
2. 用 task-level 統計特徵做簡單 clustering，先看資料自然長出哪些群。

## Rank / Variance Map

| Task | Positive Runs | Mean Positive Fraction | Variance | Mean Negative Fraction |
| --- | ---: | ---: | ---: | ---: |
| `olympiadbench_2127_0029` | 6/6 | 0.764 | 0.0572 | 0.000 |
| `olympiadbench_1977_0022` | 6/6 | 0.629 | 0.1379 | 0.000 |
| `olympiadbench_2427_0030` | 3/6 | 0.500 | 0.2500 | 0.000 |
| `olympiadbench_3037_0028` | 3/6 | 0.400 | 0.2000 | 0.000 |
| `olympiadbench_2560_0008` | 3/6 | 0.392 | 0.1537 | 0.000 |
| `olympiadbench_1866_0034` | 4/6 | 0.381 | 0.1065 | 0.125 |
| `olympiadbench_2744_0031` | 4/6 | 0.379 | 0.1282 | 0.000 |
| `olympiadbench_3080_0011` | 3/6 | 0.367 | 0.2056 | 0.042 |
| `olympiadbench_2973_0006` | 4/6 | 0.345 | 0.1227 | 0.042 |
| `olympiadbench_2897_0000` | 3/6 | 0.342 | 0.1253 | 0.000 |
| `olympiadbench_2257_0016` | 2/6 | 0.333 | 0.2222 | 0.000 |
| `olympiadbench_2666_0010` | 5/6 | 0.331 | 0.0997 | 0.000 |
| `olympiadbench_2292_0020` | 3/6 | 0.330 | 0.1150 | 0.000 |
| `olympiadbench_3008_0035` | 4/6 | 0.308 | 0.0753 | 0.000 |
| `olympiadbench_1798_0047` | 5/6 | 0.267 | 0.0622 | 0.244 |

### Highest Variance Tasks

| Task | Positive Runs | Mean Positive Fraction | Variance |
| --- | ---: | ---: | ---: |
| `olympiadbench_2427_0030` | 3/6 | 0.500 | 0.2500 |
| `olympiadbench_2257_0016` | 2/6 | 0.333 | 0.2222 |
| `olympiadbench_3080_0011` | 3/6 | 0.367 | 0.2056 |
| `olympiadbench_3037_0028` | 3/6 | 0.400 | 0.2000 |
| `olympiadbench_2560_0008` | 3/6 | 0.392 | 0.1537 |
| `olympiadbench_1977_0022` | 6/6 | 0.629 | 0.1379 |
| `olympiadbench_2744_0031` | 4/6 | 0.379 | 0.1282 |
| `olympiadbench_2897_0000` | 3/6 | 0.342 | 0.1253 |
| `olympiadbench_2973_0006` | 4/6 | 0.345 | 0.1227 |
| `olympiadbench_2292_0020` | 3/6 | 0.330 | 0.1150 |
| `olympiadbench_1866_0034` | 4/6 | 0.381 | 0.1065 |
| `olympiadbench_2741_0002` | 2/6 | 0.223 | 0.1003 |
| `olympiadbench_2666_0010` | 5/6 | 0.331 | 0.0997 |
| `olympiadbench_2712_0018` | 2/6 | 0.189 | 0.0895 |
| `olympiadbench_3008_0035` | 4/6 | 0.308 | 0.0753 |

## Clustering

### Cluster 0: `low_gain_stable` (size=8)

Centroid:
- `mean_positive_fraction`: 0.034
- `variance_positive_fraction`: 0.010
- `mean_negative_fraction`: 0.013
- `agreement_any_positive_rate`: 0.104
- `mean_step_count`: 7.292

Representative tasks:
- `olympiadbench_2667_0039`: mean_pos=0.178, var=0.0691, neg=0.000, positive_runs=2/6
- `olympiadbench_2974_0046`: mean_pos=0.062, var=0.0090, neg=0.000, positive_runs=2/6
- `olympiadbench_2349_0032`: mean_pos=0.033, var=0.0056, neg=0.000, positive_runs=1/6
- `olympiadbench_2084_0015`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6
- `olympiadbench_2453_0027`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6
- `olympiadbench_2458_0005`: mean_pos=0.000, var=0.0000, neg=0.106, positive_runs=0/6
- `olympiadbench_2524_0013`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6
- `olympiadbench_3018_0001`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6

### Cluster 1: `pairing_sensitive` (size=15)

Centroid:
- `mean_positive_fraction`: 0.401
- `variance_positive_fraction`: 0.140
- `mean_negative_fraction`: 0.014
- `agreement_any_positive_rate`: 0.611
- `mean_step_count`: 5.878

Representative tasks:
- `olympiadbench_2127_0029`: mean_pos=0.764, var=0.0572, neg=0.000, positive_runs=6/6
- `olympiadbench_1977_0022`: mean_pos=0.629, var=0.1379, neg=0.000, positive_runs=6/6
- `olympiadbench_2427_0030`: mean_pos=0.500, var=0.2500, neg=0.000, positive_runs=3/6
- `olympiadbench_3037_0028`: mean_pos=0.400, var=0.2000, neg=0.000, positive_runs=3/6
- `olympiadbench_2560_0008`: mean_pos=0.392, var=0.1537, neg=0.000, positive_runs=3/6
- `olympiadbench_1866_0034`: mean_pos=0.381, var=0.1065, neg=0.125, positive_runs=4/6
- `olympiadbench_2744_0031`: mean_pos=0.379, var=0.1282, neg=0.000, positive_runs=4/6
- `olympiadbench_3080_0011`: mean_pos=0.367, var=0.2056, neg=0.042, positive_runs=3/6

### Cluster 2: `low_gain_stable` (size=21)

Centroid:
- `mean_positive_fraction`: 0.015
- `variance_positive_fraction`: 0.004
- `mean_negative_fraction`: 0.001
- `agreement_any_positive_rate`: 0.048
- `mean_step_count`: 4.833

Representative tasks:
- `olympiadbench_2006_0033`: mean_pos=0.158, var=0.0337, neg=0.000, positive_runs=3/6
- `olympiadbench_2075_0009`: mean_pos=0.117, var=0.0347, neg=0.000, positive_runs=2/6
- `olympiadbench_3099_0012`: mean_pos=0.042, var=0.0087, neg=0.000, positive_runs=1/6
- `olympiadbench_1915_0023`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6
- `olympiadbench_1962_0044`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6
- `olympiadbench_2012_0007`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6
- `olympiadbench_2268_0040`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6
- `olympiadbench_2280_0036`: mean_pos=0.000, var=0.0000, neg=0.000, positive_runs=0/6

### Cluster 3: `negative_risk` (size=6)

Centroid:
- `mean_positive_fraction`: 0.175
- `variance_positive_fraction`: 0.052
- `mean_negative_fraction`: 0.216
- `agreement_any_positive_rate`: 0.472
- `mean_step_count`: 4.889

Representative tasks:
- `olympiadbench_1798_0047`: mean_pos=0.267, var=0.0622, neg=0.244, positive_runs=5/6
- `olympiadbench_2712_0018`: mean_pos=0.189, var=0.0895, neg=0.236, positive_runs=2/6
- `olympiadbench_2844_0041`: mean_pos=0.181, var=0.0380, neg=0.253, positive_runs=3/6
- `olympiadbench_2984_0003`: mean_pos=0.158, var=0.0337, neg=0.083, positive_runs=3/6
- `olympiadbench_1750_0024`: mean_pos=0.157, var=0.0375, neg=0.222, positive_runs=3/6
- `olympiadbench_2479_0026`: mean_pos=0.100, var=0.0500, neg=0.258, positive_runs=1/6
