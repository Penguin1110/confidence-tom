# Prefix Embedding Pilot

## 設定

- 樣本數：`480`
- embedding model：`google/gemini-embedding-001`
- embedding dim：`3072`

## Probe 結果

### `delta_positive`

- train/test：`405` / `75`
- train/test base rate：`0.272` / `0.147`
- test AUROC：`0.396`
- test F1：`0.182`
- test precision / recall：`0.114` / `0.455`

### `benchmark_is_livebench`

- train/test：`405` / `75`
- train/test base rate：`0.543` / `0.267`
- test AUROC：`1.000`
- test F1：`0.930`
- test precision / recall：`0.870` / `1.000`

### `small_family_is_llama`

- train/test：`405` / `75`
- train/test base rate：`0.331` / `0.347`
- test AUROC：`0.876`
- test F1：`0.676`
- test precision / recall：`0.548` / `0.885`

## PCA

- explained variance ratio (PC1/PC2)：`0.137` / `0.073`
