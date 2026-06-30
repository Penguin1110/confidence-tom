# Hidden State Geometry

## Question

- Treat hidden states as trajectories across prefixes.
- Compare state stability, drift, and similarity-to-final across trace categories.

## Aggregate Late-Success Minus Late-Failure Gaps

| Pooling | First-correct frac gap | Local correct rate gap | Last-small-correct gap | Consecutive cosine gap | Cosine-to-final gap | Mean displacement gap | Max displacement gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_hidden | 0.279 | -0.029 | 0.651 | 0.003 | 0.021 | -1.205 | -7.258 |
| last_token_hidden | 0.279 | -0.029 | 0.651 | 0.000 | 0.011 | 2.133 | -5.491 |

## Family Breakdown

### `gemma4_no_outliers`

- tasks: `29`
- selected layer: `42`
- hidden dim: `2560`
- category counts: `{"late-success": 117, "persistent-failure": 105, "stable-success": 140, "late-failure": 39}`

#### `mean_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 9 | 0.120 | 0.821 | 1.000 | 0.998 | 0.973 | 4.805 |
| late-success | 9 | 0.600 | 0.350 | 1.000 | 0.991 | 0.927 | 8.780 |
| late-failure | 4 | 0.710 | 0.315 | 0.750 | 0.990 | 0.905 | 9.261 |
| persistent-failure | 7 | 1.000 | 0.000 | 0.000 | 0.989 | 0.943 | 9.358 |

- late-success minus late-failure:
  first-correct frac `-0.110`, local correct rate `0.034`, last-small-correct `0.250`
  consecutive cosine `0.001`, cosine-to-final `0.022`, mean displacement `-0.480`, max displacement `-7.886`

#### `last_token_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 9 | 0.120 | 0.821 | 1.000 | 0.926 | 0.739 | 85.496 |
| late-success | 9 | 0.600 | 0.350 | 1.000 | 0.906 | 0.660 | 91.274 |
| late-failure | 4 | 0.710 | 0.315 | 0.750 | 0.908 | 0.623 | 84.353 |
| persistent-failure | 7 | 1.000 | 0.000 | 0.000 | 0.889 | 0.709 | 102.538 |

- late-success minus late-failure:
  first-correct frac `-0.110`, local correct rate `0.034`, last-small-correct `0.250`
  consecutive cosine `-0.002`, cosine-to-final `0.037`, mean displacement `6.921`, max displacement `0.281`

### `ministral_no_outliers`

- tasks: `29`
- selected layer: `36`
- hidden dim: `4096`
- category counts: `{"persistent-failure": 139, "stable-success": 6, "late-success": 17, "late-failure": 28}`

#### `mean_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 1 | 0.167 | 0.833 | 1.000 | 0.999 | 0.991 | 6.209 |
| late-success | 3 | 0.658 | 0.492 | 1.000 | 0.996 | 0.983 | 10.927 |
| late-failure | 3 | 0.428 | 0.259 | 0.000 | 0.999 | 0.988 | 6.726 |
| persistent-failure | 22 | 1.000 | 0.000 | 0.000 | 0.998 | 0.990 | 6.431 |

- late-success minus late-failure:
  first-correct frac `0.230`, local correct rate `0.233`, last-small-correct `1.000`
  consecutive cosine `-0.003`, cosine-to-final `-0.005`, mean displacement `4.201`, max displacement `6.105`

#### `last_token_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 1 | 0.167 | 0.833 | 1.000 | 0.971 | 0.940 | 65.577 |
| late-success | 3 | 0.658 | 0.492 | 1.000 | 0.965 | 0.937 | 72.936 |
| late-failure | 3 | 0.428 | 0.259 | 0.000 | 0.969 | 0.933 | 67.591 |
| persistent-failure | 22 | 1.000 | 0.000 | 0.000 | 0.973 | 0.943 | 57.615 |

- late-success minus late-failure:
  first-correct frac `0.230`, local correct rate `0.233`, last-small-correct `1.000`
  consecutive cosine `-0.004`, cosine-to-final `0.004`, mean displacement `5.345`, max displacement `-1.544`

### `mistral7_no_outliers`

- tasks: `29`
- selected layer: `32`
- hidden dim: `4096`
- category counts: `{"late-failure": 50, "persistent-failure": 142, "late-success": 10}`

#### `mean_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-success | 1 | 1.000 | 0.100 | 1.000 | 0.997 | 0.967 | 11.256 |
| late-failure | 7 | 0.392 | 0.306 | 0.143 | 0.991 | 0.973 | 14.759 |
| persistent-failure | 21 | 1.000 | 0.000 | 0.000 | 0.996 | 0.984 | 11.495 |

- late-success minus late-failure:
  first-correct frac `0.608`, local correct rate `-0.206`, last-small-correct `0.857`
  consecutive cosine `0.006`, cosine-to-final `-0.006`, mean displacement `-3.503`, max displacement `-15.295`

#### `last_token_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-success | 1 | 1.000 | 0.100 | 1.000 | 0.910 | 0.811 | 151.652 |
| late-failure | 7 | 0.392 | 0.306 | 0.143 | 0.914 | 0.843 | 143.588 |
| persistent-failure | 21 | 1.000 | 0.000 | 0.000 | 0.908 | 0.847 | 149.793 |

- late-success minus late-failure:
  first-correct frac `0.608`, local correct rate `-0.206`, last-small-correct `0.857`
  consecutive cosine `-0.004`, cosine-to-final `-0.032`, mean displacement `8.064`, max displacement `0.860`

### `qwen25_no_outliers`

- tasks: `27`
- selected layer: `48`
- hidden dim: `5120`
- category counts: `{"late-failure": 40, "persistent-failure": 53, "stable-success": 22, "late-success": 23}`

#### `mean_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 5 | 0.230 | 0.920 | 1.000 | 0.997 | 0.991 | 9.044 |
| late-success | 4 | 0.640 | 0.327 | 0.750 | 0.999 | 0.996 | 5.324 |
| late-failure | 7 | 0.450 | 0.271 | 0.000 | 0.997 | 0.991 | 8.677 |
| persistent-failure | 11 | 1.000 | 0.000 | 0.000 | 0.998 | 0.992 | 7.954 |

- late-success minus late-failure:
  first-correct frac `0.190`, local correct rate `0.057`, last-small-correct `0.750`
  consecutive cosine `0.002`, cosine-to-final `0.005`, mean displacement `-3.353`, max displacement `-4.402`

#### `last_token_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 5 | 0.230 | 0.920 | 1.000 | 0.963 | 0.915 | 66.618 |
| late-success | 4 | 0.640 | 0.327 | 0.750 | 0.967 | 0.924 | 63.882 |
| late-failure | 7 | 0.450 | 0.271 | 0.000 | 0.960 | 0.908 | 69.759 |
| persistent-failure | 11 | 1.000 | 0.000 | 0.000 | 0.960 | 0.902 | 68.302 |

- late-success minus late-failure:
  first-correct frac `0.190`, local correct rate `0.057`, last-small-correct `0.750`
  consecutive cosine `0.007`, cosine-to-final `0.016`, mean displacement `-5.877`, max displacement `-10.952`

### `qwen3_no_outliers`

- tasks: `27`
- selected layer: `40`
- hidden dim: `5120`
- category counts: `{"late-failure": 45, "persistent-failure": 74, "late-success": 51, "stable-success": 66}`

#### `mean_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 6 | 0.135 | 0.958 | 1.000 | 0.997 | 0.993 | 5.991 |
| late-success | 6 | 0.900 | 0.246 | 1.000 | 0.997 | 0.988 | 5.652 |
| late-failure | 5 | 0.423 | 0.511 | 0.600 | 0.987 | 0.902 | 8.538 |
| persistent-failure | 10 | 1.000 | 0.000 | 0.000 | 0.979 | 0.894 | 12.169 |

- late-success minus late-failure:
  first-correct frac `0.477`, local correct rate `-0.265`, last-small-correct `0.400`
  consecutive cosine `0.010`, cosine-to-final `0.086`, mean displacement `-2.887`, max displacement `-14.811`

#### `last_token_hidden`

| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 6 | 0.135 | 0.958 | 1.000 | 0.980 | 0.952 | 32.815 |
| late-success | 6 | 0.900 | 0.246 | 1.000 | 0.982 | 0.954 | 29.547 |
| late-failure | 5 | 0.423 | 0.511 | 0.600 | 0.978 | 0.926 | 33.334 |
| persistent-failure | 10 | 1.000 | 0.000 | 0.000 | 0.965 | 0.894 | 40.356 |

- late-success minus late-failure:
  first-correct frac `0.477`, local correct rate `-0.265`, last-small-correct `0.400`
  consecutive cosine `0.005`, cosine-to-final `0.028`, mean displacement `-3.787`, max displacement `-16.101`
