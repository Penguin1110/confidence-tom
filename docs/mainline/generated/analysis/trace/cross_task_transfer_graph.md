# Cross-Task Transfer Graph

## Question

- Build a graph where an edge `A -> B` means A's final correct hidden state predicts B's correct-vs-incorrect prefix distribution with AUROC >= 0.70.
- This asks whether some tasks share a transferable hidden-state correctness geometry.

## Aggregate

| Pooling | Mean cross-task AUROC | Mean edge density | Mean largest component share |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.490 | 0.373 | 0.921 |
| last_token_hidden | 0.720 | 0.627 | 0.950 |

## Family Graphs

### `gemma4_no_outliers`

#### `mean_hidden`

- tasks: `17`
- mean cross-task AUROC: `0.677`
- strong edge count: `121`
- edge density: `0.445`
- largest component: size `15`, edges `121`, categories `{"late-success": 9, "stable-success": 6}`

Top hubs:
- `a3880d3d77b8` (late-success): out=0.781, strong=9
- `43a1e0948c6a` (late-success): out=0.778, strong=9
- `3de6bc30b87b` (late-success): out=0.777, strong=9
- `caa1f15441c6` (stable-success): out=0.777, strong=9

Top receivers:
- `56ecba920668` (stable-success): in=0.906, strong=14
- `43a1e0948c6a` (late-success): in=0.870, strong=14
- `4f4cccee79cb` (late-success): in=0.852, strong=13
- `a3880d3d77b8` (late-success): in=0.821, strong=14

Top edges:
- `3de6bc30b87b` -> `caa1f15441c6`: AUROC=1.000, gap=0.022
- `3de6bc30b87b` -> `9b2101675433`: AUROC=1.000, gap=0.117
- `3de6bc30b87b` -> `4f4cccee79cb`: AUROC=1.000, gap=0.112
- `3de6bc30b87b` -> `56ecba920668`: AUROC=1.000, gap=0.091
- `3de6bc30b87b` -> `43a1e0948c6a`: AUROC=1.000, gap=0.101

#### `last_token_hidden`

- tasks: `17`
- mean cross-task AUROC: `0.785`
- strong edge count: `179`
- edge density: `0.658`
- largest component: size `17`, edges `179`, categories `{"late-success": 9, "stable-success": 8}`

Top hubs:
- `a0f4b0be2055` (stable-success): out=0.830, strong=13
- `94b238d5f10a` (stable-success): out=0.804, strong=10
- `4e049d9d8570` (late-success): out=0.801, strong=11
- `4f4cccee79cb` (late-success): out=0.800, strong=13

Top receivers:
- `9b2101675433` (late-success): in=1.000, strong=16
- `43a1e0948c6a` (late-success): in=1.000, strong=16
- `149f73699e33` (late-success): in=1.000, strong=16
- `56ecba920668` (stable-success): in=0.994, strong=16

Top edges:
- `3de6bc30b87b` -> `9b2101675433`: AUROC=1.000, gap=0.311
- `3de6bc30b87b` -> `4f4cccee79cb`: AUROC=1.000, gap=0.208
- `3de6bc30b87b` -> `56ecba920668`: AUROC=1.000, gap=0.188
- `3de6bc30b87b` -> `43a1e0948c6a`: AUROC=1.000, gap=0.249
- `3de6bc30b87b` -> `149f73699e33`: AUROC=1.000, gap=0.293

### `ministral_no_outliers`

#### `mean_hidden`

- tasks: `4`
- mean cross-task AUROC: `0.428`
- strong edge count: `5`
- edge density: `0.417`
- largest component: size `4`, edges `5`, categories `{"late-success": 3, "stable-success": 1}`

Top hubs:
- `43a1e0948c6a` (late-success): out=0.933, strong=3
- `4e049d9d8570` (stable-success): out=0.733, strong=2
- `b183db3494f4` (late-success): out=0.022, strong=0
- `a0f4b0be2055` (late-success): out=0.022, strong=0

Top receivers:
- `b183db3494f4` (late-success): in=0.667, strong=2
- `a0f4b0be2055` (late-success): in=0.667, strong=2
- `4e049d9d8570` (stable-success): in=0.267, strong=1
- `43a1e0948c6a` (late-success): in=0.111, strong=0

Top edges:
- `4e049d9d8570` -> `b183db3494f4`: AUROC=1.000, gap=0.004
- `4e049d9d8570` -> `a0f4b0be2055`: AUROC=1.000, gap=0.001
- `43a1e0948c6a` -> `b183db3494f4`: AUROC=1.000, gap=0.011
- `43a1e0948c6a` -> `a0f4b0be2055`: AUROC=1.000, gap=0.014
- `43a1e0948c6a` -> `4e049d9d8570`: AUROC=0.800, gap=0.001

#### `last_token_hidden`

- tasks: `4`
- mean cross-task AUROC: `0.861`
- strong edge count: `10`
- edge density: `0.833`
- largest component: size `4`, edges `10`, categories `{"late-success": 3, "stable-success": 1}`

Top hubs:
- `4e049d9d8570` (stable-success): out=0.933, strong=3
- `43a1e0948c6a` (late-success): out=0.933, strong=3
- `b183db3494f4` (late-success): out=0.800, strong=2
- `a0f4b0be2055` (late-success): out=0.778, strong=2

Top receivers:
- `b183db3494f4` (late-success): in=1.000, strong=3
- `a0f4b0be2055` (late-success): in=1.000, strong=3
- `4e049d9d8570` (stable-success): in=0.800, strong=3
- `43a1e0948c6a` (late-success): in=0.644, strong=1

Top edges:
- `4e049d9d8570` -> `b183db3494f4`: AUROC=1.000, gap=0.032
- `4e049d9d8570` -> `a0f4b0be2055`: AUROC=1.000, gap=0.044
- `43a1e0948c6a` -> `b183db3494f4`: AUROC=1.000, gap=0.034
- `43a1e0948c6a` -> `a0f4b0be2055`: AUROC=1.000, gap=0.051
- `b183db3494f4` -> `a0f4b0be2055`: AUROC=1.000, gap=0.055

### `mistral7_no_outliers`

#### `mean_hidden`

- tasks: `1`
- mean cross-task AUROC: `0.500`
- strong edge count: `0`
- edge density: `0.000`
- largest component: size `1`, edges `0`, categories `{"late-success": 1}`

Top hubs:
- `caa1f15441c6` (late-success): out=0.500, strong=0

Top receivers:
- `caa1f15441c6` (late-success): in=0.500, strong=0

Top edges:

#### `last_token_hidden`

- tasks: `1`
- mean cross-task AUROC: `0.500`
- strong edge count: `0`
- edge density: `0.000`
- largest component: size `1`, edges `0`, categories `{"late-success": 1}`

Top hubs:
- `caa1f15441c6` (late-success): out=0.500, strong=0

Top receivers:
- `caa1f15441c6` (late-success): in=0.500, strong=0

Top edges:

### `qwen25_no_outliers`

#### `mean_hidden`

- tasks: `5`
- mean cross-task AUROC: `0.322`
- strong edge count: `4`
- edge density: `0.200`
- largest component: size `4`, edges `4`, categories `{"late-success": 3, "stable-success": 1}`

Top hubs:
- `4e049d9d8570` (late-success): out=0.472, strong=2
- `43a1e0948c6a` (late-success): out=0.458, strong=1
- `7eb3d5b5211b` (stable-success): out=0.333, strong=1
- `0558eb2672fb` (late-success): out=0.181, strong=0

Top receivers:
- `0558eb2672fb` (late-success): in=0.750, strong=3
- `7eb3d5b5211b` (stable-success): in=0.375, strong=0
- `43a1e0948c6a` (late-success): in=0.361, strong=1
- `4e049d9d8570` (late-success): in=0.125, strong=0

Top edges:
- `4e049d9d8570` -> `0558eb2672fb`: AUROC=1.000, gap=0.005
- `43a1e0948c6a` -> `0558eb2672fb`: AUROC=1.000, gap=0.005
- `7eb3d5b5211b` -> `0558eb2672fb`: AUROC=1.000, gap=0.003
- `4e049d9d8570` -> `43a1e0948c6a`: AUROC=0.889, gap=0.002

#### `last_token_hidden`

- tasks: `5`
- mean cross-task AUROC: `0.452`
- strong edge count: `7`
- edge density: `0.350`
- largest component: size `4`, edges `7`, categories `{"late-success": 3, "stable-success": 1}`

Top hubs:
- `4e049d9d8570` (late-success): out=0.583, strong=2
- `0558eb2672fb` (late-success): out=0.521, strong=2
- `43a1e0948c6a` (late-success): out=0.521, strong=2
- `7eb3d5b5211b` (stable-success): out=0.412, strong=1

Top receivers:
- `43a1e0948c6a` (late-success): in=0.806, strong=3
- `0558eb2672fb` (late-success): in=0.600, strong=2
- `4e049d9d8570` (late-success): in=0.438, strong=2
- `7eb3d5b5211b` (stable-success): in=0.417, strong=0

Top edges:
- `0558eb2672fb` -> `43a1e0948c6a`: AUROC=1.000, gap=0.046
- `4e049d9d8570` -> `0558eb2672fb`: AUROC=1.000, gap=0.067
- `4e049d9d8570` -> `43a1e0948c6a`: AUROC=1.000, gap=0.040
- `43a1e0948c6a` -> `0558eb2672fb`: AUROC=1.000, gap=0.051
- `7eb3d5b5211b` -> `43a1e0948c6a`: AUROC=1.000, gap=0.029

### `qwen3_no_outliers`

#### `mean_hidden`

- tasks: `7`
- mean cross-task AUROC: `0.533`
- strong edge count: `18`
- edge density: `0.429`
- largest component: size `7`, edges `18`, categories `{"late-success": 6, "stable-success": 1}`

Top hubs:
- `0405cc3d80d2` (late-success): out=0.738, strong=4
- `4f4cccee79cb` (late-success): out=0.708, strong=4
- `7f1b41d1cdf3` (late-success): out=0.686, strong=3
- `94b238d5f10a` (stable-success): out=0.629, strong=3

Top receivers:
- `a0f4b0be2055` (late-success): in=1.000, strong=6
- `149f73699e33` (late-success): in=0.667, strong=4
- `94b238d5f10a` (stable-success): in=0.667, strong=3
- `7f1b41d1cdf3` (late-success): in=0.500, strong=3

Top edges:
- `7f1b41d1cdf3` -> `a0f4b0be2055`: AUROC=1.000, gap=0.014
- `7f1b41d1cdf3` -> `94b238d5f10a`: AUROC=1.000, gap=0.003
- `7f1b41d1cdf3` -> `149f73699e33`: AUROC=1.000, gap=0.019
- `4f4cccee79cb` -> `a0f4b0be2055`: AUROC=1.000, gap=0.016
- `4f4cccee79cb` -> `149f73699e33`: AUROC=1.000, gap=0.015

#### `last_token_hidden`

- tasks: `7`
- mean cross-task AUROC: `0.783`
- strong edge count: `28`
- edge density: `0.667`
- largest component: size `7`, edges `28`, categories `{"late-success": 6, "stable-success": 1}`

Top hubs:
- `94b238d5f10a` (stable-success): out=0.907, strong=6
- `4f4cccee79cb` (late-success): out=0.861, strong=4
- `a0f4b0be2055` (late-success): out=0.851, strong=5
- `149f73699e33` (late-success): out=0.813, strong=4

Top receivers:
- `4f4cccee79cb` (late-success): in=0.875, strong=5
- `7f1b41d1cdf3` (late-success): in=0.870, strong=5
- `caa1f15441c6` (late-success): in=0.861, strong=5
- `0405cc3d80d2` (late-success): in=0.833, strong=6

Top edges:
- `7f1b41d1cdf3` -> `4f4cccee79cb`: AUROC=1.000, gap=0.029
- `7f1b41d1cdf3` -> `149f73699e33`: AUROC=1.000, gap=0.055
- `4f4cccee79cb` -> `7f1b41d1cdf3`: AUROC=1.000, gap=0.024
- `4f4cccee79cb` -> `caa1f15441c6`: AUROC=1.000, gap=0.009
- `4f4cccee79cb` -> `0405cc3d80d2`: AUROC=1.000, gap=0.018

## Read

- Dense `last_token_hidden` graphs mean there is some cross-task shared correctness geometry.
- Sparse or fragmented `mean_hidden` graphs mean mean pooling is more task-specific or confounded by problem content.
- Hubs are candidate source tasks whose final correct states define broadly useful scorers.
- Receivers are target tasks whose correct prefixes align with many other tasks' correct states.
