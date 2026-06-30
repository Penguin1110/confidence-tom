# Hidden-State Commitment Dynamics

## Question

- Ask when hidden states become committed, not just whether a final answer is correct.
- Use a leave-one-task-out success-vs-failure margin as a cross-task commitment signal.
- Also compute a task-internal final-state lock time as a trajectory convergence signal.

## Aggregate

| Pooling | Commit frac | Final-state lock frac | Commit matches full outcome | Early failure-lock rate | Full-wrong early failure-lock rate | Full-correct success-lock frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_hidden | 0.328 | 0.766 | 0.551 | 0.288 | 0.257 | 0.672 |
| last_token_hidden | 0.486 | 0.999 | 0.694 | 0.340 | 0.436 | 0.724 |

## Family Breakdown

### `gemma4_no_outliers`

#### `mean_hidden`

- tasks: `29`
- margin threshold: `-0.0030`
- final success/failure margin mean: `0.0056` / `-0.0060`
- mean commit frac: `0.471`
- mean final-state lock frac: `0.757`
- early failure-lock rate: `0.000`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 9 | 0.078 | 0.078 | 1.000 | 0.661 | 0.000 | -0.022 |
| late-success | 9 | 0.619 | 0.693 | 0.926 | 0.834 | 0.000 | -0.047 |
| late-failure | 4 | 0.551 | 0.551 | 1.000 | 0.932 | 0.000 | -0.045 |
| persistent-failure | 7 | 0.739 | 1.000 | 0.739 | 0.683 | 0.000 | -0.044 |

#### `last_token_hidden`

- tasks: `29`
- margin threshold: `0.0046`
- final success/failure margin mean: `0.0285` / `-0.0186`
- mean commit frac: `0.595`
- mean final-state lock frac: `1.000`
- early failure-lock rate: `0.345`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 9 | 0.701 | 0.808 | 0.893 | 1.000 | 0.111 | 0.049 |
| late-success | 9 | 0.785 | 0.913 | 0.872 | 1.000 | 0.111 | 0.038 |
| late-failure | 4 | 0.487 | 0.932 | 0.556 | 1.000 | 0.500 | 0.025 |
| persistent-failure | 7 | 0.274 | 1.000 | 0.274 | 1.000 | 0.857 | -0.023 |

Early failure-lock examples:
- `0405cc3d80d2` (persistent-failure): commit_frac=0.040, first_correct_frac=1.000, margin_delta=-0.021
- `4d307a56c0be` (persistent-failure): commit_frac=0.043, first_correct_frac=1.000, margin_delta=0.006
- `448411c3001b` (late-failure): commit_frac=0.111, first_correct_frac=0.111, margin_delta=0.017
- `7eb3d5b5211b` (late-failure): commit_frac=0.111, first_correct_frac=1.000, margin_delta=-0.020

### `ministral_no_outliers`

#### `mean_hidden`

- tasks: `29`
- margin threshold: `-0.0388`
- final success/failure margin mean: `-0.0501` / `-0.0170`
- mean commit frac: `0.292`
- mean final-state lock frac: `0.772`
- early failure-lock rate: `0.069`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 1 | 0.167 | 1.000 | 0.167 | 0.833 | 1.000 | -0.011 |
| late-success | 3 | 0.317 | 0.750 | 0.567 | 0.725 | 0.333 | -0.014 |
| late-failure | 3 | 0.126 | 0.126 | 1.000 | 0.723 | 0.000 | -0.007 |
| persistent-failure | 22 | 0.317 | 0.317 | 1.000 | 0.782 | 0.000 | -0.008 |

#### `last_token_hidden`

- tasks: `29`
- margin threshold: `-0.0119`
- final success/failure margin mean: `-0.0103` / `-0.0149`
- mean commit frac: `0.470`
- mean final-state lock frac: `0.994`
- early failure-lock rate: `0.241`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 1 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.012 |
| late-success | 3 | 0.917 | 0.917 | 1.000 | 1.000 | 0.000 | 0.011 |
| late-failure | 3 | 0.418 | 0.451 | 0.967 | 0.974 | 0.000 | 0.008 |
| persistent-failure | 22 | 0.414 | 0.698 | 0.716 | 0.995 | 0.318 | 0.007 |

Early failure-lock examples:
- `d60075facb5e` (persistent-failure): commit_frac=0.083, first_correct_frac=1.000, margin_delta=-0.012
- `ba951e861241` (persistent-failure): commit_frac=0.091, first_correct_frac=1.000, margin_delta=0.010
- `7f1b41d1cdf3` (persistent-failure): commit_frac=0.100, first_correct_frac=1.000, margin_delta=-0.018
- `ed015ab9a9b9` (persistent-failure): commit_frac=0.100, first_correct_frac=1.000, margin_delta=-0.018

### `mistral7_no_outliers`

#### `mean_hidden`

- tasks: `28`
- margin threshold: `0.0000`
- final success/failure margin mean: `0.0000` / `-0.2549`
- mean commit frac: `0.181`
- mean final-state lock frac: `0.715`
- early failure-lock rate: `0.929`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-failure | 7 | 0.163 | 0.767 | 0.397 | 0.670 | 0.714 | 0.025 |
| persistent-failure | 21 | 0.187 | 1.000 | 0.187 | 0.730 | 1.000 | -0.003 |

Early failure-lock examples:
- `4f4cccee79cb` (late-failure): commit_frac=0.067, first_correct_frac=0.133, margin_delta=0.026
- `ba951e861241` (persistent-failure): commit_frac=0.071, first_correct_frac=1.000, margin_delta=-0.000
- `ed015ab9a9b9` (persistent-failure): commit_frac=0.071, first_correct_frac=1.000, margin_delta=0.014
- `0405cc3d80d2` (persistent-failure): commit_frac=0.083, first_correct_frac=1.000, margin_delta=-0.020

#### `last_token_hidden`

- tasks: `28`
- margin threshold: `0.0000`
- final success/failure margin mean: `0.0000` / `-0.1111`
- mean commit frac: `0.263`
- mean final-state lock frac: `1.000`
- early failure-lock rate: `0.893`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-failure | 7 | 0.373 | 0.976 | 0.397 | 1.000 | 0.714 | 0.083 |
| persistent-failure | 21 | 0.226 | 0.996 | 0.231 | 1.000 | 0.952 | 0.026 |

Early failure-lock examples:
- `4f4cccee79cb` (late-failure): commit_frac=0.067, first_correct_frac=0.133, margin_delta=0.113
- `ba951e861241` (persistent-failure): commit_frac=0.071, first_correct_frac=1.000, margin_delta=-0.017
- `ed015ab9a9b9` (persistent-failure): commit_frac=0.071, first_correct_frac=1.000, margin_delta=-0.006
- `0405cc3d80d2` (persistent-failure): commit_frac=0.083, first_correct_frac=1.000, margin_delta=0.078

### `qwen25_no_outliers`

#### `mean_hidden`

- tasks: `27`
- margin threshold: `-0.0076`
- final success/failure margin mean: `-0.0086` / `-0.0007`
- mean commit frac: `0.276`
- mean final-state lock frac: `0.777`
- early failure-lock rate: `0.370`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 5 | 0.350 | 0.650 | 0.700 | 0.720 | 0.400 | 0.002 |
| late-success | 4 | 0.182 | 1.000 | 0.182 | 0.741 | 1.000 | 0.001 |
| late-failure | 7 | 0.193 | 0.193 | 1.000 | 0.775 | 0.000 | 0.000 |
| persistent-failure | 11 | 0.330 | 0.621 | 0.708 | 0.818 | 0.364 | 0.001 |

Early failure-lock examples:
- `ed015ab9a9b9` (persistent-failure): commit_frac=0.125, first_correct_frac=1.000, margin_delta=-0.001
- `0405cc3d80d2` (persistent-failure): commit_frac=0.250, first_correct_frac=1.000, margin_delta=0.003
- `c29eb6b3c9fd` (persistent-failure): commit_frac=0.250, first_correct_frac=1.000, margin_delta=-0.000
- `3de6bc30b87b` (persistent-failure): commit_frac=0.333, first_correct_frac=1.000, margin_delta=0.002

#### `last_token_hidden`

- tasks: `27`
- margin threshold: `0.0003`
- final success/failure margin mean: `0.0039` / `-0.0063`
- mean commit frac: `0.572`
- mean final-state lock frac: `1.000`
- early failure-lock rate: `0.222`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 5 | 0.450 | 0.600 | 0.850 | 1.000 | 0.200 | 0.009 |
| late-success | 4 | 0.625 | 0.750 | 0.875 | 1.000 | 0.000 | -0.001 |
| late-failure | 7 | 0.748 | 0.748 | 1.000 | 1.000 | 0.000 | 0.004 |
| persistent-failure | 11 | 0.496 | 0.939 | 0.557 | 1.000 | 0.455 | -0.011 |

Early failure-lock examples:
- `ed015ab9a9b9` (persistent-failure): commit_frac=0.125, first_correct_frac=1.000, margin_delta=-0.028
- `56ecba920668` (persistent-failure): commit_frac=0.167, first_correct_frac=1.000, margin_delta=-0.028
- `7f1b41d1cdf3` (persistent-failure): commit_frac=0.167, first_correct_frac=1.000, margin_delta=-0.027
- `0405cc3d80d2` (persistent-failure): commit_frac=0.250, first_correct_frac=1.000, margin_delta=-0.029

### `qwen3_no_outliers`

#### `mean_hidden`

- tasks: `27`
- margin threshold: `0.0125`
- final success/failure margin mean: `0.0208` / `-0.0239`
- mean commit frac: `0.422`
- mean final-state lock frac: `0.807`
- early failure-lock rate: `0.074`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 6 | 0.260 | 0.260 | 1.000 | 0.604 | 0.000 | -0.009 |
| late-success | 6 | 0.407 | 0.453 | 0.954 | 0.862 | 0.000 | -0.015 |
| late-failure | 5 | 0.357 | 0.494 | 0.863 | 0.861 | 0.000 | -0.054 |
| persistent-failure | 10 | 0.560 | 0.752 | 0.808 | 0.868 | 0.200 | -0.066 |

Early failure-lock examples:
- `d60075facb5e` (persistent-failure): commit_frac=0.250, first_correct_frac=1.000, margin_delta=-0.114
- `51f6e96cffc7` (persistent-failure): commit_frac=0.333, first_correct_frac=1.000, margin_delta=-0.115

#### `last_token_hidden`

- tasks: `27`
- margin threshold: `0.0016`
- final success/failure margin mean: `0.0173` / `-0.0145`
- mean commit frac: `0.531`
- mean final-state lock frac: `1.000`
- early failure-lock rate: `0.000`

| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 6 | 0.135 | 0.135 | 1.000 | 1.000 | 0.000 | -0.006 |
| late-success | 6 | 0.421 | 0.421 | 1.000 | 1.000 | 0.000 | -0.011 |
| late-failure | 5 | 0.494 | 0.494 | 1.000 | 1.000 | 0.000 | -0.027 |
| persistent-failure | 10 | 0.852 | 0.910 | 0.942 | 1.000 | 0.000 | -0.043 |

## Read

- `commit_frac` is the first point where the cross-task margin stays on one side of the success/failure threshold.
- `final-state lock frac` is the first point where the trajectory has reached 90% of its own final-state similarity progress and never drops below it.
- `early failure-lock` is the proxy for premature convergence to a wrong state: the trajectory locks onto the failure side within the first third.
- This is still observational. A causal version would intervene before vs after the lock point and test reversibility.
