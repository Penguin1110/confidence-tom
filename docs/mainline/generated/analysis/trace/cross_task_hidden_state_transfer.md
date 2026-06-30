# Cross-Task Hidden-State Transfer

## Question

- Can hidden states be compared across different tasks at all?
- If task A is correct, can a score derived from task A's correct-state geometry predict the correct/incorrect prefix distribution of task B?

## Aggregate

| Pooling | Mean same-task AUROC | Mean cross-task AUROC | Mean same-vs-cross gap | Cross-task share AUROC > 0.6 | Cross-task share AUROC > 0.7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_hidden | 0.891 | 0.443 | 0.348 | 0.300 | 0.241 |
| last_token_hidden | 0.903 | 0.630 | 0.173 | 0.514 | 0.433 |

## Family Breakdown

### `gemma4_no_outliers`

#### `mean_hidden`

- correct tasks usable for transfer: `17`
- same-task mean AUROC: `0.737`
- cross-task mean AUROC: `0.589`
- same-vs-cross gap: `0.149`
- cross-task share AUROC > 0.7: `0.360`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.677 | 0.445 | 0.025 |
| correct_centroid | 0.501 | 0.276 | 0.005 |

#### `last_token_hidden`

- correct tasks usable for transfer: `17`
- same-task mean AUROC: `0.779`
- cross-task mean AUROC: `0.690`
- same-vs-cross gap: `0.088`
- cross-task share AUROC > 0.7: `0.518`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.785 | 0.658 | 0.094 |
| correct_centroid | 0.596 | 0.379 | 0.036 |


### `ministral_no_outliers`

#### `mean_hidden`

- correct tasks usable for transfer: `4`
- same-task mean AUROC: `0.900`
- cross-task mean AUROC: `0.331`
- same-vs-cross gap: `0.569`
- cross-task share AUROC > 0.7: `0.292`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.428 | 0.417 | -0.001 |
| correct_centroid | 0.233 | 0.167 | -0.006 |

#### `last_token_hidden`

- correct tasks usable for transfer: `4`
- same-task mean AUROC: `0.933`
- cross-task mean AUROC: `0.811`
- same-vs-cross gap: `0.122`
- cross-task share AUROC > 0.7: `0.750`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.861 | 0.833 | 0.029 |
| correct_centroid | 0.761 | 0.667 | 0.020 |


### `mistral7_no_outliers`

#### `mean_hidden`

- correct tasks usable for transfer: `1`
- same-task mean AUROC: `1.000`
- cross-task mean AUROC: `0.500`
- same-vs-cross gap: `0.000`
- cross-task share AUROC > 0.7: `0.000`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.500 | 0.000 | 0.000 |
| correct_centroid | 0.500 | 0.000 | 0.000 |

#### `last_token_hidden`

- correct tasks usable for transfer: `1`
- same-task mean AUROC: `1.000`
- cross-task mean AUROC: `0.500`
- same-vs-cross gap: `0.000`
- cross-task share AUROC > 0.7: `0.000`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.500 | 0.000 | 0.000 |
| correct_centroid | 0.500 | 0.000 | 0.000 |


### `qwen25_no_outliers`

#### `mean_hidden`

- correct tasks usable for transfer: `5`
- same-task mean AUROC: `0.867`
- cross-task mean AUROC: `0.282`
- same-vs-cross gap: `0.585`
- cross-task share AUROC > 0.7: `0.150`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.322 | 0.200 | -0.005 |
| correct_centroid | 0.242 | 0.100 | -0.005 |

#### `last_token_hidden`

- correct tasks usable for transfer: `5`
- same-task mean AUROC: `0.850`
- cross-task mean AUROC: `0.410`
- same-vs-cross gap: `0.440`
- cross-task share AUROC > 0.7: `0.300`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.452 | 0.350 | -0.003 |
| correct_centroid | 0.367 | 0.250 | -0.012 |


### `qwen3_no_outliers`

#### `mean_hidden`

- correct tasks usable for transfer: `7`
- same-task mean AUROC: `0.952`
- cross-task mean AUROC: `0.515`
- same-vs-cross gap: `0.438`
- cross-task share AUROC > 0.7: `0.405`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.533 | 0.429 | 0.002 |
| correct_centroid | 0.496 | 0.381 | 0.001 |

#### `last_token_hidden`

- correct tasks usable for transfer: `7`
- same-task mean AUROC: `0.952`
- cross-task mean AUROC: `0.739`
- same-vs-cross gap: `0.213`
- cross-task share AUROC > 0.7: `0.595`

| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |
| --- | ---: | ---: | ---: |
| final_correct_vector | 0.783 | 0.667 | 0.015 |
| correct_centroid | 0.696 | 0.524 | 0.011 |


## Read

- If cross-task AUROC is much above 0.5, then task-level correct-state geometry is at least partially shared across tasks.
- If same-task AUROC is much higher than cross-task AUROC, then hidden-state scores are only weakly transferable and remain task-specific.
