# Hidden State Manifolds

## Idea

- Build manifolds from final-step hidden states.
- Score every prefix by similarity to success/failure and category-specific final-state centroids.

## Aggregate Late-Success Minus Late-Failure

| Pooling | Delta sim(success) | Delta sim(failure) | Delta success-failure margin | Delta sim(stable-success) |
| --- | ---: | ---: | ---: | ---: |
| mean_hidden | 0.065 | -0.034 | 0.099 | -0.009 |
| last_token_hidden | 0.028 | 0.003 | 0.025 | 0.007 |

## Family Breakdown

### `gemma4_no_outliers`

- tasks: `29`
- selected layer: `42`
- hidden dim: `2560`

#### `mean_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 140 | 0.955 | 0.927 | 0.027 | 0.967 | 0.934 | 0.932 | 0.924 |
| late-success | 117 | 0.954 | 0.931 | 0.023 | 0.962 | 0.938 | 0.935 | 0.928 |
| late-failure | 39 | 0.942 | 0.912 | 0.030 | 0.956 | 0.919 | 0.917 | 0.908 |
| persistent-failure | 105 | 0.968 | 0.956 | 0.012 | 0.967 | 0.959 | 0.958 | 0.954 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.919 | 0.879 | 0.041 |
| late-failure:q2 | 0.930 | 0.891 | 0.038 |
| late-failure:q3 | 0.944 | 0.912 | 0.032 |
| late-failure:q4 | 0.966 | 0.951 | 0.015 |
| late-success:q1 | 0.926 | 0.887 | 0.040 |
| late-success:q2 | 0.945 | 0.912 | 0.034 |
| late-success:q3 | 0.966 | 0.950 | 0.016 |
| late-success:q4 | 0.977 | 0.969 | 0.007 |
| persistent-failure:q1 | 0.941 | 0.906 | 0.035 |
| persistent-failure:q2 | 0.966 | 0.950 | 0.016 |
| persistent-failure:q3 | 0.979 | 0.976 | 0.003 |
| persistent-failure:q4 | 0.981 | 0.983 | -0.002 |
| stable-success:q1 | 0.932 | 0.896 | 0.036 |
| stable-success:q2 | 0.948 | 0.917 | 0.031 |
| stable-success:q3 | 0.963 | 0.938 | 0.025 |
| stable-success:q4 | 0.973 | 0.954 | 0.019 |

#### `last_token_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 140 | 0.764 | 0.763 | 0.002 | 0.780 | 0.730 | 0.750 | 0.755 |
| late-success | 117 | 0.753 | 0.754 | -0.000 | 0.757 | 0.731 | 0.741 | 0.746 |
| late-failure | 39 | 0.699 | 0.710 | -0.011 | 0.709 | 0.671 | 0.693 | 0.705 |
| persistent-failure | 105 | 0.777 | 0.800 | -0.022 | 0.777 | 0.757 | 0.778 | 0.796 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.620 | 0.639 | -0.018 |
| late-failure:q2 | 0.648 | 0.662 | -0.014 |
| late-failure:q3 | 0.695 | 0.705 | -0.010 |
| late-failure:q4 | 0.795 | 0.800 | -0.005 |
| late-success:q1 | 0.658 | 0.670 | -0.012 |
| late-success:q2 | 0.707 | 0.714 | -0.007 |
| late-success:q3 | 0.789 | 0.787 | 0.002 |
| late-success:q4 | 0.846 | 0.832 | 0.014 |
| persistent-failure:q1 | 0.685 | 0.689 | -0.004 |
| persistent-failure:q2 | 0.774 | 0.791 | -0.017 |
| persistent-failure:q3 | 0.814 | 0.842 | -0.027 |
| persistent-failure:q4 | 0.820 | 0.857 | -0.037 |
| stable-success:q1 | 0.696 | 0.707 | -0.011 |
| stable-success:q2 | 0.747 | 0.748 | -0.001 |
| stable-success:q3 | 0.776 | 0.781 | -0.006 |
| stable-success:q4 | 0.826 | 0.805 | 0.021 |

### `ministral_no_outliers`

- tasks: `29`
- selected layer: `36`
- hidden dim: `4096`

#### `mean_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 6 | 0.925 | 0.911 | 0.014 | 0.993 | 0.867 | 0.889 | 0.909 |
| late-success | 17 | 0.902 | 0.872 | 0.029 | 0.798 | 0.906 | 0.849 | 0.870 |
| late-failure | 28 | 0.872 | 0.890 | -0.018 | 0.838 | 0.851 | 0.914 | 0.882 |
| persistent-failure | 139 | 0.897 | 0.919 | -0.022 | 0.859 | 0.877 | 0.885 | 0.918 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.880 | 0.893 | -0.014 |
| late-failure:q2 | 0.880 | 0.899 | -0.019 |
| late-failure:q3 | 0.870 | 0.891 | -0.021 |
| late-failure:q4 | 0.861 | 0.879 | -0.018 |
| late-success:q1 | 0.897 | 0.865 | 0.032 |
| late-success:q2 | 0.904 | 0.872 | 0.032 |
| late-success:q3 | 0.906 | 0.876 | 0.030 |
| late-success:q4 | 0.901 | 0.875 | 0.026 |
| persistent-failure:q1 | 0.905 | 0.919 | -0.015 |
| persistent-failure:q2 | 0.899 | 0.919 | -0.021 |
| persistent-failure:q3 | 0.899 | 0.923 | -0.024 |
| persistent-failure:q4 | 0.889 | 0.915 | -0.026 |
| stable-success:q1 | 0.922 | 0.906 | 0.016 |
| stable-success:q2 | 0.928 | 0.913 | 0.015 |
| stable-success:q3 | 0.927 | 0.913 | 0.014 |
| stable-success:q4 | 0.924 | 0.912 | 0.012 |

#### `last_token_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 6 | 0.944 | 0.938 | 0.006 | 0.950 | 0.927 | 0.934 | 0.935 |
| late-success | 17 | 0.933 | 0.919 | 0.014 | 0.890 | 0.932 | 0.914 | 0.916 |
| late-failure | 28 | 0.927 | 0.929 | -0.002 | 0.905 | 0.920 | 0.935 | 0.924 |
| persistent-failure | 139 | 0.895 | 0.918 | -0.022 | 0.872 | 0.889 | 0.894 | 0.917 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.913 | 0.920 | -0.007 |
| late-failure:q2 | 0.934 | 0.929 | 0.005 |
| late-failure:q3 | 0.929 | 0.928 | 0.001 |
| late-failure:q4 | 0.930 | 0.935 | -0.005 |
| late-success:q1 | 0.901 | 0.899 | 0.002 |
| late-success:q2 | 0.926 | 0.913 | 0.013 |
| late-success:q3 | 0.942 | 0.925 | 0.018 |
| late-success:q4 | 0.955 | 0.934 | 0.021 |
| persistent-failure:q1 | 0.880 | 0.908 | -0.028 |
| persistent-failure:q2 | 0.903 | 0.921 | -0.019 |
| persistent-failure:q3 | 0.902 | 0.922 | -0.020 |
| persistent-failure:q4 | 0.894 | 0.918 | -0.024 |
| stable-success:q1 | 0.915 | 0.922 | -0.007 |
| stable-success:q2 | 0.935 | 0.937 | -0.001 |
| stable-success:q3 | 0.954 | 0.939 | 0.015 |
| stable-success:q4 | 0.962 | 0.947 | 0.015 |

### `mistral7_no_outliers`

- tasks: `29`
- selected layer: `32`
- hidden dim: `4096`

#### `mean_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-success | 10 | 0.970 | 0.707 | 0.264 | 0.000 | 0.970 | 0.802 | 0.665 |
| late-failure | 50 | 0.696 | 0.872 | -0.176 | 0.000 | 0.696 | 0.895 | 0.851 |
| persistent-failure | 142 | 0.616 | 0.915 | -0.298 | 0.000 | 0.616 | 0.876 | 0.913 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.666 | 0.862 | -0.196 |
| late-failure:q2 | 0.703 | 0.867 | -0.164 |
| late-failure:q3 | 0.696 | 0.884 | -0.189 |
| late-failure:q4 | 0.708 | 0.873 | -0.165 |
| late-success:q1 | 0.917 | 0.712 | 0.205 |
| late-success:q2 | 0.964 | 0.706 | 0.258 |
| late-success:q3 | 0.991 | 0.706 | 0.285 |
| late-success:q4 | 0.999 | 0.705 | 0.295 |
| persistent-failure:q1 | 0.610 | 0.905 | -0.294 |
| persistent-failure:q2 | 0.619 | 0.916 | -0.297 |
| persistent-failure:q3 | 0.616 | 0.917 | -0.301 |
| persistent-failure:q4 | 0.618 | 0.918 | -0.299 |

#### `last_token_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-success | 10 | 0.830 | 0.860 | -0.031 | 0.000 | 0.830 | 0.871 | 0.846 |
| late-failure | 50 | 0.759 | 0.878 | -0.119 | 0.000 | 0.759 | 0.872 | 0.869 |
| persistent-failure | 142 | 0.717 | 0.878 | -0.161 | 0.000 | 0.717 | 0.836 | 0.881 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.704 | 0.850 | -0.146 |
| late-failure:q2 | 0.728 | 0.878 | -0.149 |
| late-failure:q3 | 0.765 | 0.890 | -0.125 |
| late-failure:q4 | 0.809 | 0.887 | -0.077 |
| late-success:q1 | 0.713 | 0.834 | -0.121 |
| late-success:q2 | 0.788 | 0.881 | -0.094 |
| late-success:q3 | 0.863 | 0.853 | 0.010 |
| late-success:q4 | 0.928 | 0.862 | 0.066 |
| persistent-failure:q1 | 0.682 | 0.848 | -0.165 |
| persistent-failure:q2 | 0.699 | 0.878 | -0.179 |
| persistent-failure:q3 | 0.723 | 0.889 | -0.166 |
| persistent-failure:q4 | 0.751 | 0.889 | -0.137 |

### `qwen25_no_outliers`

- tasks: `27`
- selected layer: `48`
- hidden dim: `5120`

#### `mean_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 22 | 0.936 | 0.924 | 0.012 | 0.939 | 0.917 | 0.915 | 0.921 |
| late-success | 23 | 0.951 | 0.956 | -0.004 | 0.932 | 0.960 | 0.951 | 0.950 |
| late-failure | 40 | 0.950 | 0.955 | -0.005 | 0.935 | 0.953 | 0.957 | 0.945 |
| persistent-failure | 53 | 0.955 | 0.962 | -0.007 | 0.941 | 0.958 | 0.941 | 0.967 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.942 | 0.947 | -0.005 |
| late-failure:q2 | 0.951 | 0.957 | -0.006 |
| late-failure:q3 | 0.952 | 0.957 | -0.005 |
| late-failure:q4 | 0.954 | 0.959 | -0.005 |
| late-success:q1 | 0.947 | 0.952 | -0.005 |
| late-success:q2 | 0.952 | 0.956 | -0.004 |
| late-success:q3 | 0.948 | 0.952 | -0.004 |
| late-success:q4 | 0.956 | 0.960 | -0.004 |
| persistent-failure:q1 | 0.945 | 0.952 | -0.007 |
| persistent-failure:q2 | 0.956 | 0.963 | -0.007 |
| persistent-failure:q3 | 0.957 | 0.964 | -0.007 |
| persistent-failure:q4 | 0.961 | 0.968 | -0.007 |
| stable-success:q1 | 0.927 | 0.918 | 0.009 |
| stable-success:q2 | 0.937 | 0.926 | 0.011 |
| stable-success:q3 | 0.939 | 0.927 | 0.012 |
| stable-success:q4 | 0.939 | 0.924 | 0.014 |

#### `last_token_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 22 | 0.932 | 0.924 | 0.008 | 0.930 | 0.919 | 0.928 | 0.912 |
| late-success | 23 | 0.937 | 0.935 | 0.003 | 0.925 | 0.938 | 0.938 | 0.923 |
| late-failure | 40 | 0.928 | 0.929 | -0.000 | 0.918 | 0.926 | 0.933 | 0.916 |
| persistent-failure | 53 | 0.921 | 0.930 | -0.009 | 0.909 | 0.922 | 0.919 | 0.928 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.897 | 0.898 | -0.001 |
| late-failure:q2 | 0.923 | 0.925 | -0.002 |
| late-failure:q3 | 0.940 | 0.937 | 0.002 |
| late-failure:q4 | 0.947 | 0.948 | -0.001 |
| late-success:q1 | 0.901 | 0.900 | 0.001 |
| late-success:q2 | 0.924 | 0.925 | -0.001 |
| late-success:q3 | 0.944 | 0.941 | 0.003 |
| late-success:q4 | 0.966 | 0.959 | 0.007 |
| persistent-failure:q1 | 0.900 | 0.905 | -0.005 |
| persistent-failure:q2 | 0.918 | 0.924 | -0.006 |
| persistent-failure:q3 | 0.932 | 0.939 | -0.006 |
| persistent-failure:q4 | 0.931 | 0.947 | -0.016 |
| stable-success:q1 | 0.896 | 0.893 | 0.003 |
| stable-success:q2 | 0.921 | 0.919 | 0.002 |
| stable-success:q3 | 0.940 | 0.933 | 0.007 |
| stable-success:q4 | 0.960 | 0.944 | 0.017 |

### `qwen3_no_outliers`

- tasks: `27`
- selected layer: `40`
- hidden dim: `5120`

#### `mean_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 66 | 0.973 | 0.947 | 0.026 | 0.989 | 0.926 | 0.954 | 0.940 |
| late-success | 51 | 0.964 | 0.933 | 0.032 | 0.915 | 0.979 | 0.940 | 0.925 |
| late-failure | 45 | 0.958 | 0.940 | 0.018 | 0.924 | 0.958 | 0.946 | 0.933 |
| persistent-failure | 74 | 0.963 | 0.951 | 0.012 | 0.929 | 0.964 | 0.951 | 0.948 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.964 | 0.932 | 0.031 |
| late-failure:q2 | 0.967 | 0.941 | 0.025 |
| late-failure:q3 | 0.969 | 0.951 | 0.018 |
| late-failure:q4 | 0.937 | 0.936 | 0.001 |
| late-success:q1 | 0.960 | 0.921 | 0.038 |
| late-success:q2 | 0.965 | 0.931 | 0.034 |
| late-success:q3 | 0.966 | 0.935 | 0.031 |
| late-success:q4 | 0.966 | 0.940 | 0.026 |
| persistent-failure:q1 | 0.969 | 0.939 | 0.030 |
| persistent-failure:q2 | 0.976 | 0.957 | 0.019 |
| persistent-failure:q3 | 0.977 | 0.961 | 0.016 |
| persistent-failure:q4 | 0.936 | 0.948 | -0.012 |
| stable-success:q1 | 0.972 | 0.945 | 0.027 |
| stable-success:q2 | 0.974 | 0.948 | 0.025 |
| stable-success:q3 | 0.973 | 0.948 | 0.026 |
| stable-success:q4 | 0.973 | 0.948 | 0.025 |

#### `last_token_hidden`

| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable-success | 66 | 0.963 | 0.923 | 0.041 | 0.967 | 0.947 | 0.934 | 0.913 |
| late-success | 51 | 0.953 | 0.931 | 0.022 | 0.939 | 0.957 | 0.936 | 0.925 |
| late-failure | 45 | 0.954 | 0.937 | 0.017 | 0.941 | 0.955 | 0.943 | 0.930 |
| persistent-failure | 74 | 0.921 | 0.912 | 0.009 | 0.907 | 0.925 | 0.908 | 0.910 |

| Category:StepBin | sim(success) | sim(failure) | margin |
| --- | ---: | ---: | ---: |
| late-failure:q1 | 0.946 | 0.926 | 0.020 |
| late-failure:q2 | 0.955 | 0.935 | 0.020 |
| late-failure:q3 | 0.957 | 0.936 | 0.021 |
| late-failure:q4 | 0.955 | 0.947 | 0.008 |
| late-success:q1 | 0.940 | 0.918 | 0.023 |
| late-success:q2 | 0.953 | 0.930 | 0.023 |
| late-success:q3 | 0.953 | 0.928 | 0.026 |
| late-success:q4 | 0.963 | 0.946 | 0.018 |
| persistent-failure:q1 | 0.937 | 0.915 | 0.023 |
| persistent-failure:q2 | 0.922 | 0.906 | 0.016 |
| persistent-failure:q3 | 0.909 | 0.897 | 0.012 |
| persistent-failure:q4 | 0.919 | 0.928 | -0.010 |
| stable-success:q1 | 0.956 | 0.918 | 0.039 |
| stable-success:q2 | 0.964 | 0.923 | 0.041 |
| stable-success:q3 | 0.964 | 0.920 | 0.044 |
| stable-success:q4 | 0.967 | 0.929 | 0.039 |
