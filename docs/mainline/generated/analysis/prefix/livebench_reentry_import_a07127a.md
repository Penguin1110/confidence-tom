# Imported LiveBench Re-entry Results: results-livebench-reentry-a07127a

This report summarizes the re-entry results copied from `origin/results-livebench-reentry-a07127a`.

## Re-entry Summary

| run | rows | full_rerun_match_rate | reentry_repeat_match_rate | marker_boundary_match_rate | fenced_boundary_match_rate |
|---|---:|---:|---:|---:|---:|
| _reentry_livebench_local_v1_gemma4_no_outliers | 401 | 0.07231920199501247 | 1.0 | 0.3915211970074813 | 0.3915211970074813 |
| _reentry_livebench_local_v1_ministral_no_outliers | 190 | 0.24736842105263157 | 1.0 | 0.5263157894736842 | 0.47368421052631576 |
| _reentry_livebench_local_v1_mistral7_no_outliers | 202 | 0.0 | 1.0 | 0.12376237623762376 | 0.12871287128712872 |
| _reentry_livebench_local_v1_qwen25_no_outliers | 138 | 0.2246376811594203 | 1.0 | 0.47101449275362317 | 0.427536231884058 |
| _reentry_livebench_local_v1_qwen3 | 1 | 0.0 | 1.0 | 0.0 | 0.0 |
| _reentry_livebench_local_v1_qwen3_no_outliers | 236 | 0.0 | 1.0 | 0.3516949152542373 | 0.3389830508474576 |

## Row-level Sanity Check

| run | rows | full_rerun_matches_original_full | reentry_repeat_matches_first | reentry_marker_matches_exact | reentry_fenced_matches_exact |
|---|---:|---:|---:|---:|---:|
| _reentry_livebench_local_v1_gemma4_no_outliers | 401 | 29 | 401 | 157 | 157 |
| _reentry_livebench_local_v1_ministral_no_outliers | 190 | 47 | 190 | 100 | 90 |
| _reentry_livebench_local_v1_mistral7_no_outliers | 202 | 0 | 202 | 25 | 26 |
| _reentry_livebench_local_v1_qwen25_no_outliers | 138 | 31 | 138 | 65 | 59 |
| _reentry_livebench_local_v1_qwen3 | 1 | 0 | 1 | 0 | 0 |
| _reentry_livebench_local_v1_qwen3_no_outliers | 236 | 0 | 236 | 83 | 80 |

## Prepare Summary

| prepare run | tasks | full_trace_correct_count | full_trace_correct_rate |
|---|---:|---:|---:|
| reentry_livebench_gemma4_30 | 30 | 18 | 0.6 |
| reentry_livebench_ministral_30 | 30 | 4 | 0.13333333333333333 |
| reentry_livebench_mistral7_30 | 30 | 1 | 0.03333333333333333 |
| reentry_livebench_qwen25_30 | 30 | 9 | 0.3 |
| reentry_livebench_qwen3_1 | 1 | 1 | 1.0 |
| reentry_livebench_qwen3_30 | 30 | 13 | 0.43333333333333335 |

## Notes

- `original_small_answer_key` is empty for every imported run in this branch, so exact-match-to-original-small is not a meaningful metric here.
- `reentry_repeat_matches_first` is 100% for every imported run in this branch.
- This suggests the rerun path is internally stable once rerun.
- `qwen3` appears in two states: a 1-row partial run and a larger `qwen3_no_outliers` run with 236 rows.
- The strongest prepare-time full-trace rate in this imported bundle is `gemma4` at 0.6; the weakest is `mistral7` at 0.0333.
