# Late-Success vs Late-Failure (Ollama-only)

Source: server-synced `results/` filtered to `small_model.startswith("ollama/")`, 30-task LiveBench reasoning runs only.

## Headline

- `late-success` means the trace eventually lands on the correct final answer, but reusable correctness appears late or is not stable enough to count as `stable-success`.
- `late-failure` means the trace shows some locally correct prefix during re-entry, but the final full trace still ends wrong.
- The key contrast is not just “when correctness first appears,” but whether that correctness is preserved to the end.

## Summary Table

| Category | N | Share | Mean steps | Mean first-correct frac | Median first-correct frac | Mean local correct rate | Last-small-correct rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late-success | 32 | 0.133 | 3.875 | 0.628 | 0.500 | 0.562 | 0.781 |
| late-failure | 39 | 0.163 | 3.436 | 0.527 | 0.333 | 0.424 | 0.333 |

## What Separates Them

- `late-success` appears later on average than `late-failure` in terms of first correct prefix: `0.628` vs `0.527`.
- But `late-success` has better persistence once correctness appears: mean local correctness `0.562` vs `0.424`.
- Final-step local correctness is also much stronger for `late-success`: `25/32` vs `13/39`.
- There are `4` `late-success` cases where the full trace is correct even though no local re-entry prefix was correct; these are absorbed into `late-success` by the current taxonomy definition.

## Per-Model Breakdown

| Small model | Late-success | Late-failure | Persistent-failure | Stable-success |
| --- | ---: | ---: | ---: | ---: |
| ollama/gemma3:4b | 6 | 11 | 40 | 3 |
| ollama/gemma4:e4b | 7 | 13 | 13 | 27 |
| ollama/ministral-3:3b | 9 | 9 | 37 | 5 |
| ollama/qwen3.5:4b | 10 | 6 | 6 | 38 |

## Interpretation

- `late-success` suggests the trace can eventually accumulate enough usable information, even if that signal emerges late.
- `late-failure` is the more diagnostic failure mode: the trace already contains a usable local state, but the full continuation fails to preserve it.
- So for these two categories, the key variable is not simply step count; it is whether the model can retain and carry forward a locally correct state.
- This supports the framing that re-entry is a probe of trace stability, not just a search for a universal best prefix length.
