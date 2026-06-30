# Hidden-State Progress Report

## Scope

- Source run: `outputs/results/imported/results-livebench-reentry-a07127a`
- Focus: LiveBench re-entry, `*_no_outliers`
- Main question:
  - Why can a full reasoning trace be correct while prefix re-entry from the same trace often fails?
  - What do hidden states reveal about this mismatch?

## Current Framing

The central phenomenon is no longer "find the best re-entry step."

The better framing is:

> `full-trace correctness`, `prefix-level reusability`, and `state persistence` are related but not equivalent.

In particular:

- Some tasks are `full correct` but do **not** expose an early reusable prefix.
- Some tasks expose a reusable correct prefix but the original full trace still fails to preserve it.

## Behavioral Summary

Task-level taxonomy on the `141` LiveBench tasks:

| Category | Count | Share | Mean first-correct frac | Mean local correct rate | Last-small-correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| stable-success | 21 | 0.149 | 0.153 | 0.884 | 1.000 |
| late-success | 23 | 0.163 | 0.710 | 0.326 | 0.957 |
| late-failure | 26 | 0.184 | 0.467 | 0.332 | 0.269 |
| persistent-failure | 71 | 0.504 | 1.000 | 0.000 | 0.000 |

Important anomaly:

- Full-correct tasks: `44`
- Stable-success among them: `21`
- Late-success among them: `23`

So `52.3%` of full-correct tasks are **not** early-stable-reusable.

## What Hidden States Currently Show

### 1. Local re-entry correctness is more legible than full-trace success

From the hidden-state probe analysis:

| Pooling | Target | Weighted AUROC |
| --- | --- | ---: |
| mean_hidden | full_trace_correct | 0.370 |
| mean_hidden | reentry_exact_correct | 0.524 |
| last_token_hidden | full_trace_correct | 0.463 |
| last_token_hidden | reentry_exact_correct | 0.757 |

Interpretation:

- Hidden state is better at reading whether a prefix is in a **locally usable continuation state**
- It is much worse at reading whether the **entire full trace** will succeed

This supports the idea that full-trace success depends on rollout dynamics, not just a static prefix state snapshot.

### 2. `late-success` vs `late-failure` is mostly about persistence, not earlier emergence

Aggregate behavioral contrast:

- `late-success first-correct frac = 0.710`
- `late-failure first-correct frac = 0.467`
- `late-success last-small-correct = 0.957`
- `late-failure last-small-correct = 0.269`

Interpretation:

- `late-success` often becomes reusable **later**
- But once it gets there, it is much more likely to survive to the end

So the difference is not "who gets there first."
It is closer to "whose correct state survives."

### 3. `late-failure` is not a single mode

We added a persistence-oriented subtype analysis:

- `delayed-lock-in`: `17` tasks
- `oscillatory-success`: `5` tasks
- `fragile-flash`: `8` tasks
- `rescue-then-collapse`: `11` tasks
- `terminal-near-miss`: `7` tasks

This sharpened the main taxonomy:

- `late-success` is mostly dominated by `delayed-lock-in`
- `late-failure` splits into at least two qualitatively different modes:
  - `fragile-flash`: only a brief local correct flash
  - `rescue-then-collapse`: a more meaningful correct state appears, but is later lost

Category-level persistence summary:

| Category | First-correct frac | Tail correct rate | Collapse count | Longest correct streak |
| --- | ---: | ---: | ---: | ---: |
| stable-success | 0.153 | 0.900 | 0.905 | 7.619 |
| late-success | 0.710 | 0.878 | 0.522 | 2.522 |
| late-failure | 0.467 | 0.545 | 1.000 | 1.923 |
| persistent-failure | 1.000 | 0.000 | 0.000 | 0.000 |

Interpretation:

- `late-success` really is later on average
- But once it turns correct, the post-onset survival is much stronger than `late-failure`
- `late-failure` is better understood as a preservation problem than as a pure formation problem

### 4. Geometry supports the state-persistence story

Aggregate `late-success - late-failure` gaps:

#### `mean_hidden`

- cosine-to-final: `+0.021`
- mean displacement: `-1.205`
- max displacement: `-7.258`

Interpretation:

- `late-success` trajectories tend to stay closer to their own final successful state
- They also drift less

Strong family example: `qwen3`

#### `qwen3` `mean_hidden`

- `late-success mean cosine-to-final = 0.988`
- `late-failure mean cosine-to-final = 0.902`
- `late-success mean displacement = 5.652`
- `late-failure mean displacement = 8.538`

This is consistent with:

> `late-failure` is often not a total absence of a correct state.
> It looks more like a correct or near-correct state that later drifts away.

### 5. The onset state predicts collapse better than it predicts final success

We also isolated the hidden state at the **first correct prefix** for tasks that ever become locally correct.

Aggregate AUROC:

| Pooling | Predict full-trace correct | Predict last-small-correct | Predict collapse-after-onset |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.354 | 0.389 | 0.667 |
| last_token_hidden | 0.438 | 0.500 | 0.833 |

Interpretation:

- The hidden state at first-correct onset is **not** a good predictor of eventual full-trace success
- But it is much more informative about whether the trajectory will **collapse after onset**

This is an important refinement:

> hidden state at onset seems to encode `fragility` more clearly than `eventual success`.

Strong family example: `qwen3`

- `mean_hidden`
  - predict full-trace correct: `0.750`
  - predict last-small-correct: `0.667`
  - predict collapse-after-onset: `1.000`
- `last_token_hidden`
  - predict full-trace correct: `0.750`
  - predict last-small-correct: `1.000`
  - predict collapse-after-onset: `1.000`

This is small-sample and should be treated cautiously, but directionally it is very consistent with the state-persistence story.

### 6. Manifold analysis supports "success-likeness", but not immediate reusability

Aggregate `late-success - late-failure` gaps:

#### `mean_hidden`

- `sim(success) = +0.065`
- `sim(failure) = -0.034`
- `success-failure margin = +0.099`

Interpretation:

- `late-success` prefixes are more success-like than `late-failure` prefixes
- But being more success-like is still not the same as being early and stably reusable

This is why the main question has shifted from "which step is sufficient?" to:

> Why does success-like internal state not automatically become reusable prefix state?

### 7. Cross-task transfer shows a shared correctness geometry, mostly in `last_token_hidden`

We tested whether a correct-state score from task A can predict the correct/incorrect prefix distribution of task B.

Aggregate cross-task transfer:

| Pooling | Same-task AUROC | Cross-task AUROC |
| --- | ---: | ---: |
| mean_hidden | 0.891 | 0.443 |
| last_token_hidden | 0.903 | 0.630 |

Interpretation:

- `mean_hidden` is mostly task-specific and does not transfer reliably
- `last_token_hidden` has a moderate shared correctness geometry across tasks

Graph analysis makes this clearer. We build an edge `A -> B` when A's final correct hidden state predicts B's correct-vs-incorrect prefixes with AUROC >= `0.70`.

Aggregate graph results:

| Pooling | Mean cross-task AUROC | Mean edge density | Mean largest component share |
| --- | ---: | ---: | ---: |
| mean_hidden | 0.490 | 0.373 | 0.921 |
| last_token_hidden | 0.720 | 0.627 | 0.950 |

Strong examples:

- `qwen3` `last_token_hidden`
  - mean cross-task AUROC: `0.783`
  - edge density: `0.667`
  - largest component covers all `7` usable correct tasks
- `gemma4` `last_token_hidden`
  - mean cross-task AUROC: `0.785`
  - edge density: `0.658`
  - largest component covers all `17` usable correct tasks

This suggests:

> There is a partially shared cross-task correctness axis, but it is much more visible in the last-token state than in mean-pooled hidden states.

Important caveat:

- This graph only uses full-correct tasks that contain both correct and incorrect prefixes, so the usable task count is small in some families.

### 8. Commitment dynamics: moving from "will it be correct?" to "when does it lock?"

We added a first-pass commitment analysis around the proposed direction:

> not just whether a trace will be correct, but when the hidden state commits, whether it commits to the success side, and whether wrong traces show early failure-lock.

Operationalization:

- Build a leave-one-task-out success-vs-failure margin from final hidden states.
- A trajectory is considered committed when its margin crosses the family threshold and stays on that side.
- `early failure-lock` means the trajectory locks onto the failure side within the first third.
- Also compute a task-internal final-state lock time based on when the trajectory reaches 90% of its own final-state similarity progress.

Aggregate:

| Pooling | Commit frac | Final-state lock frac | Commit matches full outcome | Early failure-lock rate | Full-wrong early failure-lock rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_hidden | 0.328 | 0.766 | 0.551 | 0.288 | 0.257 |
| last_token_hidden | 0.486 | 0.999 | 0.694 | 0.340 | 0.436 |

Interpretation:

- `last_token_hidden` gives a cleaner cross-task commitment signal than `mean_hidden`.
- Full-wrong traces often show early failure-lock: aggregate `0.436` for `last_token_hidden`.
- This is a first observational proxy for premature convergence to a wrong state.

Strong examples:

- `gemma4` `last_token_hidden`
  - persistent-failure early failure-lock rate: `0.857`
  - late-failure early failure-lock rate: `0.500`
- `mistral7` `last_token_hidden`
  - persistent-failure early failure-lock rate: `0.952`
  - late-failure early failure-lock rate: `0.714`
- `qwen3` `last_token_hidden`
  - stable-success success-lock frac: `0.135`
  - late-success success-lock frac: `0.421`
  - persistent-failure failure-lock is not early under this proxy, which suggests family-specific dynamics

This gives a concrete next claim to test causally:

> Some incorrect trajectories enter a stable failure-side hidden-state basin early, before the generated reasoning has finished.
> The next experiment should intervene before vs after this lock point and measure reversibility.

Critical caveat:

The current `early failure-lock` result is still vulnerable to label-signal circularity. The behavioral labels (`persistent-failure`, `late-failure`, `stable-success`) are defined from prefix correctness and final correctness, while the hidden-state lock is also computed along the same trajectory. A reviewer could argue that high early failure-lock in `persistent-failure` is just another readout of "this trace stays wrong," not a new dynamics claim.

The key go/no-go test is precedence:

> Does hidden-state failure-lock happen before the model verbalizes or stabilizes the wrong intermediate answer?

If hidden-state lock systematically leads verbal answer commit, the result becomes evidence for look-ahead dynamics. If it happens at the same time or later, then hidden-state lock is mostly a redundant readout of already-written reasoning.

Current data status:

- We have original full-trace segments and prefix text.
- We have per-prefix re-entry answers (`reentry_exact_answer_key`) and correctness labels.
- We do **not** currently have a clean per-prefix annotation of the original trace's intermediate answer state.

Immediate workaround:

- Heuristically extract candidate answers from each original prefix.
- Define `verbal_commit_time` as the first prefix where the extracted candidate equals the final model answer and remains stable.
- Compare hidden `commit_time` vs verbal `commit_time`.

Stronger version:

- Run a lightweight answer-extraction/evaluation pass over every original prefix, not the re-entry continuation.
- Store per-prefix `original_prefix_answer_key` and `original_prefix_answer_matches_final`.
- Then compute lead/lag distributions directly.

This precedence analysis should be treated as the next required step before making a strong "point-of-no-return" claim.

## Prompt Perturbation vs State Reset

We also checked whether the phenomenon is likely to be just prompt-surface noise.

Control results across all imported re-entry rows:

- `reentry_exact_correct = 0.288`
- `reentry_marker_correct = 0.275`
- `reentry_fenced_correct = 0.280`
- `reentry_repeat_correct = 0.288`
- `reentry_repeat_matches_first = 1.000`

Interpretation:

- Different re-entry prompt surfaces (`exact`, `marker`, `fenced`) produce very similar correctness
- Repeating the same exact re-entry prompt reproduces the same answer essentially perfectly

So the current evidence does **not** point to simple prompt wording perturbation as the main cause.

The stronger hypothesis is:

> Re-entry behaves like a lossy serialization of state.
> It preserves the text prefix, but not the original rollout's latent reasoning state.

## Current Best Hypothesis

The best working explanation is:

1. A full correct trace may rely on trajectory-dependent latent state that is not yet fully encoded in the visible prefix text.
2. Re-entry from text-only prefix resets that latent state.
3. Some prefixes are enough to reconstruct a usable local state.
4. Many are not.
5. Even when a usable state appears, some trajectories fail to preserve it to the end.

In short:

> The bottleneck seems closer to `state formation + state persistence under reset`, not just missing information or prompt wording.

## What Is Most Solid Right Now

- `last_token_hidden` is strongly predictive of local re-entry correctness
- `late-success` vs `late-failure` is better explained by persistence than by earlier first-correct
- `late-failure` splits into at least a fragile-flash mode and a rescue-then-collapse mode
- onset hidden state seems more predictive of later collapse than of final success
- cross-task transfer is possible, especially with `last_token_hidden`, suggesting a partly shared correctness geometry
- commitment dynamics shows a measurable early failure-lock signal in full-wrong traces
- Geometry and manifolds both support the persistence / drift interpretation
- Prompt-surface controls do not currently support a "just prompt perturbation" explanation

## What Is Still Not Proven

- We do **not** yet have causal evidence that hidden-state reset is the mechanism
- We only have one selected layer per trace, not full layer-wise trajectories
- Family sample sizes are still small

## Recommended Next Steps

1. Extract multi-layer hidden states and run layer-wise trajectory analysis.
2. Run stronger prompt-matching controls that minimize full-vs-reentry instruction differences.
3. Do activation patching or cache-reuse continuation on `late-success` / `late-failure` cases, especially `qwen3`.
4. Refine taxonomy inside `late-success` into:
   - late reusable
   - unstable reusable
   - final-correct without clearly reusable prefix

## Go/No-Go Pilot: Reset Failure vs Ordinary Failure

We ran a one-day pilot comparing two failure modes:

- `reset-failure`: full trace is correct, but this prefix re-entry is wrong
- `ordinary-failure`: full trace is wrong, and this prefix re-entry is wrong

Both groups use only rows with `reentry_exact_correct = 0`, so local correctness itself is held fixed.

Main repeated task-split result:

| Feature | Repeated mean AUROC | Orientation-free separability |
| --- | ---: | ---: |
| mean_hidden | 0.601 | 0.804 |
| last_token_hidden | 0.571 | 0.764 |
| step/token controls | 0.647 | 0.727 |

Interpretation:

- There is some separability, especially orientation-free, but it is not cleanly hidden-state-specific.
- Step/token controls are also strong, so prefix length or position may be carrying part of the signal.
- `qwen3` is the strongest positive case:
  - `last_token_hidden` repeated AUROC: `0.799`
  - control repeated AUROC: `0.706`
- `gemma4` also separates, but control is close:
  - `mean_hidden` repeated AUROC: `0.766`
  - control repeated AUROC: `0.693`

Current go/no-go read:

> This is not a universal green light yet.
> The strongest seed is `qwen3 last_token_hidden`, but the next pilot must be length/step matched before we spend months on this direction.

Related file:

- [Hidden-State Failure Mode Pilot](./mainline/generated/analysis/trace/hidden_state_failure_mode_pilot.md)

## Related Files

- [LiveBench Re-entry Main Report](./mainline/generated/analysis/trace/livebench_reentry_multiview_report_zh.md)
- [Hidden State Probe](./mainline/generated/analysis/trace/hidden_state_livebench_reentry.md)
- [Late Success vs Late Failure](./mainline/generated/analysis/trace/hidden_state_late_success_vs_failure.md)
- [Geometry](./mainline/generated/analysis/trace/hidden_state_geometry.md)
- [Manifolds](./mainline/generated/analysis/trace/hidden_state_manifolds.md)
- [Trace Persistence Subtypes](./mainline/generated/analysis/trace/trace_persistence_subtypes.md)
- [Hidden State At First-Correct Onset](./mainline/generated/analysis/trace/hidden_state_onset_persistence.md)
- [Cross-Task Hidden-State Transfer](./mainline/generated/analysis/trace/cross_task_hidden_state_transfer.md)
- [Cross-Task Transfer Graph](./mainline/generated/analysis/trace/cross_task_transfer_graph.md)
- [Hidden-State Commitment Dynamics](./mainline/generated/analysis/trace/hidden_state_commitment_dynamics.md)
