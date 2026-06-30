# Logic Commitment Probe Spec

## Goal

This probe is the go/no-go control for the current `type C` story.

We are no longer asking only:

- can the model flag edited reasoning on `MATH`?

We are now asking the sharper question:

- is low `path_inconsistency` share a **weak phenomenon** problem, or a **bad substrate** problem?

The control must therefore use a benchmark where:

- commitment structure is explicit
- local-vs-global validity can be checked independently of the target model
- `type C` can be constructed without relying on a judge LLM's taste

## Phase 1 Choice

Start with `ProntoQA`.

Reasons:

- it is already reachable from Hugging Face in this repo's environment
- each example contains a short proof-like chain of thought
- claims, rules, and intermediate states are much more explicit than in `MATH`
- it is good enough for an existence-style control even if it is more synthetic than `ProofWriter`

`ProofWriter` stays the preferred phase-2 confirmation set if we want stronger formal guarantees later.

## Run Shape

Run only the minimum cells needed to cut the key confusion:

- benchmark: `ProntoQA`
- substrate pool: `20`
- edit types: `clean`, `C_inconsistent`
- elicitations: `S0_continue`, `S1_review`
- model: same target model as the current pilot unless we explicitly change it

Do not add:

- `A_error`
- `B_style`
- extra models
- scale sweeps

This control is about attribution purity, not headline flag rate.

## Task Representation

Each task should contain:

- the full rule/fact context
- the explicit prove query
- the original valid chain of thought

The current loader maps each task to:

- `question`: rendered as `Facts and rules` plus `Goal`
- `reference_answer`: the normalized prove target
- `metadata.chain_of_thought`: the original proof steps

## Type C Definition for Logic

`type C` should mean:

- the edited step is locally plausible or valid in some proof context
- the edited step is not a standalone factual or rule error
- the edited step becomes problematic only relative to the current proof state

In practice, prefer these conflict families:

1. Subgoal switch
- the inserted step proves a different intermediate claim than the current chain is pursuing

2. Branch switch
- the inserted step belongs to a different valid proof branch than the one established earlier

3. Scope switch
- the inserted step changes which entity, case, or assumption is currently active

4. Rule-commitment switch
- the inserted step uses a rule application that is valid elsewhere but not licensed by the current state

5. Context narrowing
- the inserted step silently assumes a stricter condition than the proof has established so far

Avoid:

- wrong facts
- wrong rule applications obvious from the sentence alone
- undefined references
- arithmetic-style local mistakes

## Construction Pipeline

Use the same high-level pattern as the current pilot:

1. Generate or recover an original valid proof trace
2. Generate an alternate valid proof trace for the same query
3. Extract a donor step from the alternate trace
4. Transplant the donor step into the current trace with notation aligned
5. Apply a logic-specific verifier before the sample is admitted

## Admission Checks

The verifier should enforce all of:

- `internally_consistent = true`
- `locally_plausible = true`
- `globally_inconsistent = true`
- `locally_checkable_error = false`
- `requires_cross_step_context = true`

For the logic control, these should not remain purely LLM-judged if we can avoid it.

## Logic-Specific Verifier

Minimum viable verifier:

- reconstruct a proof state from the prefix up to the edited step
- ask whether the edited step's claim is licensed by that proof state
- ask whether the edited step's claim is still a sensible proof step under some alternate valid route

Operational approximation for the first pass:

- extract atomic statements from the prefix
- track the active current goal
- mark a candidate as invalid for admission if:
  - it introduces a claim unrelated to the active goal path
  - it uses premises not established in the prefix
  - it depends on a different branch or entity binding
  - it is directly false from the explicit world facts

If we later add a more formal proof-state checker, it should replace this heuristic layer rather than change the task definition.

## Judge Setup

Keep the target model interaction simple:

- `S0_continue`: continue the proof and give the final conclusion
- `S1_review`: review the proof, then continue and give the final conclusion

For attribution after a `FLAG`, classify only:

- `error_catch`
- `path_inconsistency`

Do not optimize for total `FLAG` rate.

The main dependent variable is:

- `path_inconsistency_share = path_inconsistency / all_flagged_C`

Secondary metrics:

- `clean false_flag_rate`
- `C flag_rate`
- `C follow_rate`

## Go / No-Go Thresholds

Use these as pilot thresholds, not final publication thresholds.

Green:

- `clean false_flag_rate` remains low
- `path_inconsistency_share > 0.60` on logic `C`

Yellow:

- `0.40 <= path_inconsistency_share <= 0.60`

Red:

- `path_inconsistency_share < 0.40`

Interpretation:

- if logic `path_inconsistency_share` is high while `MATH` stays low, the phenomenon is likely real and `MATH` is the wrong primary substrate
- if logic `path_inconsistency_share` also stays low, the weakness is probably conceptual, not just dataset-specific

## Immediate Next Step

Implement the logic control before any new `MATH`-only generator work.

Priority order:

1. adapt the current pilot runner to accept `prontoqa`
2. use the stored proof trace from the task metadata as substrate material when available
3. add the logic-specific `type C` admission checks
4. run `clean + C` on `pool=20`, `S0 + S1`
5. attribute all flagged `C` responses and compare `path_inconsistency_share` against the current `MATH` baseline

## Phase 1 Update

The initial logic-control result changed the interpretation of the next step.

What we now know:

- automatic `ProntoQA` construction still collapses many `C` samples into local errors
- a hand-crafted floor test with clean agenda-switch samples shows that the model can
  detect cross-step inconsistency when explicitly asked
- the same clean samples, when presented under ordinary continuation rather than an
  explicit consistency-check instruction, still trigger spontaneous noticing in a
  substantial subset of cases

So the next question is no longer only:

- can the model detect cross-step inconsistency at all?

It is now also:

- which kinds of inconsistency does the model spontaneously notice, and which kinds
  does it systematically miss?

This moves the project from a pure existence probe toward a **blindspot map**.

## Blindspot-Map Objective

The paper-level object is not just a single flag rate.

It is a map over inconsistency recipes:

- recipe family
- explicit-check detection rate
- spontaneous-continuation detection rate
- typical failure mode when detection does not happen

At the moment, the most promising recipe families are:

1. `focus_shift`
- the proof abruptly pursues an irrelevant side-branch instead of the already active goal

2. `branch_follow`
- the proof follows a valid branch from an established fact, but not the branch the current proof was already committed to

3. `goal_deferral`
- the proof explicitly postpones a direct conclusion and opens another route first

The current evidence suggests:

- `focus_shift` and `branch_follow` trigger more spontaneous noticing
- `goal_deferral` is more likely to be tolerated or treated as harmless

That asymmetry is a candidate core result rather than a bug to remove.

## Phase 2a: Recipe Baseline

Before running a larger blindspot map, first establish a per-recipe ability
baseline under the strongest explicit consistency-check prompt.

This matters because a low spontaneous rate can mean two different things:

- the model has a genuine blindspot for that recipe
- the recipe construction is weak or dirty

Run:

- `n = 10` per recipe family
- explicit consistency-check prompt only
- report `PATH_INCONSISTENT`, `LOCAL_ERROR`, `CONSISTENT`, and `UNCLEAR` by recipe

Gate for moving to the blindspot-map expansion:

- each recipe should have high explicit-check `PATH_INCONSISTENT` rate
- each recipe should have low `LOCAL_ERROR` rate

If `goal_deferral` fails this baseline, do not interpret its spontaneous miss rate
as a model blindspot yet. First revise or retire that recipe.

### Phase 2a Result

The per-recipe explicit baseline was run with `n = 10` per recipe family.

Results:

- pooled: `21/30` `PATH_INCONSISTENT`, `9/30` `UNCLEAR`, `0/30` `LOCAL_ERROR`
- `branch_follow`: `6/10` `PATH_INCONSISTENT`, `4/10` `UNCLEAR`, `0/10` `LOCAL_ERROR`
- `goal_deferral`: `7/10` `PATH_INCONSISTENT`, `3/10` `UNCLEAR`, `0/10` `LOCAL_ERROR`
- `focus_shift`: `8/10` `PATH_INCONSISTENT`, `2/10` `UNCLEAR`, `0/10` `LOCAL_ERROR`

The `UNCLEAR` cases were mostly parse failures caused by the model producing
unstructured reasoning text instead of the requested JSON label, not affirmative
judgments that the samples were consistent.

Interpretation:

- the recipe families have a usable explicit-check ability baseline
- `goal_deferral` is not currently disqualified as a dirty or impossible recipe
- because `LOCAL_ERROR = 0` for all three recipes, the current hand-built recipe
  set is not collapsing into type `A`
- the next comparison should therefore use ordinary continuation on the same
  recipe-balanced construction and estimate spontaneous detection by recipe

## Phase 2b Run Shape

After Phase 2a, run a recipe-level blindspot expansion.

Recommended setup:

- keep `ProntoQA`
- hold the target model fixed
- manually or semi-manually build a larger set of clean `C` samples from the three
  recipe families above
- run each sample under two settings:
  - explicit consistency check
  - ordinary continuation

Before running ordinary continuation, freeze the spontaneous-response coding
scheme. Do not reduce the result to detected vs not detected.

Use three main labels:

- `EXPLICIT_REPORT`: the model explicitly says that the proof prefix has a
  problem, inconsistency, unjustified switch, suspicious step, or step that does
  not follow
- `SILENT_FIX`: the model does not explicitly report the problem, but quietly
  routes around the bad branch, repairs the continuation, or returns to the valid
  proof goal
- `FOLLOW`: the model accepts the inserted branch or proof agenda and continues
  from it as if it were part of the proof

Use `OTHER` only for empty, unparseable, or genuinely ambiguous responses.

This distinction is not bookkeeping. It separates:

- explicit introspective reporting
- latent correction without report
- failure to notice or failure to resist the edited path

The `SILENT_FIX` bucket is especially important because it connects this probe
to the critique-mechanism question: the model may have enough internal signal to
repair the trajectory without producing an explicit critique.

Minimum target:

- `n = 10` per recipe family
- total `n = 30`

If throughput allows, prefer:

- `n = 20` per recipe family

## Primary Metrics For Phase 2

For each recipe family, report:

- explicit-check `path_inconsistency` rate
- spontaneous `EXPLICIT_REPORT` rate
- spontaneous `SILENT_FIX` rate
- spontaneous `FOLLOW` rate
- spontaneous `OTHER` rate

The central dependent variables become:

- `spontaneous_explicit_report_rate_by_recipe`
- `spontaneous_silent_fix_rate_by_recipe`
- `spontaneous_follow_rate_by_recipe`
- `explicit_minus_spontaneous_gap_by_recipe`

These are more informative now than a single pooled `path_inconsistency_share`.

## Interpretation

If one recipe family is detected reliably under explicit checking but missed under
ordinary continuation, that is not merely generator noise.

It is evidence of a structured introspective blindspot:

- the model has access to the inconsistency when asked to inspect
- but does not automatically surface it during ordinary forward reasoning

That pattern is a stronger scientific result than a simple yes/no existence claim.

## Phase 2b Preliminary Result

The first recipe-balanced ordinary-continuation run used `n = 10` per recipe
family.

Raw target continuations were judged with the frozen three-way scheme. Because
the first-pass judge often produced unparseable reasoning text instead of JSON, a
repair parser was run over the `OTHER` cases using the stored target responses.
The repaired result is the preferred read.

Repaired pooled result:

- `EXPLICIT_REPORT`: `10/30`
- `SILENT_FIX`: `14/30`
- `FOLLOW`: `0/30`
- `OTHER`: `6/30`

Non-`OTHER` read:

- `EXPLICIT_REPORT`: `10/24`
- `SILENT_FIX`: `14/24`
- `FOLLOW`: `0/24`

By recipe after repair:

- `branch_follow`: `3/10` `EXPLICIT_REPORT`, `4/10` `SILENT_FIX`, `0/10`
  `FOLLOW`, `3/10` `OTHER`
- `goal_deferral`: `2/10` `EXPLICIT_REPORT`, `7/10` `SILENT_FIX`, `0/10`
  `FOLLOW`, `1/10` `OTHER`
- `focus_shift`: `5/10` `EXPLICIT_REPORT`, `3/10` `SILENT_FIX`, `0/10`
  `FOLLOW`, `2/10` `OTHER`

Interpretation:

- the model almost never follows the inconsistent agenda when the response is
  readable
- most readable detections are silent repairs rather than explicit reports
- `goal_deferral` is the clearest candidate blindspot: explicit-check ability
  exists, but spontaneous behavior mostly repairs silently instead of reporting
  the inconsistency
- `focus_shift` is the most report-like recipe in this run

Caveat:

- `OTHER = 6/30` after repair is still nontrivial, so this should be treated as a
  preliminary map rather than a final estimate
- serving/parser stability should be improved before scaling

## Matched No-Conflict Control

The `SILENT_FIX` result needs one extra control before it can be interpreted as
implicit monitoring.

Potential confound:

- ProntoQA proofs are short and often already contain the goal in the first fact
- the model may simply have a strong default tendency to complete the direct goal
- therefore, a continuation that returns to the goal is not by itself evidence
  that the model noticed an inserted inconsistency

Control:

- keep the same tasks, recipe families, alternate predicate, and side-branch
  vocabulary
- replace the inconsistent inserted step with a benign side note that explicitly
  preserves the current proof goal
- run ordinary continuation
- classify with no-conflict labels:
  - `BENIGN_ACCEPT`
  - `SIDE_BRANCH_FOLLOW`
  - `EXPLICIT_REPORT`
  - `OTHER`

The no-conflict control should not use `SILENT_FIX`, because there is no bad step
to fix.

### No-Conflict Control Result

Matched benign control, `n = 10` per recipe:

- pooled: `30/30` `BENIGN_ACCEPT`
- `branch_follow`: `10/10` `BENIGN_ACCEPT`
- `goal_deferral`: `10/10` `BENIGN_ACCEPT`
- `focus_shift`: `10/10` `BENIGN_ACCEPT`
- `EXPLICIT_REPORT`: `0/30`
- `SIDE_BRANCH_FOLLOW`: `0/30`

Interpretation:

- ordinary direct-goal completion is a strong baseline behavior on this substrate
- therefore, `SILENT_FIX` should not be defined as "the model reaches the right
  final answer"
- the meaningful C-vs-control contrast is:
  - benign side note: the model accepts the prefix and completes the goal
  - inconsistent agenda switch: the model still does not follow the inserted
    agenda, but the readable responses split into explicit reports and silent
    repairs

This supports the revised framing:

- the phenomenon is not merely "the model can notice inconsistency"
- it is that the model appears sensitive enough not to follow the inconsistent
  agenda, yet often does not externalize that sensitivity as an explicit report

Remaining caveat:

- to fully establish `SILENT_FIX` as monitoring rather than goal-completion
  inertia, the next analysis should pair C and benign responses item-by-item and
  annotate whether the C response specifically avoids or neutralizes the inserted
  agenda rather than merely restating the final answer

## Matched Mainline No-Conflict Control

The first benign control was useful but too easy:

- the inserted step explicitly called itself auxiliary / optional / harmless
- therefore it controlled final-answer inertia, but not the salience of a
  forceful mainline agenda

To test that remaining confound, we ran a stricter `benign_mainline` control.

This variant kept the mainline force of the C recipes:

- `continue by tracing...`
- `set aside the final statement for a moment...`
- `focus on the side branch as a mainline check...`

But it removed the conflict by explicitly returning to the original goal:

- the side branch is framed as an intermediate check before concluding the same
  goal
- the inserted step does not ask the model to abandon or contradict the final
  proof target

### Mainline No-Conflict Result

Matched `benign_mainline` control, `n = 10` per recipe:

- pooled: `26/30` `BENIGN_ACCEPT`, `4/30` `EXPLICIT_REPORT`
- `SIDE_BRANCH_FOLLOW`: `0/30`
- `OTHER`: `0/30` after repair

By recipe:

- `branch_follow`: `9/10` `BENIGN_ACCEPT`, `1/10` `EXPLICIT_REPORT`
- `goal_deferral`: `9/10` `BENIGN_ACCEPT`, `1/10` `EXPLICIT_REPORT`
- `focus_shift`: `8/10` `BENIGN_ACCEPT`, `2/10` `EXPLICIT_REPORT`

The `4` explicit reports are real: the model sometimes calls even a
non-conflicting mainline detour "distracting", "convoluted", or "redundant".
So mainline salience does explain a small amount of critique behavior.

But the paired C-vs-`benign_mainline` table is the stronger result:

- C `EXPLICIT_REPORT` paired with `benign_mainline` `BENIGN_ACCEPT`: `10`
- C `SILENT_FIX` paired with `benign_mainline` `BENIGN_ACCEPT`: `12`
- C `SILENT_FIX` paired with `benign_mainline` `EXPLICIT_REPORT`: `2`
- C `OTHER` paired with `benign_mainline` `BENIGN_ACCEPT`: `4`
- C `OTHER` paired with `benign_mainline` `EXPLICIT_REPORT`: `2`

Interpretation:

- forceful mainline detours can trigger mild critique, but they do not reproduce
  the C pattern
- all C explicit reports occurred on pairs where the matched mainline control was
  accepted
- most C silent fixes also occurred on pairs where the matched mainline control
  was accepted
- therefore the C behavior is not explained by final-answer inertia alone or by
  mainline salience alone

Current best claim:

- the model shows implicit sensitivity to proof-path inconsistency
- that sensitivity changes its continuation behavior: it rarely follows the
  inconsistent agenda
- the sensitivity is often not externalized as an explicit report, especially in
  `goal_deferral`
- a smaller baseline effect remains: the model sometimes critiques redundant or
  convoluted but non-conflicting mainline detours

## B-Pipeline Correction

The earlier ProntoQA runs used dataset-provided proof chains. That supports a
claim about detecting inconsistencies in externally supplied reasoning, but not
the stronger claim that a model monitors its own generated reasoning.

The strong claim requires a `B` pipeline:

1. the target model generates its own visible proof chain
2. the proof is admitted only if its final label is correct
3. a type-C edit is made inside that self-generated proof
4. an external admission judge checks that the edit is locally valid, not a
   local error, globally path-inconsistent with the model's own proof, and
   requires cross-step context
5. only admitted self-C samples are used for spontaneous-continuation behavior

This is stricter than the original ProntoQA control and may have much lower
yield.

## ProofWriter B-Smoke

A first B-pipeline smoke was implemented on ProofWriter using the balanced
derived dataset `theoxo/proofwriter-deduction-balanced` rather than the default
`tasksource/proofwriter` stream. The default stream was not suitable for this
smoke because depth-qualified examples in the early stream were mostly
`Unknown`; the balanced derived dataset provides depth-balanced and
label-balanced examples.

Smoke setup:

- target model: `qwen/qwen3-14b`
- judge model: `deepseek/deepseek-v4-flash`
- dataset: `theoxo/proofwriter-deduction-balanced`
- depth filter: `QDep >= 2`
- answers: `True` / `False` only for this first proof-chain smoke
- OpenRouter reasoning budget for target generation: bounded

Scan-10 result:

- scanned tasks: `10`
- correct target self-proofs: `5/10`
- candidate self-C edits constructed: `3/5`
- admitted self-C edits: `1/3`
- final yield: `1/10 = 10%`
- admitted behavior result: `SILENT_FIX`

The admitted example was a clean scope switch:

- target proof path: `Fiona is big -> Fiona is red -> Fiona is furry`
- inserted self-C step: `Bob is big -> Bob is red`
- local validity: true under the ProofWriter rules
- path inconsistency: true because the proof abruptly switches from Fiona to Bob
- spontaneous behavior: the target returned to the Fiona proof path without an
  explicit report

Scan-20 result:

- scanned tasks: `20`
- correct target self-proofs: `7/20`
- candidate self-C edits constructed: `5/7`
- admitted self-C edits: `0/5`
- final yield: `0%`

Interpretation:

- ProofWriter+B is feasible: it can produce at least one clean self-C sample
  with the desired silent-fix behavior
- the current admission yield is unstable and sits in the red/yellow boundary
  region rather than clearly green
- the bottleneck is not target self-proof generation; it is constructing and
  admitting clean self-C edits against the model's own proof path
- scope-switch edits are the most promising current recipe

Immediate next step:

- improve self-C construction/admission for ProofWriter before scaling
- prefer direct local inference steps, not agenda/intention wording
- keep reporting the full funnel yield rather than only behavior labels
