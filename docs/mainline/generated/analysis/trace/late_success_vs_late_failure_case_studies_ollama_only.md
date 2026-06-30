# Case Studies: Late-Success vs Late-Failure (Ollama-only)

Source: server-synced `results/`, filtered to `small_model.startswith("ollama/")`. These examples are drawn from the 30-task LiveBench reasoning runs used in the ollama-only taxonomy summary.

## fragile_late_failure

- Category: `late-failure`
- Why this case: Late-failure where correctness appears only briefly and too late to stabilize
- Result file: `results/ordered_gemma3_livebench_gemma_to_openai_30/Ollama_gemma3_4b_to_GPT_5_4.json`
- Task id: `livebench_reasoning_84df2180fa0e999d23d30017d9e1f01bb99e58b280b35ecd8508881a8d31ed5d_0018`
- Small model: `ollama/gemma3:4b`
- Large model: `openai/gpt-5.4`
- Full trace correct: `False`
- Full trace answer: `yes, yes, yes`
- Step count: `6`
- First correct frac: `0.833`
- Local correct rate: `0.167`
- Last small correct: `False`

Prefix outcomes:
- Step 1: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 2: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 3: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no, yes, no`
- Step 4: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `The person at the gym tells the truth (yes), the person at the zoo lies (no), and the person at the bookstore lies (no).`
- Step 5: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, yes, yes`
- Step 6: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `yes, yes, no`

Segment summaries:
- Segment 1: We establish the locations of the individuals: Devika (B), Zara (BK), Ethan (BG), Priya (V), Ryan (O), Grace (M), Farid (G), Olivia (C), Isabella (Z), and Hiroshi (F). Each person is either a truth-teller or a liar.
- Segment 2: We analyze the statements: Hiroshi (F) says the person at the gym (G) lies. Since the person at the farm (F) lies, Hiroshi must be a truth-teller, meaning the person at the gym (Farid) is a liar.
- Segment 3: Farid (G) says the person at the vineyard (V) lies. Since Farid is a liar, the person at the vineyard (Priya) must be a truth-teller. Olivia (C) says the person at the vineyard (V) tells the truth; since Priya is a...
- Segment 4: Ryan (O) says the person at the campground (C) tells the truth. Since Olivia (C) is a truth-teller, Ryan is a truth-teller. Grace (M) says the person at the observatory (O) tells the truth; since Ryan is a truth-...
- Segment 5: Isabella (Z) says the person at the bookstore (BK) tells the truth. Zara (BK) says the person at the library (L) lies. Emily (L) says the person at the museum (M) lies. Since Grace (M) is a truth-teller, Emily (L) is...
- Segment 6: Devika (B) says the person at the zoo (Z) tells the truth. Since Isabella (Z) is a truth-teller, Devika is a truth-teller.

Read:
- Correctness appears only weakly or briefly, and the continuation cannot stabilize it. This is a more fragile late-failure pattern than the stronger rescue-then-collapse case above.

## fragile_late_failure

- Category: `late-failure`
- Why this case: Late-failure where correctness appears only briefly and too late to stabilize
- Result file: `results/ordered_gemma3_livebench_gemma_to_anthropic_30/Ollama_gemma3_4b_to_Claude_Opus_4_6.json`
- Task id: `livebench_reasoning_84df2180fa0e999d23d30017d9e1f01bb99e58b280b35ecd8508881a8d31ed5d_0018`
- Small model: `ollama/gemma3:4b`
- Large model: `anthropic/claude-opus-4.6`
- Full trace correct: `False`
- Full trace answer: `yes, yes, yes`
- Step count: `6`
- First correct frac: `0.833`
- Local correct rate: `0.167`
- Last small correct: `False`

Prefix outcomes:
- Step 1: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 2: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `yes, no, no`
- Step 3: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no`
- Step 4: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `yes, yes, yes`
- Step 5: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no, no, yes`

Segment summaries:
- Segment 1: Initial setup identifies the locations of nine individuals: Devika (B), Zara (BK), Ethan (BG), Priya (V), Ryan (O), Grace (M), Farid (G), Olivia (C), and Isabella (Z). Each person is either a truth-teller or a liar.
- Segment 2: The person at the farm (Hiroshi) lies, which implies that the statement made by the person at the gym (Farid) is false. Since Hiroshi (F) says the person at the gym (G) is lying, and the person at the gym is indeed...
- Segment 3: Analyzing the chain of statements: Ryan (O) says the person at the campground (C) tells the truth. Olivia (C) says the person at the vineyard (V) tells the truth. Farid (G) says the person at the vineyard (V) lies....
- Segment 4: Grace (M) says the person at the observatory (O) tells the truth. Since Ryan (O) is a truth-teller, Grace (M) is a truth-teller. Emily (L) says the person at the museum (M) lies. Since Grace (M) is a truth-teller,...
- Segment 5: Zara (BK) says the person at the library (L) lies. Since Emily (L) is a liar, Zara (BK) is a truth-teller. Isabella (Z) says the person at the bookstore (BK) tells the truth. Since Zara (BK) is a truth-teller,...

Read:
- Correctness appears only weakly or briefly, and the continuation cannot stabilize it. This is a more fragile late-failure pattern than the stronger rescue-then-collapse case above.

## strong_late_failure

- Category: `late-failure`
- Why this case: Late-failure with locally correct prefixes throughout, yet wrong final full trace
- Result file: `results/ordered_gemma3_livebench_gemma_to_openai_30/Ollama_gemma3_4b_to_GPT_5_4.json`
- Task id: `livebench_reasoning_b183db3494f4b53e5c292683be85069129641ea43070f3e405884573b4d6cf7d_0012`
- Small model: `ollama/gemma3:4b`
- Large model: `openai/gpt-5.4`
- Full trace correct: `False`
- Full trace answer: `no, yes, no`
- Step count: `3`
- First correct frac: `0.333`
- Local correct rate: `1.000`
- Last small correct: `True`

Prefix outcomes:
- Step 1: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `yes, no, yes`
- Step 2: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `yes, no, yes`
- Step 3: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `yes, no, yes`

Segment summaries:
- Segment 1: We establish the truth values for several locations: the person at the city hall (Quan) tells the truth, while the persons at the restaurant (Beatriz), ice skating rink (Farid), and hotel (Mateo) are lying. The person...
- Segment 2: The person at the art gallery (Luna) states that the person at the bookstore tells the truth. Since the bookstore person is indeed truthful, Luna is telling the truth.
- Segment 3: We analyze the statements of Zara (barbershop), Ethan (zoo), and Emily (planetarium). Zara claims Emily lies; Ethan claims Luna tells the truth; Emily claims Ethan lies. Through logical deduction and testing the truth...

Read:
- This is the clearest “could have been right” failure: the re-entry probe repeatedly finds correct local continuations, but the original full trace does not preserve that state to the end.

## strong_late_failure

- Category: `late-failure`
- Why this case: Late-failure with locally correct prefixes throughout, yet wrong final full trace
- Result file: `results/ordered_gemma3_livebench_gemma_to_anthropic_30/Ollama_gemma3_4b_to_Claude_Opus_4_6.json`
- Task id: `livebench_reasoning_b183db3494f4b53e5c292683be85069129641ea43070f3e405884573b4d6cf7d_0012`
- Small model: `ollama/gemma3:4b`
- Large model: `anthropic/claude-opus-4.6`
- Full trace correct: `False`
- Full trace answer: `The person at the barbershop lies, the person at the planetarium tells the truth, and the person at the zoo lies.`
- Step count: `3`
- First correct frac: `0.333`
- Local correct rate: `1.000`
- Last small correct: `True`

Prefix outcomes:
- Step 1: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no, yes, no`
- Step 2: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `no, yes, no`
- Step 3: `small_continue_correct=False`, `large_takeover_correct=False`, small answer `yes, yes, no`

Segment summaries:
- Segment 1: We are given that the person at the city hall (Quan) tells the truth, while the persons at the restaurant (Beatriz), ice skating rink (Farid), and hotel (Mateo) are lying. The person at the bookstore (Tala) is also...
- Segment 2: Luna is at the art gallery and thinks the person at the bookstore (Tala) is lying. However, since Tala is confirmed to be telling the truth, Luna's statement is false, meaning Luna is lying.
- Segment 3: Analyzing the remaining individuals: Zara is at the barbershop, Ethan is at the zoo, and Emily is at the planetarium. Ethan (at the zoo) says Luna tells the truth; since Luna is lying, Ethan is lying. Emily (at the...

Read:
- This is the clearest “could have been right” failure: the re-entry probe repeatedly finds correct local continuations, but the original full trace does not preserve that state to the end.

## edge_late_success

- Category: `late-success`
- Why this case: Edge-case late-success: final answer correct but no local small-model prefix succeeds
- Result file: `results/ordered_gemma3_livebench_gemma_to_openai_30/Ollama_gemma3_4b_to_GPT_5_4.json`
- Task id: `livebench_reasoning_a0f4b0be2055ef97c455771f2de3eb3f41c9018603d96c93225be734b70dcdce_0009`
- Small model: `ollama/gemma3:4b`
- Large model: `openai/gpt-5.4`
- Full trace correct: `True`
- Full trace answer: `4`
- Step count: `3`
- First correct frac: `1.000`
- Local correct rate: `0.000`
- Last small correct: `False`

Prefix outcomes:
- Step 1: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `7`
- Step 2: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `7`
- Step 3: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `7`

Segment summaries:
- Segment 1: We start with a regular pentagon ABCDE. The cut AC divides the pentagon into two polygons: triangle ABC and quadrilateral ACDE. The cut BD divides the pentagon into two polygons: triangle BCD and quadrilateral ABDE.
- Segment 2: The cuts AC and BD intersect at a single point inside the pentagon. When two lines (or segments) intersect within a region, they divide the original area into additional pieces. Specifically, the intersection of these...
- Segment 3: Re-evaluating the geometry: The cut AC creates two pieces. The cut BD then intersects the existing pieces. The intersection of the two diagonals AC and BD creates four regions: triangle ABC, triangle BCD, triangle...

Read:
- This is the taxonomy edge case: the full trace reaches the right answer, but the local prefix probes do not reveal a reusable correct state. It reminds us that the current taxonomy lumps these into `late-success`.

## edge_late_success

- Category: `late-success`
- Why this case: Edge-case late-success: final answer correct but no local small-model prefix succeeds
- Result file: `results/ordered_gemma3_livebench_gemma_to_anthropic_30/Ollama_gemma3_4b_to_Claude_Opus_4_6.json`
- Task id: `livebench_reasoning_a0f4b0be2055ef97c455771f2de3eb3f41c9018603d96c93225be734b70dcdce_0009`
- Small model: `ollama/gemma3:4b`
- Large model: `anthropic/claude-opus-4.6`
- Full trace correct: `True`
- Full trace answer: `4`
- Step count: `3`
- First correct frac: `1.000`
- Local correct rate: `0.000`
- Last small correct: `False`

Prefix outcomes:
- Step 1: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `7`
- Step 2: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `7`
- Step 3: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `7`

Segment summaries:
- Segment 1: We start with a regular pentagon ABCDE. Making the cut AC divides the pentagon into two polygons: triangle ABC and quadrilateral ACDE. Making the cut BD divides the pentagon into two polygons: triangle BCD and...
- Segment 2: The cuts AC and BD intersect at a single point inside the pentagon. Because the two diagonals intersect, the pentagon is divided into four distinct regions: triangle ABC, triangle BCD, triangle AED, and the central...
- Segment 3: Evaluating the geometry of the cuts, the intersection of the two diagonals AC and BD within the pentagon results in the pentagon being divided into 4 pieces.

Read:
- This is the taxonomy edge case: the full trace reaches the right answer, but the local prefix probes do not reveal a reusable correct state. It reminds us that the current taxonomy lumps these into `late-success`.

## strong_late_success

- Category: `late-success`
- Why this case: Late-success with strong persistence after first correct prefix
- Result file: `results/ordered_gemma4_livebench_gemma_to_openai_30/Ollama_gemma4_e4b_to_GPT_5_4.json`
- Task id: `livebench_reasoning_448411c3001bf1a38e6b6a39a7f7e8a0b213247c667d2b8b1ddb8d1dc4ed73a9_0002`
- Small model: `ollama/gemma4:e4b`
- Large model: `openai/gpt-5.4`
- Full trace correct: `True`
- Full trace answer: `no, no, no`
- Step count: `6`
- First correct frac: `0.167`
- Local correct rate: `0.833`
- Last small correct: `True`

Prefix outcomes:
- Step 1: `small_continue_correct=True`, `large_takeover_correct=False`, small answer `no, no, no`
- Step 2: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `Library: No, Park: No, Bookstore: No`
- Step 3: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 4: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 5: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 6: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`

Segment summaries:
- Segment 1: Establish known truth values: Nadia (Farm) is a truth-teller (T) and Emily (Museum) is a liar (L).
- Segment 2: Determine Isabella's status (Hotel): Isabella says Emily lies. Since Emily is a liar, Isabella's statement is true, so Isabella is a truth-teller (T).
- Segment 3: Determine Nia's status (Bookstore): Nia says Isabella lies. Since Isabella is a truth-teller, Nia's statement is false, so Nia is a liar (L).
- Segment 4: Determine Jaxon's status (Park): Jaxon says Nia tells the truth. Since Nia is a liar, Jaxon's statement is false, so Jaxon is a liar (L).
- Segment 5: Determine Ryan's status (Library): Ryan says Jaxon tells the truth. Since Jaxon is a liar, Ryan's statement is false, so Ryan is a liar (L).
- Segment 6: Conclusion: Ryan (Library), Jaxon (Park), and Nia (Bookstore) are all liars.

Read:
- A reusable correct state appears only after some reasoning has accumulated, but once it appears it is preserved through most later prefixes and the full trace lands correctly.

## strong_late_success

- Category: `late-success`
- Why this case: Late-success with strong persistence after first correct prefix
- Result file: `results/ordered_gemma4_livebench_gemma_to_anthropic_30/Ollama_gemma4_e4b_to_Claude_Opus_4_6.json`
- Task id: `livebench_reasoning_448411c3001bf1a38e6b6a39a7f7e8a0b213247c667d2b8b1ddb8d1dc4ed73a9_0002`
- Small model: `ollama/gemma4:e4b`
- Large model: `anthropic/claude-opus-4.6`
- Full trace correct: `True`
- Full trace answer: `no, no, no`
- Step count: `6`
- First correct frac: `0.167`
- Local correct rate: `0.833`
- Last small correct: `True`

Prefix outcomes:
- Step 1: `small_continue_correct=False`, `large_takeover_correct=True`, small answer `Library: False, Park: False, Bookstore: False`
- Step 2: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 3: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 4: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`
- Step 5: `small_continue_correct=True`, `large_takeover_correct=True`, small answer `no, no, no`

Segment summaries:
- Segment 1: Establish known truth values: Nadia (Farm) is a truth-teller (T) and Emily (Museum) is a liar (L).
- Segment 2: Determine Isabella's status (Hotel): Isabella says Emily lies. Since Emily is L, Isabella's statement is true, so Isabella is T.
- Segment 3: Determine Nia's status (Bookstore): Nia says Isabella lies. Since Isabella is T, Nia's statement is false, so Nia is L.
- Segment 4: Determine Jaxon's status (Park): Jaxon says Nia tells the truth. Since Nia is L, Jaxon's statement is false, so Jaxon is L.
- Segment 5: Determine Ryan's status (Library): Ryan says Jaxon tells the truth. Since Jaxon is L, Ryan's statement is false, so Ryan is L.

Read:
- A reusable correct state appears only after some reasoning has accumulated, but once it appears it is preserved through most later prefixes and the full trace lands correctly.
