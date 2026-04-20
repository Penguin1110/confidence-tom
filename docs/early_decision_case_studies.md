# Early Decision Case Studies

這份筆記把 aggregate 指標拆回題目層級，目標是看清楚三件事：

- 哪些題的 early diagnosis 和 early takeover 很早就對齊
- 哪些題的 takeover 機會出現得比穩定診斷更早
- 哪些題根本沒有穩定的 MSP

## Case 1: Early Diagnosis and Early Takeover Align

這題是乾淨的 aligned case。MSP、cross70 和第一個 positive takeover 幾乎同一步成熟，適合用來說明 early diagnosis 可以直接對應 intervention timing。

| Field | Value |
| --- | --- |
| `run_name` | livebench_mistral_to_openai_30 |
| `task_id` | livebench_reasoning_3de6bc30b87b4698a32f4e705b8e1b7aeb4a37b06bc53da26ce9739720f13a62_0015 |
| `benchmark` | livebench_reasoning |
| `small_family` | mistral |
| `large_family` | openai |
| `full_trace_correct` | False |
| `minimal_sufficient_step` | 1 |
| `first_cross_60` | 1 |
| `first_cross_70` | 1 |
| `earliest_positive_step` | 1 |
| `earliest_negative_step` | - |
| `positive_steps` | 1 |
| `negative_steps` | 0 |
| `total_steps` | 5 |

### Segment Outline

- step 1: The problem involves 4 people in a line with unique attributes: Transport (scooter, jet-ski, bike, ship), Nationality (colombian, thai, german, egyptian), Food (cucumber, pomegr...
- step 2: Premise analysis establishes a partial ordering: german must be to the left of cucumber, and cucumber must be to the left of cauliflower (german < cucumber < cauliflower). Premi...
- step 3: Premise 3 and 4 imply that the person who listens to pop is either the german or likes cauliflower, and is either the egyptian or likes pomegranate (but not both). Testing the c...
- step 4: Testing the case where the pop-listener is the german who likes pomegranate allows for a consistent assignment of attributes across the four positions, despite complex interacti...
- step 5: After evaluating the constraints and resolving the relative positions of the attributes, the first person in the line is identified as the german, who likes cucumber.

### Step-Level View

| step | p(correct label) | pred | jump | small | large | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.933 | 0 | 0.933 | 0 | 1 | 1 |
| 2 | 0.809 | 0 | -0.123 | 0 | 0 | 0 |
| 3 | 0.801 | 0 | -0.008 | 0 | 0 | 0 |
| 4 | 0.815 | 0 | 0.014 | 0 | 0 | 0 |
| 5 | 0.836 | 0 | 0.022 | 0 | 0 | 0 |

### Reading

- MSP 不晚於第一個 positive takeover step，代表穩定診斷與介入機會是同步或診斷更早成熟。
- `cross70` 出現在 step 1，可視為較高信心判斷真正形成的時間。
- positive takeover 出現在 step 1。

## Case 2: Positive Takeover Appears Before Stable Diagnosis

這題是 misaligned case。positive takeover 很早就出現，但 MSP 和 cross70 稍後才穩定，說明 intervention opportunity 可能比高信心診斷更早成熟。

| Field | Value |
| --- | --- |
| `run_name` | livebench_llama_to_anthropic_30 |
| `task_id` | livebench_reasoning_7f1b41d1cdf3a3cf65c6107f8bb29f112137dc875cff80dc667c4cab83c4037a_0010 |
| `benchmark` | livebench_reasoning |
| `small_family` | llama |
| `large_family` | anthropic |
| `full_trace_correct` | False |
| `minimal_sufficient_step` | 2 |
| `first_cross_60` | 2 |
| `first_cross_70` | 2 |
| `earliest_positive_step` | 1 |
| `earliest_negative_step` | - |
| `positive_steps` | 5 |
| `negative_steps` | 0 |
| `total_steps` | 5 |

### Segment Outline

- step 1: Analyze the given information: 1. Ice-hockey is not to the right of Thai. 2. Mirinda is to the left of 7up. 3. Mexican likes skydiving. 4. Board-games is on the far left or far ...
- step 2: Determine the partial orderings: Thai and Canadian are to the left of the ice-hockey player. The Mexican person likes skydiving. Based on statement 6, the order for attributes i...
- step 3: Combine constraints to test a configuration: Assume position 1 is Canadian, position 2 is Thai, and position 3 is Mexican. Assigning attributes: Person 1 (Canadian, Mirinda, boa...
- step 4: Verify the configuration against all rules: The setup satisfies all conditions, including the relative positions of beverages, sports, and nationalities.
- step 5: Extract final answers: The person in position 3 drinks 7up, the person in position 2 plays soccer, the person who likes skydiving is in position 3, and the person in position 2 ...

### Step-Level View

| step | p(correct label) | pred | jump | small | large | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.488 | 1 | 0.488 | 0 | 1 | 1 |
| 2 | 0.753 | 0 | 0.265 | 0 | 1 | 1 |
| 3 | 0.827 | 0 | 0.074 | 0 | 1 | 1 |
| 4 | 0.811 | 0 | -0.016 | 0 | 1 | 1 |
| 5 | 0.702 | 0 | -0.109 | 0 | 1 | 1 |

### Reading

- 第一個 positive takeover step 早於 MSP，代表介入機會先出現，但穩定診斷稍後才成熟。
- `cross70` 出現在 step 2，可視為較高信心判斷真正形成的時間。
- positive takeover 出現在 step 1, 2, 3, 4, 5。

## Case 3: No Stable MSP Despite Long Trace

這題代表 long-tail OlympiadBench failure。prediction trajectory 會來回翻，最後沒有形成穩定 MSP，很適合說明 heterogeneous benchmark 的診斷困難。

| Field | Value |
| --- | --- |
| `run_name` | llama_to_openai_50 |
| `task_id` | olympiadbench_2453_0027 |
| `benchmark` | olympiadbench |
| `small_family` | llama |
| `large_family` | openai |
| `full_trace_correct` | False |
| `minimal_sufficient_step` | - |
| `first_cross_60` | 1 |
| `first_cross_70` | 1 |
| `earliest_positive_step` | - |
| `earliest_negative_step` | - |
| `positive_steps` | 0 |
| `negative_steps` | 0 |
| `total_steps` | 15 |

### Segment Outline

- step 1: ## Step 1: Understand the conditions for a "savage" number To find even savage integers less than 100, we must understand the conditions that make a number "savage". A positive ...
- step 2: ## 2: Determine the total sum of numbers from 1 to $n$ The sum of the first $n$ positive integers is given by the formula $S = \frac{n(n + 1)}{2}$. This total sum must be divisi...
- step 3: ## 3: Consider the properties of sets $A, B$, and $C$ Set $A$ contains only odd numbers, set $B$ contains only even numbers, and set $C$ contains every multiple of 3. The sum of...
- step 4: ## 4: Analyze the sum of odd numbers in set $A$ The sum of the first $k$ odd numbers is $k^2$, where $k$ is the number of odd terms. If there are $n_o$ odd numbers, then the sum...
- step 5: ## 5: Analyze the sum of even numbers in set $B$ The sum of the first $m/2$ even numbers is $m(m/2 + 1)$, but since we're considering the sum of even numbers up to $n$, we need ...
- step 6: ## 6: Consider the requirement for $n$ to be even and savage For $n$ to be even and savage, the total sum $S$ must be divisible by 3, and the sums of $A, B$, and $C$ must be equ...

### Step-Level View

| step | p(correct label) | pred | jump | small | large | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.967 | 0 | 0.967 | 0 | 0 | 0 |
| 2 | 0.704 | 0 | -0.263 | 0 | 0 | 0 |
| 3 | 0.680 | 0 | -0.025 | 0 | 0 | 0 |
| 4 | 0.648 | 1 | -0.032 | 0 | 0 | 0 |
| 5 | 0.690 | 0 | 0.042 | 0 | 0 | 0 |
| 6 | 0.647 | 1 | -0.043 | 0 | 0 | 0 |
| 7 | 0.579 | 1 | -0.068 | 0 | 0 | 0 |
| 8 | 0.581 | 1 | 0.002 | 0 | 0 | 0 |
| 9 | 0.542 | 1 | -0.039 | 0 | 0 | 0 |
| 10 | 0.536 | 1 | -0.006 | 0 | 0 | 0 |
| 11 | 0.518 | 1 | -0.018 | 0 | 0 | 0 |
| 12 | 0.433 | 1 | -0.085 | 0 | 0 | 0 |
| 13 | 0.452 | 1 | 0.019 | 0 | 0 | 0 |
| 14 | 0.470 | 1 | 0.017 | 0 | 0 | 0 |
| 15 | 0.393 | 1 | -0.077 | 0 | 0 | 0 |

### Reading

- 整條 trajectory 沒有長出穩定 MSP，表示這題的 outcome 診斷一路都不夠穩。
- `cross70` 出現在 step 1，可視為較高信心判斷真正形成的時間。

## Case 4: Late High-Confidence Bottleneck

這題的高信心 crossing 出現在最後一步，適合拿來說明有些題不是沒有訊號，而是 decisive information 的確偏晚才出現。

| Field | Value |
| --- | --- |
| `run_name` | qwen_to_openai_50 |
| `task_id` | olympiadbench_2746_0038 |
| `benchmark` | olympiadbench |
| `small_family` | qwen |
| `large_family` | openai |
| `full_trace_correct` | False |
| `minimal_sufficient_step` | 3 |
| `first_cross_60` | 3 |
| `first_cross_70` | 3 |
| `earliest_positive_step` | - |
| `earliest_negative_step` | - |
| `positive_steps` | 0 |
| `negative_steps` | 0 |
| `total_steps` | 3 |

### Segment Outline

- step 1: Determine the number of ways to choose letters from each half of the alphabet: The first half (A-M) has 13 letters. The number of ways to choose 2 letters from this group is 78....
- step 2: Calculate the total number of distinct sets of four letters: Since the choices from the two halves are independent, the total number of distinct sets of four letters (2 from the...
- step 3: Compute the probability that both players choose the same set: Once Aditya has chosen a set, Ayesha must choose the exact same set. Since all sets are equally likely, the probab...

### Step-Level View

| step | p(correct label) | pred | jump | small | large | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.382 | 1 | 0.382 | 0 | 0 | 0 |
| 2 | 0.450 | 1 | 0.068 | 0 | 0 | 0 |
| 3 | 0.997 | 0 | 0.547 | 0 | 0 | 0 |

### Reading

- `cross70` 出現在 step 3，可視為較高信心判斷真正形成的時間。
