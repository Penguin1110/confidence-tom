# Minimal Sufficient Prefix Analysis

## 設定
- target: `small_full_trace_success`
- feature set: `state + small_family + benchmark`
- 定義: 最早一步 `t*`，使得該步 prediction 已等於真實 label，且之後所有步都不再翻轉

## 整體
- test tasks: `96`
- tasks with MSP: `45` / `96` (0.469)
- mean MSP: `1.9555555555555555`
- distribution: `{'1': 27, '2': 7, '3': 3, '4': 3, '5': 4, '6': 1}`

## By Benchmark
### `livebench_reasoning`
- coverage: `10/18` (0.556)
- mean MSP: `1.1`
- distribution: `{'1': 9, '2': 1}`

### `olympiadbench`
- coverage: `35/78` (0.449)
- mean MSP: `2.2`
- distribution: `{'1': 18, '2': 6, '3': 3, '4': 3, '5': 4, '6': 1}`

## By Small Family
### `llama`
- coverage: `16/32` (0.500)
- mean MSP: `2.6875`
- distribution: `{'1': 6, '2': 3, '3': 1, '4': 3, '5': 2, '6': 1}`

### `mistral`
- coverage: `10/32` (0.312)
- mean MSP: `2.2`
- distribution: `{'1': 4, '2': 4, '5': 2}`

### `qwen`
- coverage: `19/32` (0.594)
- mean MSP: `1.2105263157894737`
- distribution: `{'1': 17, '3': 2}`

## Representative Early-MSP Tasks
- `livebench_qwen_to_openai_30 :: livebench_reasoning_7f1b41d1cdf3a3cf65c6107f8bb29f112137dc875cff80dc667c4cab83c4037a_0010`: msp=`1`, prefixes=`1`, label=`1`
- `livebench_qwen_to_anthropic_30 :: livebench_reasoning_7f1b41d1cdf3a3cf65c6107f8bb29f112137dc875cff80dc667c4cab83c4037a_0010`: msp=`1`, prefixes=`1`, label=`1`
- `livebench_mistral_to_anthropic_30 :: livebench_reasoning_43a1e0948c6ac3b20ddc4282804434aac8a70bc3ba9dfea9c28c32977c7ab9bf_0024`: msp=`1`, prefixes=`3`, label=`1`
- `mistral_to_anthropic_50 :: olympiadbench_2257_0016`: msp=`1`, prefixes=`3`, label=`1`
- `qwen_to_openai_50 :: olympiadbench_2667_0039`: msp=`1`, prefixes=`3`, label=`1`
- `qwen_to_anthropic_50 :: olympiadbench_2976_0048`: msp=`1`, prefixes=`3`, label=`1`
- `livebench_llama_to_openai_30 :: livebench_reasoning_3de6bc30b87b4698a32f4e705b8e1b7aeb4a37b06bc53da26ce9739720f13a62_0015`: msp=`1`, prefixes=`4`, label=`0`
- `livebench_mistral_to_openai_30 :: livebench_reasoning_43a1e0948c6ac3b20ddc4282804434aac8a70bc3ba9dfea9c28c32977c7ab9bf_0024`: msp=`1`, prefixes=`4`, label=`1`

## Representative No-MSP Tasks
- `llama_to_openai_50 :: olympiadbench_2453_0027`: prefixes=`15`, label=`0`
- `llama_to_anthropic_50 :: olympiadbench_2453_0027`: prefixes=`12`, label=`0`
- `llama_to_openai_50 :: olympiadbench_2292_0020`: prefixes=`11`, label=`0`
- `llama_to_openai_50 :: olympiadbench_2349_0032`: prefixes=`11`, label=`0`
- `llama_to_openai_50 :: olympiadbench_1915_0023`: prefixes=`10`, label=`0`
- `llama_to_anthropic_50 :: olympiadbench_2292_0020`: prefixes=`9`, label=`0`
- `llama_to_openai_50 :: olympiadbench_2667_0039`: prefixes=`9`, label=`0`
- `qwen_to_anthropic_50 :: olympiadbench_2349_0032`: prefixes=`7`, label=`0`
