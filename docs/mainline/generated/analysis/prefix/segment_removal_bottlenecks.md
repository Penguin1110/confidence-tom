# Segment Removal Bottlenecks

這份分析用第一版可重算 probe 檢查：拿掉 `cross70` 對應的 decisive segment 之後，Early Decision 的信心會不會明顯崩掉。

## Overall

- evaluated tasks: `22`
- probe test AUROC: `0.575`
- mean decisive drop: `0.098`
- mean control drop: `0.009`
- mean drop gap: `0.089`
- decisive below 0.7 rate: `1.000`
- control below 0.7 rate: `0.682`

## By Benchmark

### olympiadbench

- count: `19`
- mean decisive drop: `0.091`
- mean control drop: `0.018`
- mean drop gap: `0.073`
- decisive below 0.7 rate: `1.000`
- control below 0.7 rate: `0.789`

### livebench_reasoning

- count: `3`
- mean decisive drop: `0.142`
- mean control drop: `-0.047`
- mean drop gap: `0.189`
- decisive below 0.7 rate: `1.000`
- control below 0.7 rate: `0.000`

## Representative Large Effects

### qwen_to_openai_50 :: olympiadbench_2746_0038

- benchmark: `olympiadbench`
- small_family: `qwen`
- cross70_step: `3`
- original correct-label prob: `0.762`
- decisive-removed prob: `0.467`
- control-removed prob: `0.917`
- decisive drop: `0.296`
- control drop: `-0.155`

### livebench_llama_to_anthropic_30 :: livebench_reasoning_7f1b41d1cdf3a3cf65c6107f8bb29f112137dc875cff80dc667c4cab83c4037a_0010

- benchmark: `livebench_reasoning`
- small_family: `llama`
- cross70_step: `2`
- original correct-label prob: `0.761`
- decisive-removed prob: `0.488`
- control-removed prob: `0.936`
- decisive drop: `0.273`
- control drop: `-0.175`

### qwen_to_anthropic_50 :: olympiadbench_3008_0035

- benchmark: `olympiadbench`
- small_family: `qwen`
- cross70_step: `2`
- original correct-label prob: `0.639`
- decisive-removed prob: `0.477`
- control-removed prob: `0.855`
- decisive drop: `0.162`
- control drop: `-0.216`

### qwen_to_anthropic_50 :: olympiadbench_2746_0038

- benchmark: `olympiadbench`
- small_family: `qwen`
- cross70_step: `3`
- original correct-label prob: `0.837`
- decisive-removed prob: `0.587`
- control-removed prob: `0.924`
- decisive drop: `0.250`
- control drop: `-0.087`

### qwen_to_anthropic_50 :: olympiadbench_2667_0039

- benchmark: `olympiadbench`
- small_family: `qwen`
- cross70_step: `3`
- original correct-label prob: `0.670`
- decisive-removed prob: `0.545`
- control-removed prob: `0.725`
- decisive drop: `0.125`
- control drop: `-0.056`
