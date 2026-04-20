# Probability-Stable MSP Analysis

這版 MSP 比離散版更嚴格：從某一步開始，對正確 label 的機率必須一直高於門檻。

## Threshold `0.6`
- coverage: `31/96` (0.323)
- mean MSP: `1.5806451612903225`
- distribution: `{'1': 23, '2': 2, '3': 4, '4': 1, '6': 1}`

### By Benchmark
- `livebench_reasoning`: coverage=`9/18` (0.500), mean_msp=`1.4444444444444444`, dist=`{'1': 7, '2': 1, '4': 1}`
- `olympiadbench`: coverage=`22/78` (0.282), mean_msp=`1.6363636363636365`, dist=`{'1': 16, '2': 1, '3': 4, '6': 1}`

### By Small Family
- `llama`: coverage=`11/32` (0.344), mean_msp=`1.3636363636363635`, dist=`{'1': 9, '2': 1, '4': 1}`
- `mistral`: coverage=`11/32` (0.344), mean_msp=`1.4545454545454546`, dist=`{'1': 10, '6': 1}`
- `qwen`: coverage=`9/32` (0.281), mean_msp=`2.0`, dist=`{'1': 4, '2': 1, '3': 4}`

## Threshold `0.7`
- coverage: `9/96` (0.094)
- mean MSP: `2.3333333333333335`
- distribution: `{'1': 2, '2': 3, '3': 3, '4': 1}`

### By Benchmark
- `livebench_reasoning`: coverage=`5/18` (0.278), mean_msp=`2.2`, dist=`{'1': 2, '2': 1, '3': 1, '4': 1}`
- `olympiadbench`: coverage=`4/78` (0.051), mean_msp=`2.5`, dist=`{'2': 2, '3': 2}`

### By Small Family
- `llama`: coverage=`3/32` (0.094), mean_msp=`1.6666666666666667`, dist=`{'1': 1, '2': 2}`
- `mistral`: coverage=`2/32` (0.062), mean_msp=`1.5`, dist=`{'1': 1, '2': 1}`
- `qwen`: coverage=`4/32` (0.125), mean_msp=`3.25`, dist=`{'3': 3, '4': 1}`

## Threshold `0.8`
- coverage: `3/96` (0.031)
- mean MSP: `2.3333333333333335`
- distribution: `{'1': 1, '3': 2}`

### By Benchmark
- `livebench_reasoning`: coverage=`1/18` (0.056), mean_msp=`1.0`, dist=`{'1': 1}`
- `olympiadbench`: coverage=`2/78` (0.026), mean_msp=`3.0`, dist=`{'3': 2}`

### By Small Family
- `llama`: coverage=`0/32` (0.000), mean_msp=`None`, dist=`{}`
- `mistral`: coverage=`1/32` (0.031), mean_msp=`1.0`, dist=`{'1': 1}`
- `qwen`: coverage=`2/32` (0.062), mean_msp=`3.0`, dist=`{'3': 2}`
