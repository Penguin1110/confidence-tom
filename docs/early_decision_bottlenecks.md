# Early Decision Bottleneck Analysis

這版先做最穩的 step-level bottleneck proxy：看哪一步讓對正確 label 的機率跳最多，以及最早在哪一步跨過固定門檻。

## 整體
- max_jump_step distribution: `{'1': 95, '3': 1}`
- first_cross_60 distribution: `{'1': 39, '2': 6, '3': 4, '4': 1}`
- first_cross_70 distribution: `{'1': 8, '2': 17, '3': 4, '4': 1}`

## By Benchmark
- `livebench_reasoning`: max_jump=`{'1': 18}`, cross60=`{'1': 12, '2': 1, '4': 1}`, cross70=`{'1': 7, '2': 1, '3': 1, '4': 1}`
- `olympiadbench`: max_jump=`{'1': 77, '3': 1}`, cross60=`{'1': 27, '2': 5, '3': 4}`, cross70=`{'1': 1, '2': 16, '3': 3}`

## By Small Family
- `llama`: max_jump=`{'1': 32}`, cross60=`{'1': 16, '2': 2, '4': 1}`, cross70=`{'1': 3, '2': 11}`
- `mistral`: max_jump=`{'1': 32}`, cross60=`{'1': 17, '2': 1}`, cross70=`{'1': 4, '2': 5}`
- `qwen`: max_jump=`{'1': 31, '3': 1}`, cross60=`{'1': 6, '2': 3, '3': 4}`, cross70=`{'1': 1, '2': 1, '3': 4, '4': 1}`
