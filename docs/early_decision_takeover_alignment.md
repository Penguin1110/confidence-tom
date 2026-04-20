# Early Decision / Takeover Alignment

這份分析把 Early Decision / MSP 結果接回 oracle gain，檢查 early diagnosability 是否通常早於或同步於 positive takeover opportunity。

## 整體
- tasks with any positive takeover: `28` / `96`
- mean earliest positive step: `1.6428571428571428`
- mean MSP given positive: `2.1666666666666665`
- MSP <= earliest positive rate: `0.6666666666666666`
- cross60 <= earliest positive rate: `0.8947368421052632`
- cross70 <= earliest positive rate: `0.6923076923076923`

## By Benchmark
- `livebench_reasoning`: positive_tasks=`13`, mean_pos_step=`1.4615384615384615`, mean_msp=`1.2`, MSP<=pos=`0.8`, cross60<=pos=`0.7777777777777778`, cross70<=pos=`0.875`
- `olympiadbench`: positive_tasks=`15`, mean_pos_step=`1.8`, mean_msp=`2.857142857142857`, MSP<=pos=`0.5714285714285714`, cross60<=pos=`1.0`, cross70<=pos=`0.4`

## By Small Family
- `llama`: positive_tasks=`14`, mean_pos_step=`1.3571428571428572`, mean_msp=`2.6`, MSP<=pos=`0.4`, cross60<=pos=`0.8333333333333334`, cross70<=pos=`0.5`
- `mistral`: positive_tasks=`11`, mean_pos_step=`1.9090909090909092`, mean_msp=`2.0`, MSP<=pos=`0.8333333333333334`, cross60<=pos=`1.0`, cross70<=pos=`1.0`
- `qwen`: positive_tasks=`3`, mean_pos_step=`2.0`, mean_msp=`1.0`, MSP<=pos=`1.0`, cross60<=pos=`1.0`, cross70<=pos=`1.0`
