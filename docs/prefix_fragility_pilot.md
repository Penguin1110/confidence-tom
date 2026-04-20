# Prefix Fragility Pilot

- rows: `24`
- normalized correctness changed rate: `0.292`
- rewritten correctness changed rate: `0.292`
- normalized answer changed rate: `0.708`
- rewritten answer changed rate: `0.792`

## By Benchmark

- `livebench_reasoning`: rows=12, rewrite_correctness_change=0.167, normalize_correctness_change=0.250
- `olympiadbench`: rows=12, rewrite_correctness_change=0.417, normalize_correctness_change=0.333

## By Small Family

- `livebench`: rows=12, rewrite_correctness_change=0.167, normalize_correctness_change=0.250
- `llama`: rows=4, rewrite_correctness_change=0.000, normalize_correctness_change=0.000
- `mistral`: rows=4, rewrite_correctness_change=0.250, normalize_correctness_change=0.000
- `qwen`: rows=4, rewrite_correctness_change=1.000, normalize_correctness_change=1.000
