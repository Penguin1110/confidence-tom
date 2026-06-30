# Prefix Re-entry Controls

- rows: `1220`
- full rerun match rate: `0.082`
- re-entry match rate: `0.256`
- re-entry repeat match rate: `0.602`
- marker boundary match rate: `0.328`
- fenced boundary match rate: `0.284`
- P(full-trace success | re-entry match): `0.647`
- P(full-trace success | re-entry mismatch): `0.339`
- P(positive takeover | re-entry match): `0.109`
- P(positive takeover | re-entry mismatch): `0.272`

## By Benchmark

- `olympiadbench`: rows=865, reentry_match=0.247, full_rerun_match=0.086, p_pos|match=0.079, p_pos|mismatch=0.226
- `livebench_reasoning`: rows=355, reentry_match=0.276, full_rerun_match=0.073, p_pos|match=0.173, p_pos|mismatch=0.389

## By Small Family

- `qwen`: rows=361, reentry_match=0.366, full_rerun_match=0.100, p_pos|match=0.030, p_pos|mismatch=0.153
- `mistral`: rows=361, reentry_match=0.357, full_rerun_match=0.150, p_pos|match=0.109, p_pos|mismatch=0.297
- `llama`: rows=498, reentry_match=0.102, full_rerun_match=0.020, p_pos|match=0.314, p_pos|mismatch=0.320
