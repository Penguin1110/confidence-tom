# Prefix Re-entry Controls

- rows: `55`
- full rerun match rate: `0.145`
- re-entry match rate: `0.200`
- re-entry repeat match rate: `0.182`
- marker boundary match rate: `0.145`
- fenced boundary match rate: `0.145`
- P(full-trace success | re-entry match): `0.455`
- P(full-trace success | re-entry mismatch): `0.250`
- P(positive takeover | re-entry match): `0.000`
- P(positive takeover | re-entry mismatch): `0.295`

## By Benchmark

- `olympiadbench`: rows=16, reentry_match=0.250, full_rerun_match=0.312, p_pos|match=0.000, p_pos|mismatch=0.000
- `livebench_reasoning`: rows=39, reentry_match=0.179, full_rerun_match=0.077, p_pos|match=0.000, p_pos|mismatch=0.406

## By Small Family

- `qwen`: rows=55, reentry_match=0.200, full_rerun_match=0.145, p_pos|match=0.000, p_pos|mismatch=0.295
