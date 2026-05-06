# Re-entry Mainline Scale Plan

## Goal

把 `re-entry` 從控制實驗提升成 mainline 主軸，讓我們能系統化回答：

- 哪些 small model 會形成可重入的正確 state
- 哪些 small model 會在 full trace 中丟失原本可重入的 state
- 這些 pattern 是否跨 benchmark 穩定存在

## Mainline Shape

新的主線分成三個 phase：

1. `prepare`
   先跑 prefix family sweep，建立可供 re-entry 消費的標準 run directories。
2. `reentry`
   對既有 run 依 benchmark / family / taxonomy category 做 `re-entry stability controls`。
3. `analyze`
   重算 `reentry_summary.json` 並更新 `prefix_reentry_controls.md`。

批次入口：

- `experiments/mainline/run/batch/run_reentry_mainline.py`
- `experiments/mainline/run/batch/reentry_presets.json`

## Current Expanded Matrix

目前先放進 mainline 的大規模 preset：

- `reentry_livebench_ollama`
- `reentry_olympiad_ollama`

對應 config：

- `configs/prefix_family_sweep_reentry_livebench_ollama.yaml`
- `configs/prefix_family_sweep_reentry_olympiad_ollama.yaml`

這兩個 config 目前把 `re-entry` 放在既有 bench 上放大：

- `LiveBench reasoning`
- `OlympiadBench`

small-model family 目前優先納入：

- `Qwen`: `qwen3:14b`, `qwen3.5:27b`
- `Gemma`: `gemma3:12b`, `gemma4:31b`
- `Mistral`: `ministral-3:3b`, `mistral-small3.2:24b`
- `OLMo`: `olmo-3.1:32b`

## Suggested Next Datasets

在不破壞現有 static-eval pipeline 的前提下，下一批最值得接的是：

1. `AIME 2024`
2. `MATH-500`
3. `GPQA`

原則：

- 先接 objective answer benchmark
- 先讓 `prefix_oracle_steps` 和 `re-entry` 跑通
- judge-heavy benchmark 之後再考慮

## Representation Track

這次先把 `re-entry` 的生成主線做完整，不把 hidden-state probing 綁進核心 runner。

後續建議拆成第二條線：

- `vLLM`: 大量生成
- `transformers`: hidden states / attention 抽取

這樣可以避免把生成吞吐與 probe 顯存需求綁死在同一個 job。

## Smoke Test

先用 dry-run 檢查命令組裝：

```bash
uv run python experiments/mainline/run/batch/run_reentry_mainline.py \
  --preset reentry_livebench_ollama \
  --phase both \
  --dry-run
```

再跑小量 re-entry：

```bash
uv run python experiments/mainline/run/core/run_prefix_reentry_controls.py \
  --run-name-prefix reentry_livebench_ \
  --benchmark livebench_reasoning \
  --max-rows 5 \
  --small-backend ollama \
  --small-local-model-map qwen=qwen3:14b \
  --small-local-model-map gemma=gemma3:4b \
  --small-local-model-map mistral=mistral-small3.2:24b \
  --small-local-model-map olmo=olmo-3.1:32b
```

## Cloud Run Split

Cloud Run 建議拆兩種 job：

- `generation` job：prefix / oracle / re-entry
- `probe` job：hidden state / attention 抽取

第一階段先把 `generation` job 跑穩，再補 `probe` image。
