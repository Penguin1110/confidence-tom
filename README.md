# Confidence-ToM

這個 repo 現在的主線是：

> **prefix / re-entry / oracle / trace taxonomy**

我們要研究的是：
- 小模型在 prefix / re-entry 時的局部穩定性
- 這些局部訊號能不能預測整題 correctness
- 這些訊號能不能當成 intervention / routing 的 prior

## 研究分層

### Mainline
- `experiments/mainline/`
- `docs/mainline/`

這裡放的是現在的主研究線：
- prefix re-entry controls
- oracle gain / fragility
- trace taxonomy
- prefix predictor / minimal sufficient prefix
- routing / intervention 相關分析

### Core library
- `src/confidence_tom/`

這裡是共用核心程式：
- infra/: client / paths / model_config
- data/: task schemas / dataset loaders / dynamic benchmark adapters
- eval/: evaluators / metrics
- intervention/: features / VOI / router / structured parse
- compat/: 舊 generator / observer 相容層

## 專案結構

```text
confidence-tom/
├── src/confidence_tom/
│   ├── intervention/   # 主線方法：prefix / re-entry / routing / VOI
│   ├── infra/          # client, paths, model config
│   ├── data/           # task schemas, dataset loaders, benchmark adapters
│   ├── eval/           # evaluators and metrics
│   ├── benchmarks/     # benchmark-specific runners/adapters
│   ├── compat/         # 舊 generator / observer 相容層
│   ├── generator/      # 對 compat 的穩定 import shim
│   └── observer/       # 對 compat 的穩定 import shim
├── experiments/
│   └── mainline/
│       ├── run/
│       │   ├── core/
│       │   ├── batch/
│       │   └── remote/
│       ├── analysis/
│       └── data/
├── docs/
│   └── mainline/
│       ├── notes/
│       │   ├── reports/
│       │   └── proposals/
│       └── generated/
│           └── analysis/
├── configs/
├── outputs/
│   ├── results/
│   └── logs/
└── tests/
```

## 安裝

```bash
uv sync --all-groups
uv run pre-commit install
```

## 常用入口

### Mainline rerun / analysis
```bash
uv run python experiments/mainline/run/core/run_prefix_reentry_controls.py --category fragile-success --small-backend local --small-local-model-name Qwen/Qwen3-14B
uv run python experiments/mainline/run/batch/run_reentry_mainline.py --preset reentry_livebench_local --phase all --dry-run
uv run python experiments/mainline/run/core/run_prefix_reentry_probe.py --rows outputs/results/_reentry_livebench_local_v1/reentry_rows.jsonl --output-dir outputs/results/_reentry_livebench_local_v1/probe --local-model-map qwen=Qwen/Qwen3-14B
uv run python experiments/mainline/run/core/run_prefix_oracle_gain_mapping.py --help
uv run python experiments/mainline/analysis/trace/analyze_trace_taxonomy.py
```

## Mainline Re-entry Experiment

The current mainline experiment studies how local models behave when they are
asked to continue from their own partial reasoning prefixes.

Pipeline:

```text
prepare -> re-entry controls -> probe
```

- `prepare` generates full reasoning traces, extracts final answers, evaluates
  correctness, and segments each trace into prefix checkpoints.
- `re-entry controls` rebuild cumulative prefixes (`1`, `1+2`, `1+2+3`, ...)
  and asks the same model to continue. It records answer correctness and
  stability across exact, repeat, marker, and fenced prefix formats.
- `probe` runs transformer forward passes over completed re-entry rows and
  stores hidden-state / attention summaries for prefix trajectories.

Configured benchmark presets live in
`experiments/mainline/run/batch/reentry_presets.json` and currently cover:

```text
livebench_reasoning
olympiadbench
aime_2024
math500
gpqa_diamond
```

The expanded LiveBench local family matrix uses:

```text
qwen3      -> Qwen/Qwen3-14B
qwen25     -> Qwen/Qwen2.5-14B-Instruct
gemma4     -> google/gemma-4-E4B-it
gemma3     -> google/gemma-3-4b-it
ministral  -> mistralai/Ministral-8B-Instruct-2410
mistral7   -> mistralai/Mistral-7B-Instruct-v0.3
olmo       -> allenai/olmo-2-13b-instruct
```

### Output Layout

```text
outputs/results/reentry_livebench_<family>_30/
  *_small_only.json          # prepare output: full traces, segments, metadata
  summary.json               # prepare summary
  partials/                  # resumable per-task prepare state

outputs/results/_reentry_livebench_local_v1_<family>_no_outliers/
  reentry_rows.jsonl         # re-entry control rows
  reentry_summary.json       # re-entry aggregate metrics
  probe/
    reentry_probe_rows.jsonl # hidden-state / attention summaries
    reentry_probe_summary.json

outputs/logs/
  reentry_controls_*.log
  reentry_probe_*.log
```

Prepare results include robust segment-count outlier metadata:

```text
segment_count
segment_count_outlier
segment_count_outlier_stats
```

Use `--exclude-segment-count-outliers` during re-entry to skip pathological
traces that were split into unusually many segments.

### VM / tmux Basics

```bash
ssh george15672@35.184.74.240
cd ~/confidence-tom

tmux new -s reentry-no-outliers
tmux attach -t reentry-no-outliers
```

Detach without stopping the job:

```text
Ctrl-b
d
```

Stop a session:

```bash
tmux kill-session -t reentry-no-outliers
pkill -f "run_prefix_reentry_controls.py" || true
nvidia-smi
```

### Prepare

Dry-run the full orchestrator:

```bash
uv run python experiments/mainline/run/batch/run_reentry_mainline.py \
  --preset reentry_livebench_local \
  --phase all \
  --dry-run
```

Run prepare for one LiveBench family:

```bash
uv run python experiments/mainline/run/batch/run_prefix_family_sweep.py \
  --config-name prefix_family_sweep_reentry_livebench_ollama \
  '+launcher.only_small_families=[qwen3]'
```

Backfill segment-count outlier metadata into existing prepare outputs:

```bash
uv run python - <<'PY'
from pathlib import Path
from experiments.mainline.run.core.run_prefix_oracle_gain_mapping import (
    annotate_segment_count_outliers,
)

for path in Path("outputs/results").glob("reentry_livebench_*_30/*.json"):
    if path.name in {"summary.json", "dataset_meta.json", "_run_status.json"}:
        continue
    annotate_segment_count_outliers(path)
    print("annotated", path)
PY
```

### Re-entry Controls

Run no-outlier LiveBench re-entry controls for completed families:

```bash
cd ~/confidence-tom
mkdir -p outputs/logs

COMMON_MAPS=(
  --small-local-model-map qwen3=Qwen/Qwen3-14B
  --small-local-model-map qwen25=Qwen/Qwen2.5-14B-Instruct
  --small-local-model-map gemma4=google/gemma-4-E4B-it
  --small-local-model-map gemma3=google/gemma-3-4b-it
  --small-local-model-map ministral=mistralai/Ministral-8B-Instruct-2410
  --small-local-model-map mistral7=mistralai/Mistral-7B-Instruct-v0.3
  --small-local-model-map olmo=allenai/olmo-2-13b-instruct
)

run_reentry_family_no_outliers () {
  family="$1"
  run_name="reentry_livebench_${family}_30"
  out="outputs/results/_reentry_livebench_local_v1_${family}_no_outliers"
  log="outputs/logs/reentry_controls_${family}_no_outliers_$(date +%Y%m%d_%H%M%S).log"

  echo "===== START ${family} ${run_name} $(date -u) =====" | tee -a "$log"

  uv run python experiments/mainline/run/core/run_prefix_reentry_controls.py \
    --output-dir "$out" \
    --small-backend local \
    --concurrency 1 \
    --max-tokens 2048 \
    --full-rerun-temperature 0.0 \
    --reentry-temperature 0.0 \
    --run-name "$run_name" \
    --benchmark livebench_reasoning \
    --small-family "$family" \
    --exclude-segment-count-outliers \
    "${COMMON_MAPS[@]}" 2>&1 | tee -a "$log"

  status=${PIPESTATUS[0]}
  echo "===== END ${family} status=${status} $(date -u) =====" | tee -a "$log"
  return "$status"
}

for family in qwen3 qwen25 gemma4 ministral mistral7; do
  run_reentry_family_no_outliers "$family" || echo "Family $family failed; continuing."
done
```

Check re-entry progress:

```bash
for family in qwen3 qwen25 gemma4 ministral mistral7; do
  echo "===== $family ====="
  wc -l outputs/results/_reentry_livebench_local_v1_${family}_no_outliers/reentry_rows.jsonl 2>/dev/null || echo "no rows yet"
done
```

Watch the latest log:

```bash
tail -f "$(ls -t outputs/logs/reentry_controls_*_no_outliers_*.log | head -n 1)"
```

### Probe Hidden States

Run representation probes after re-entry rows exist:

```bash
cd ~/confidence-tom
mkdir -p outputs/logs

COMMON_PROBE_MAPS=(
  --local-model-map qwen3=Qwen/Qwen3-14B
  --local-model-map qwen25=Qwen/Qwen2.5-14B-Instruct
  --local-model-map gemma4=google/gemma-4-E4B-it
  --local-model-map gemma3=google/gemma-3-4b-it
  --local-model-map ministral=mistralai/Ministral-8B-Instruct-2410
  --local-model-map mistral7=mistralai/Mistral-7B-Instruct-v0.3
  --local-model-map olmo=allenai/olmo-2-13b-instruct
)

run_probe_family () {
  family="$1"
  rows="outputs/results/_reentry_livebench_local_v1_${family}_no_outliers/reentry_rows.jsonl"
  out="outputs/results/_reentry_livebench_local_v1_${family}_no_outliers/probe"
  log="outputs/logs/reentry_probe_${family}_$(date +%Y%m%d_%H%M%S).log"

  echo "===== START probe ${family} $(date -u) =====" | tee -a "$log"

  uv run python experiments/mainline/run/core/run_prefix_reentry_probe.py \
    --rows "$rows" \
    --output-dir "$out" \
    --backend transformers \
    --selected-layer -1 \
    "${COMMON_PROBE_MAPS[@]}" 2>&1 | tee -a "$log"

  status=${PIPESTATUS[0]}
  echo "===== END probe ${family} status=${status} $(date -u) =====" | tee -a "$log"
  return "$status"
}

for family in qwen3 qwen25 gemma4 ministral mistral7; do
  run_probe_family "$family" || echo "Probe $family failed; continuing."
done
```

Check probe progress:

```bash
for family in qwen3 qwen25 gemma4 ministral mistral7; do
  echo "===== $family ====="
  wc -l outputs/results/_reentry_livebench_local_v1_${family}_no_outliers/probe/reentry_probe_rows.jsonl 2>/dev/null || echo "no probe rows yet"
done
```

### Download VM Results

On the VM:

```bash
cd ~/confidence-tom
tar -czf /tmp/reentry_livebench_no_outliers_latest.tgz \
  outputs/results/reentry_livebench_* \
  outputs/results/_reentry_livebench_local_v1_*_no_outliers \
  outputs/logs
```

On local Windows PowerShell:

```powershell
cd C:\Users\admin\Desktop\Experiment\confidence-tom

gcloud compute scp --zone us-central1-c `
  george15672@instance-20260428-090929:/tmp/reentry_livebench_no_outliers_latest.tgz .

tar -xzf .\reentry_livebench_no_outliers_latest.tgz -C .
```

### Determinism audit
```bash
uv run python experiments/mainline/run/core/run_api_determinism_audit.py
```

### Remote / queue helpers
```bash
uv run python experiments/mainline/run/remote/run_remote_prefix_reentry_controls.py --mode status
uv run python experiments/mainline/run/remote/run_remote_ollama_livebench_ordered.py --mode status
```

## Notes

- `outputs/` 是唯一的生成物根目錄，`results/`、`logs/`、其他中間產物都會放在它底下，並已在 `.gitignore` 排除。
- `src/confidence_tom/compat/` 目前只保留 generator / observer 的相容層，不再當作對外實驗入口。
- `.env.example` 提供本地與遠端執行常用欄位；真實密鑰請放在 `.env`。
- `uv sync --all-groups` 會安裝主線與測試所需依賴。

## License

MIT
