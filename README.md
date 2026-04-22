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
uv run python experiments/mainline/run/core/run_prefix_reentry_controls.py --category fragile-success --small-backend ollama
uv run python experiments/mainline/run/core/run_prefix_oracle_gain_mapping.py --help
uv run python experiments/mainline/analysis/trace/analyze_trace_taxonomy.py
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
