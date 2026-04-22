from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
REMOTE_PIPELINE = ROOT / "experiments" / "legacy" / "run_remote_pipeline.py"
REMOTE_PROJECT_DIR = os.environ.get("REMOTE_PROJECT_DIR", "~/confidence-tom")
REMOTE_OLLAMA_BASE_URL = os.environ.get("REMOTE_OLLAMA_BASE_URL", "http://127.0.0.1:11435/v1")


class OrderedModelSpec(TypedDict):
    family: str
    ollama_model: str
    small_model: str
    small_label: str
    num_ctx: int
    num_predict: int
    output_prefix: str


DEFAULT_ORDER: list[OrderedModelSpec] = [
    {
        "family": "mistral",
        "ollama_model": "ministral-3:3b",
        "small_model": "ollama/ministral-3:3b",
        "small_label": "Ollama-ministral-3:3b",
        "num_ctx": 16384,
        "num_predict": 3072,
        "output_prefix": "ordered_ministral3",
    },
    {
        "family": "gemma",
        "ollama_model": "gemma4:e4b",
        "small_model": "ollama/gemma4:e4b",
        "small_label": "Ollama-gemma4:e4b",
        "num_ctx": 16384,
        "num_predict": 3072,
        "output_prefix": "ordered_gemma4",
    },
    {
        "family": "gemma",
        "ollama_model": "gemma3:4b",
        "small_model": "ollama/gemma3:4b",
        "small_label": "Ollama-gemma3:4b",
        "num_ctx": 16384,
        "num_predict": 3072,
        "output_prefix": "ordered_gemma3",
    },
    {
        "family": "qwen",
        "ollama_model": "qwen3.5:4b",
        "small_model": "ollama/qwen3.5:4b",
        "small_label": "Ollama-qwen3.5:4b",
        "num_ctx": 16384,
        "num_predict": 3072,
        "output_prefix": "ordered_qwen35_4b",
    },
]


def _run(cmd: list[str]) -> None:
    print("$", shlex.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _remote_path(remote_dir: str) -> str:
    if remote_dir == "~":
        return "$HOME"
    if remote_dir.startswith("~/"):
        return f"$HOME/{shlex.quote(remote_dir[2:])}"
    return shlex.quote(remote_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote launcher for ordered LiveBench Ollama sweeps on the company machine."
    )
    parser.add_argument("--mode", choices=["run", "start", "status", "tail"], default="start")
    parser.add_argument("--job-name", default="ordered_ollama_livebench")
    parser.add_argument("--remote-timeout", type=int, default=7200)
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--full-trace-sec", type=int, default=1800)
    parser.add_argument("--small-worker-sec", type=int, default=1800)
    parser.add_argument("--large-worker-sec", type=int, default=900)
    parser.add_argument("--task-sec", type=int, default=14400)
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--prep-max-attempts", type=int, default=12)
    parser.add_argument("--prep-sleep-sec", type=int, default=10)
    parser.add_argument("--healthcheck-max-attempts", type=int, default=3)
    parser.add_argument("--healthcheck-sleep-sec", type=int, default=20)
    parser.add_argument("--healthcheck-timeout-sec", type=int, default=30)
    args = parser.parse_args()

    ordered: list[OrderedModelSpec] = DEFAULT_ORDER
    if args.models:
        requested = set(args.models)
        ordered = [item for item in DEFAULT_ORDER if item["ollama_model"] in requested]
        missing = [
            name
            for name in args.models
            if name not in {item["ollama_model"] for item in DEFAULT_ORDER}
        ]
        if missing:
            raise SystemExit(f"Unknown models: {', '.join(missing)}")

    if args.mode == "status":
        _run(
            [
                "uv",
                "run",
                "python",
                str(REMOTE_PIPELINE),
                "--timeout",
                str(args.remote_timeout),
                "status",
                "--job-name",
                args.job_name,
            ]
        )
        return

    if args.mode == "tail":
        _run(
            [
                "uv",
                "run",
                "python",
                str(REMOTE_PIPELINE),
                "--timeout",
                str(args.remote_timeout),
                "tail",
                "--job-name",
                args.job_name,
                "--lines",
                str(args.tail_lines),
            ]
        )
        return

    if not args.skip_sync:
        _run(
            [
                "uv",
                "run",
                "python",
                str(REMOTE_PIPELINE),
                "--timeout",
                str(args.remote_timeout),
                "sync",
            ]
        )

    remote_lines = [
        f"cd {_remote_path(REMOTE_PROJECT_DIR)}",
        "export OLLAMA_HOST=http://127.0.0.1:11435",
        f"export OLLAMA_BASE_URL={shlex.quote(REMOTE_OLLAMA_BASE_URL)}",
        "set +e",
        "stop_ollama_models() {",
        "  attempts=0",
        f'  while [ "$attempts" -lt {int(args.prep_max_attempts)} ]; do',
        "    current_models=$(ollama ps | awk 'NR>1 && $1 != \"NAME\" {print $1}')",
        '    if [ -z "$current_models" ]; then',
        "      return 0",
        "    fi",
        '    echo "[prep] stopping: $current_models"',
        "    for model in $current_models; do",
        '      ollama stop "$model" >/dev/null 2>&1 || true',
        "    done",
        f"    sleep {int(args.prep_sleep_sec)}",
        "    attempts=$((attempts + 1))",
        "  done",
        '  echo "[prep] proceeding despite residual models"',
        "}",
        "healthcheck_model() {",
        '  model="$1"',
        '  timeout="$2"',
        '  python3 - "$OLLAMA_BASE_URL" "$model" "$timeout" <<\'PY\'',
        "import json",
        "import sys",
        "import urllib.error",
        "import urllib.request",
        "",
        "base = sys.argv[1].rstrip('/')",
        "model = sys.argv[2]",
        "timeout = int(sys.argv[3])",
        "",
        "payload = json.dumps({",
        '    "model": model,',
        '    "messages": [{"role": "user", "content": "ping"}],',
        '    "temperature": 0,',
        '    "max_tokens": 1,',
        '    "stream": False,',
        '}).encode("utf-8")',
        "",
        "req = urllib.request.Request(",
        '    f"{base}/chat/completions",',
        "    data=payload,",
        '    headers={"Content-Type": "application/json"},',
        '    method="POST",',
        ")",
        "try:",
        "    with urllib.request.urlopen(req, timeout=timeout) as resp:",
        "        _ = resp.read(1024)",
        "except Exception as exc:",
        '    print(f"[healthcheck] {model}: {exc}")',
        "    raise SystemExit(1)",
        "PY",
        "}",
        "ensure_model_ready() {",
        '  model="$1"',
        "  attempts=0",
        f'  while [ "$attempts" -lt {int(args.healthcheck_max_attempts)} ]; do',
        (
            f'    if healthcheck_model "$model" '
            f"{int(args.healthcheck_timeout_sec)} >/dev/null 2>&1; then"
        ),
        "      return 0",
        "    fi",
        "    attempts=$((attempts + 1))",
        '    echo "[prep] healthcheck failed for $model (attempt $attempts)"',
        f"    sleep {int(args.healthcheck_sleep_sec)}",
        "  done",
        '  echo "[skip] unhealthy model: $model"',
        "  return 1",
        "}",
        "run() {",
        "  echo \"[$(date '+%F %T')] START $*\"",
        '  "$@"',
        "  rc=$?",
        "  echo \"[$(date '+%F %T')] END rc=$rc $*\"",
        "  return 0",
        "}",
    ]

    for item in ordered:
        remote_lines.append("stop_ollama_models")
        remote_lines.append(f"if ! ensure_model_ready {shlex.quote(item['ollama_model'])}; then")
        remote_lines.append(f'  echo "[skip] skipping {item["ollama_model"]}"')
        remote_lines.append("else")
        cmd = [
            "uv",
            "run",
            "python",
            "experiments/mainline/run/core/run_ollama_prefix_sweep.py",
            "--family",
            item["family"],
            "--ollama-model",
            item["ollama_model"],
            "--benchmark",
            "livebench_reasoning",
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--task-concurrency",
            str(args.task_concurrency),
            "--num-ctx",
            str(item["num_ctx"]),
            "--num-predict",
            str(item["num_predict"]),
            "--enable-thinking",
            "false",
            "--output-prefix",
            item["output_prefix"],
            "--full-trace-sec",
            str(args.full_trace_sec),
            "--small-worker-sec",
            str(args.small_worker_sec),
            "--large-worker-sec",
            str(args.large_worker_sec),
            "--task-sec",
            str(args.task_sec),
            "--small-model",
            item["small_model"],
            "--small-label",
            item["small_label"],
        ]
        if args.run_analysis:
            cmd.append("--run-analysis")
        remote_lines.append(f"  run {shlex.join(cmd)}")
        remote_lines.append("fi")

    remote_cmd = [
        "bash",
        "-lc",
        "\n".join(remote_lines),
    ]
    subcommand = "run-bg" if args.mode == "start" else "run"
    cmd = [
        "uv",
        "run",
        "python",
        str(REMOTE_PIPELINE),
        "--timeout",
        str(args.remote_timeout),
        subcommand,
        "--skip-sync",
    ]
    if subcommand == "run-bg":
        cmd += ["--job-name", args.job_name]
    cmd += ["--", *remote_cmd]
    _run(cmd)


if __name__ == "__main__":
    main()
