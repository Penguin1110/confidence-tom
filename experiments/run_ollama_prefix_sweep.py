from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_prefix_oracle_gain_mapping.py"
ANALYZER = ROOT / "experiments" / "analyze_prefix_oracle_gain.py"

LARGE_WORKERS = [
    {
        "family": "openai",
        "model": "openai/gpt-5.4",
        "label": "GPT-5.4",
        "max_tokens": 16384,
    },
    {
        "family": "anthropic",
        "model": "anthropic/claude-opus-4.6",
        "label": "Claude-Opus-4.6",
        "max_tokens": 16384,
    },
]

SMALL_DEFAULTS = {
    "qwen": {
        "model": "qwen/qwen3-14b:nitro",
        "label": "Qwen-3-14B-Ollama",
        "max_tokens": 12288,
    },
    "mistral": {
        "model": "mistralai/ministral-8b-2512",
        "label": "Ministral-8B-Ollama",
        "max_tokens": 12288,
    },
    "llama": {
        "model": "meta-llama/llama-4-scout",
        "label": "Llama-Ollama",
        "max_tokens": 12288,
    },
    "gemma": {
        "model": "google/gemma-3-4b-it",
        "label": "Gemma-3-4B-Ollama",
        "max_tokens": 8192,
    },
}


def _sanitize_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")


def _default_small_label(ollama_model: str) -> str:
    # Use the actual Ollama tag in outputs so result filenames map to reality.
    return f"Ollama-{ollama_model}"


def _run(cmd: list[str]) -> None:
    print("$", shlex.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prefix takeover sweep with small model via Ollama and large model via OpenRouter."
    )
    parser.add_argument("--family", choices=sorted(SMALL_DEFAULTS.keys()), required=True)
    parser.add_argument(
        "--ollama-model", required=True, help="Exact Ollama model tag served on port 11435."
    )
    parser.add_argument(
        "--benchmark", choices=["olympiadbench", "livebench_reasoning"], required=True
    )
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--small-model")
    parser.add_argument("--small-label")
    parser.add_argument("--small-max-tokens", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--full-trace-sec", type=int, default=1800)
    parser.add_argument("--small-worker-sec", type=int, default=1800)
    parser.add_argument("--large-worker-sec", type=int, default=900)
    parser.add_argument("--task-sec", type=int, default=14400)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--output-prefix", default="ollama")
    args = parser.parse_args()

    small_defaults = SMALL_DEFAULTS[args.family]
    small_model = args.small_model or small_defaults["model"]
    small_label = args.small_label or _default_small_label(args.ollama_model)
    small_max_tokens = args.small_max_tokens or int(small_defaults["max_tokens"])

    for large in LARGE_WORKERS:
        run_name = f"{args.output_prefix}_{args.family}_to_{large['family']}_{args.limit}"
        if args.benchmark == "livebench_reasoning":
            run_name = (
                f"{args.output_prefix}_livebench_{args.family}_to_{large['family']}_{args.limit}"
            )
        output_dir = ROOT / "results" / run_name
        result_path = (
            output_dir / f"{_sanitize_label(small_label)}_to_{_sanitize_label(large['label'])}.json"
        )
        if result_path.exists() and not args.overwrite_output_dir:
            print(f"[skip] final result already exists: {result_path}")
            continue

        cmd = [
            "uv",
            "run",
            "python",
            str(RUNNER),
            f"output_dir={output_dir}",
            f"dataset.benchmark={args.benchmark}",
            f"dataset.limit={args.limit}",
            f"execution.task_concurrency={int(args.task_concurrency)}",
            f"execution.retry_attempts={args.retry_attempts}",
            f"execution.retry_backoff_sec={args.retry_backoff_sec}",
            "execution.resume_from_partials=true",
            "execution.retain_partials=true",
            f"timeouts.full_trace_sec={args.full_trace_sec}",
            f"timeouts.small_worker_sec={args.small_worker_sec}",
            f"timeouts.large_worker_sec={args.large_worker_sec}",
            f"timeouts.task_sec={args.task_sec}",
            f"small_worker.model={small_model}",
            f"small_worker.label={small_label}",
            f"small_worker.max_tokens={small_max_tokens}",
            "+small_worker.backend=ollama",
            f"+small_worker.local_model_name={args.ollama_model}",
            f"large_worker.model={large['model']}",
            f"large_worker.label={large['label']}",
            f"large_worker.max_tokens={int(large['max_tokens'])}",
        ]
        if args.seed is not None:
            cmd.append(f"small_worker.seed={args.seed}")
        if args.top_p is not None:
            cmd.append(f"small_worker.top_p={args.top_p}")
        if args.top_k is not None:
            cmd.append(f"small_worker.top_k={args.top_k}")
        if args.num_ctx is not None:
            cmd.append(f"small_worker.num_ctx={args.num_ctx}")
        if args.num_predict is not None:
            cmd.append(f"small_worker.num_predict={args.num_predict}")
        if args.enable_thinking is not None:
            cmd.append(f"small_worker.enable_thinking={args.enable_thinking == 'true'}")
        if args.benchmark == "olympiadbench":
            cmd.append(f"dataset.olympiadbench={args.limit}")
        else:
            cmd.append(f"dataset.livebench={args.limit}")
            cmd.append(f"dataset.livebench_reasoning={args.limit}")
        _run(cmd)

        if args.run_analysis:
            analysis_cmd = [
                "uv",
                "run",
                "python",
                str(ANALYZER),
                f"output_dir={output_dir}",
                f"dataset.benchmark={args.benchmark}",
                f"analysis.summary_json={output_dir / 'summary.json'}",
                f"analysis.per_prefix_rows_csv={output_dir / 'per_prefix_rows.csv'}",
            ]
            if args.benchmark == "olympiadbench":
                analysis_cmd.append(f"dataset.olympiadbench={args.limit}")
            else:
                analysis_cmd.append(f"dataset.livebench_reasoning={args.limit}")
            _run(analysis_cmd)


if __name__ == "__main__":
    main()
