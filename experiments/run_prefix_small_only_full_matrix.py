from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_prefix_small_only_mapping.py"


# Tuned for L4 + Ollama-serving style usage.
MODEL_PRESETS: dict[str, dict[str, str | int]] = {
    "qwen3_8b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3:8b",
        "small_local_model_name": "qwen3:8b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 4096,
    },
    "qwen3_14b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3:14b",
        "small_local_model_name": "qwen3:14b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 4096,
    },
    "qwen3_32b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3:32b",
        "small_local_model_name": "qwen3:32b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 3072,
    },
    "qwen35_9b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3.5:9b",
        "small_local_model_name": "qwen3.5:9b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 4096,
    },
    "qwen35_27b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3.5:27b",
        "small_local_model_name": "qwen3.5:27b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 3072,
    },
    "gemma3_12b": {
        "small_model": "google/gemma-3-4b-it",
        "small_label": "Ollama-gemma3:12b",
        "small_local_model_name": "gemma3:12b",
        "small_max_tokens": 8192,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "gemma3_27b": {
        "small_model": "google/gemma-3-4b-it",
        "small_label": "Ollama-gemma3:27b",
        "small_local_model_name": "gemma3:27b",
        "small_max_tokens": 8192,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "mistral_small_24b": {
        "small_model": "mistralai/ministral-8b-2512",
        "small_label": "Ollama-mistral-small3.2:24b",
        "small_local_model_name": "mistral-small3.2:24b",
        "small_max_tokens": 12288,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "llama31_8b": {
        "small_model": "meta-llama/llama-4-scout",
        "small_label": "Ollama-llama3.1:8b",
        "small_local_model_name": "llama3.1:8b",
        "small_max_tokens": 12288,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "olmo31_32b": {
        "small_model": "allenai/olmo-2-13b-instruct",
        "small_label": "Ollama-olmo-3.1:32b",
        "small_local_model_name": "olmo-3.1:32b",
        "small_max_tokens": 12288,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
}

LOCAL_MODEL_PRESETS: dict[str, dict[str, str | int]] = {
    # HF/Transformers-friendly presets for A100 runtime.
    "qwen25_7b": {
        "small_model": "Qwen/Qwen2.5-7B-Instruct",
        "small_label": "HF-Qwen2.5-7B-Instruct",
        "small_local_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "small_max_tokens": 8192,
        "num_ctx": 16384,
        "num_predict": 1024,
    },
    "qwen25_14b": {
        "small_model": "Qwen/Qwen2.5-14B-Instruct",
        "small_label": "HF-Qwen2.5-14B-Instruct",
        "small_local_model_name": "Qwen/Qwen2.5-14B-Instruct",
        "small_max_tokens": 8192,
        "num_ctx": 16384,
        "num_predict": 1024,
    },
    "gemma2_9b": {
        "small_model": "google/gemma-2-9b-it",
        "small_label": "HF-gemma-2-9b-it",
        "small_local_model_name": "google/gemma-2-9b-it",
        "small_max_tokens": 6144,
        "num_ctx": 12288,
        "num_predict": 1024,
    },
    "mistral7b_v03": {
        "small_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "small_label": "HF-Mistral-7B-Instruct-v0.3",
        "small_local_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "small_max_tokens": 6144,
        "num_ctx": 12288,
        "num_predict": 1024,
    },
}


def _run(cmd: list[str]) -> None:
    print("$", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_optional(cmd: list[str]) -> int:
    print("$", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot queue runner for small-only matrix on Colab/L4."
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma list from presets, or 'all'.",
    )
    parser.add_argument(
        "--benchmarks",
        default="olympiadbench,livebench_reasoning",
        help="Comma list of benchmarks.",
    )
    parser.add_argument(
        "--olympiad-limit",
        type=int,
        default=0,
        help="0 means full split.",
    )
    parser.add_argument(
        "--livebench-limit",
        type=int,
        default=0,
        help="0 means full split.",
    )
    parser.add_argument("--output-prefix", default="colab_full_small")
    parser.add_argument(
        "--small-backend", choices=["ollama", "openrouter", "local"], default="ollama"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", choices=["true", "false"], default="false")
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--full-trace-sec", type=int, default=1200)
    parser.add_argument("--small-worker-sec", type=int, default=900)
    parser.add_argument("--task-sec", type=int, default=7200)
    parser.add_argument("--extractor-enabled", action="store_true")
    parser.add_argument("--extractor-model", default="openai/gpt-5.4")
    parser.add_argument(
        "--extractor-backend", choices=["openrouter", "ollama", "local"], default="openrouter"
    )
    parser.add_argument(
        "--pull-before-run",
        action="store_true",
        help="For Ollama backend: pull each model before running it.",
    )
    parser.add_argument(
        "--delete-after-run",
        action="store_true",
        help="For Ollama backend: remove each model after all benchmarks finish.",
    )
    args = parser.parse_args()

    preset_table = MODEL_PRESETS if args.small_backend != "local" else LOCAL_MODEL_PRESETS
    if args.models == "all":
        model_keys = list(preset_table.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in model_keys if m not in preset_table]
        if unknown:
            raise SystemExit(f"Unknown model preset(s): {unknown}")

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for b in benchmarks:
        if b not in {"olympiadbench", "livebench_reasoning"}:
            raise SystemExit(f"Unsupported benchmark: {b}")

    total = len(model_keys) * len(benchmarks)
    idx = 0
    for model_key in model_keys:
        preset = preset_table[model_key]
        local_tag = str(preset["small_local_model_name"])
        if args.small_backend == "ollama" and args.pull_before_run:
            _run(["ollama", "pull", local_tag])
        for benchmark in benchmarks:
            idx += 1
            limit = args.olympiad_limit if benchmark == "olympiadbench" else args.livebench_limit
            out_dir = (
                ROOT
                / "results"
                / f"{args.output_prefix}_{model_key}_{benchmark}_{'full' if limit == 0 else limit}"
            )
            print(
                f"\n[{idx}/{total}] model={model_key} benchmark={benchmark} limit={limit} out={out_dir}",
                flush=True,
            )

            cmd = [
                "uv",
                "run",
                "python",
                str(RUNNER),
                "--benchmark",
                benchmark,
                "--limit",
                str(limit),
                "--output-dir",
                str(out_dir),
                "--small-model",
                str(preset["small_model"]),
                "--small-label",
                str(preset["small_label"]),
                "--small-backend",
                args.small_backend,
                "--small-local-model-name",
                str(preset["small_local_model_name"]),
                "--small-max-tokens",
                str(preset["small_max_tokens"]),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--top-k",
                str(args.top_k),
                "--seed",
                str(args.seed),
                "--num-ctx",
                str(preset["num_ctx"]),
                "--num-predict",
                str(preset["num_predict"]),
                "--enable-thinking",
                args.enable_thinking,
                "--task-concurrency",
                str(args.task_concurrency),
                "--retry-attempts",
                str(args.retry_attempts),
                "--retry-backoff-sec",
                str(args.retry_backoff_sec),
                "--full-trace-sec",
                str(args.full_trace_sec),
                "--small-worker-sec",
                str(args.small_worker_sec),
                "--task-sec",
                str(args.task_sec),
            ]
            if args.extractor_enabled:
                cmd.extend(
                    [
                        "--extractor-enabled",
                        "--extractor-model",
                        args.extractor_model,
                        "--extractor-backend",
                        args.extractor_backend,
                    ]
                )
            _run(cmd)
        if args.small_backend == "ollama" and args.delete_after_run:
            # Best-effort cleanup for constrained disks (e.g. Colab ephemeral storage).
            _run_optional(["ollama", "rm", local_tag])


if __name__ == "__main__":
    main()
