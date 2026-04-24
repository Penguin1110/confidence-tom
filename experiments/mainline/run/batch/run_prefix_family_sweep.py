from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def _sanitize_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")


_OPTIONAL_WORKER_KEYS = [
    "backend",
    "local_model_name",
    "reasoning_effort",
    "top_p",
    "top_k",
    "seed",
    "num_ctx",
    "num_predict",
    "enable_thinking",
]


def _append_worker_overrides(cmd: list[str], prefix: str, worker: DictConfig) -> None:
    for key in _OPTIONAL_WORKER_KEYS:
        value = worker.get(key)
        if value is None:
            continue
        if key in {"backend", "local_model_name"}:
            cmd.append(f"+{prefix}.{key}={value}")
        else:
            cmd.append(f"{prefix}.{key}={value}")


def _dataset_override_args(benchmark: str, limit: int) -> list[str]:
    if benchmark == "olympiadbench":
        return [f"dataset.olympiadbench={limit}"]
    if benchmark == "livebench_reasoning":
        return [f"dataset.livebench={limit}", f"dataset.livebench_reasoning={limit}"]
    if benchmark in {"aime_2024", "math500", "gpqa_diamond"}:
        return [f"dataset.{benchmark}={limit}"]
    raise ValueError(f"Unsupported benchmark for family sweep: {benchmark}")


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="prefix_family_sweep",
)
def main(cfg: DictConfig) -> None:
    root = Path(to_absolute_path("."))
    runner = (
        root / "experiments" / "mainline" / "run" / "core" / "run_prefix_oracle_gain_mapping.py"
    )
    analyzer = (
        root / "experiments" / "mainline" / "analysis" / "prefix" / "analyze_prefix_oracle_gain.py"
    )

    benchmark = str(cfg.dataset.benchmark)
    limit = int(cfg.dataset.limit)
    task_concurrency = int(cfg.execution.task_concurrency)
    retry_attempts = int(cfg.execution.retry_attempts)
    retry_backoff_sec = float(cfg.execution.retry_backoff_sec)
    run_analysis = bool(cfg.launcher.get("run_analysis", True))
    run_name_prefix = str(cfg.launcher.get("run_name_prefix", ""))
    continue_on_error = bool(cfg.launcher.get("continue_on_error", False))
    timeouts = cfg.get("timeouts")

    for small in cfg.small_workers:
        for large in cfg.large_workers:
            run_name = f"{run_name_prefix}{small.family}_to_{large.family}_{limit}"
            output_dir = root / "outputs" / "results" / run_name
            result_path = (
                output_dir
                / f"{_sanitize_label(str(small.label))}_to_{_sanitize_label(str(large.label))}.json"
            )
            if result_path.exists() and not bool(cfg.launcher.get("overwrite_output_dir", False)):
                print(f"[skip] {run_name} final result already exists: {result_path}")
                continue

            cmd = [
                "uv",
                "run",
                "python",
                str(runner),
                f"output_dir={output_dir}",
                f"dataset.benchmark={benchmark}",
                f"dataset.limit={limit}",
                f"execution.task_concurrency={task_concurrency}",
                f"execution.retry_attempts={retry_attempts}",
                f"execution.retry_backoff_sec={retry_backoff_sec}",
                f"small_worker.model={small.model}",
                f"small_worker.label={small.label}",
                f"small_worker.max_tokens={int(small.max_tokens)}",
                f"large_worker.model={large.model}",
                f"large_worker.label={large.label}",
                f"large_worker.max_tokens={int(large.max_tokens)}",
            ]
            if timeouts is not None:
                for key, value in timeouts.items():
                    cmd.append(f"timeouts.{key}={value}")
            cmd.extend(_dataset_override_args(benchmark, limit))
            _append_worker_overrides(cmd, "small_worker", small)
            _append_worker_overrides(cmd, "large_worker", large)
            print("\n[run]")
            print(" ".join(shlex.quote(part) for part in cmd))
            try:
                subprocess.run(cmd, cwd=root, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[error] run failed: {run_name} returncode={exc.returncode}")
                if continue_on_error:
                    continue
                raise

            if run_analysis:
                analysis_cmd = [
                    "uv",
                    "run",
                    "python",
                    str(analyzer),
                    f"output_dir={output_dir}",
                    f"dataset.benchmark={benchmark}",
                    f"analysis.summary_json={output_dir / 'summary.json'}",
                    f"analysis.per_prefix_rows_csv={output_dir / 'per_prefix_rows.csv'}",
                ]
                analysis_cmd.extend(_dataset_override_args(benchmark, limit))
                print("\n[analyze]")
                print(" ".join(shlex.quote(part) for part in analysis_cmd))
                try:
                    subprocess.run(analysis_cmd, cwd=root, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"[error] analyze failed: {run_name} returncode={exc.returncode}")
                    if not continue_on_error:
                        raise


if __name__ == "__main__":
    main()
