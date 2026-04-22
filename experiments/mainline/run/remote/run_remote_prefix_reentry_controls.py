from __future__ import annotations

import argparse
import os
import shlex
import subprocess

from confidence_tom.infra.paths import project_root

ROOT = project_root()
REMOTE_PIPELINE = ROOT / "experiments" / "mainline" / "run" / "remote" / "run_remote_pipeline.py"
REMOTE_PROJECT_DIR = os.environ.get("REMOTE_PROJECT_DIR", "~/confidence-tom")


def _remote_path(remote_dir: str) -> str:
    if remote_dir == "~":
        return "$HOME"
    if remote_dir.startswith("~/"):
        return f"$HOME/{shlex.quote(remote_dir[2:])}"
    return shlex.quote(remote_dir)


def _run(cmd: list[str]) -> None:
    print("$", shlex.join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync repo and run prefix re-entry controls on the remote machine."
    )
    parser.add_argument("--mode", choices=["run", "start", "status", "tail"], default="run")
    parser.add_argument("--job-name", default="prefix_reentry_controls")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--full-rerun-temperature", type=float, default=0.0)
    parser.add_argument("--reentry-temperature", type=float, default=0.0)
    parser.add_argument(
        "--small-backend", choices=["openrouter", "ollama", "local"], default="openrouter"
    )
    parser.add_argument("--small-local-model-name", default=None)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--remote-timeout", type=int, default=7200)
    parser.add_argument("--tail-lines", type=int, default=60)
    args = parser.parse_args()

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

    remote_cmd = [
        "bash",
        "-lc",
        (
            f"cd {_remote_path(REMOTE_PROJECT_DIR)} && uv run python "
            "experiments/mainline/run/core/run_prefix_reentry_controls.py "
        )
        + (
            (" ".join(f"--run-name {shlex.quote(name)}" for name in args.run_name) + " ")
            if args.run_name
            else ""
        )
        + (
            (" ".join(f"--category {shlex.quote(cat)}" for cat in args.category) + " ")
            if args.category
            else ""
        )
        + (f"--max-rows {args.max_rows} " if args.max_rows is not None else "")
        + f"--concurrency {args.concurrency} "
        + f"--max-tokens {args.max_tokens} "
        + f"--full-rerun-temperature {args.full_rerun_temperature} "
        + f"--reentry-temperature {args.reentry_temperature} "
        + f"--small-backend {args.small_backend} "
        + (
            f"--small-local-model-name {shlex.quote(args.small_local_model_name)} "
            if args.small_local_model_name
            else ""
        ),
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
