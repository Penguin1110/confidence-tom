from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_PIPELINE = ROOT / "experiments" / "legacy" / "run_remote_pipeline.py"
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


def _remote_subprocess(
    *,
    job_name: str,
    remote_timeout: int,
    mode: str,
    tail_lines: int,
    config_name: str,
    run_name_prefix: str,
    skip_sync: bool,
) -> None:
    if mode == "status":
        _run(
            [
                "uv",
                "run",
                "python",
                str(REMOTE_PIPELINE),
                "--timeout",
                str(remote_timeout),
                "status",
                "--job-name",
                job_name,
            ]
        )
        return

    if mode == "tail":
        _run(
            [
                "uv",
                "run",
                "python",
                str(REMOTE_PIPELINE),
                "--timeout",
                str(remote_timeout),
                "tail",
                "--job-name",
                job_name,
                "--lines",
                str(tail_lines),
            ]
        )
        return

    if not skip_sync:
        _run(
            ["uv", "run", "python", str(REMOTE_PIPELINE), "--timeout", str(remote_timeout), "sync"]
        )

    remote_cmd = [
        "bash",
        "-lc",
        (
            f"cd {_remote_path(REMOTE_PROJECT_DIR)} && "
            "uv run python experiments/mainline/run/batch/run_prefix_family_sweep.py "
            f"--config-name {shlex.quote(config_name)} "
            f"launcher.run_name_prefix={shlex.quote(run_name_prefix)} "
            "launcher.overwrite_output_dir=false"
        ),
    ]
    subcommand = "run-bg" if mode == "start" else "run"
    cmd = [
        "uv",
        "run",
        "python",
        str(REMOTE_PIPELINE),
        "--timeout",
        str(remote_timeout),
        subcommand,
        "--skip-sync",
    ]
    if subcommand == "run-bg":
        cmd += ["--job-name", job_name]
    cmd += ["--", *remote_cmd]
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote-only rerun launcher for prefix family sweeps on the company machine."
    )
    parser.add_argument("--target", choices=["olympiad", "livebench", "both"], default="both")
    parser.add_argument("--mode", choices=["run", "start", "status", "tail"], default="start")
    parser.add_argument("--remote-timeout", type=int, default=7200)
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--run-prefix-base", default="dgx_rerun_")
    args = parser.parse_args()

    jobs: list[tuple[str, str, str]] = []
    if args.target in {"olympiad", "both"}:
        jobs.append(
            (
                "prefix_family_sweep_olympiad_dgx_rerun",
                "prefix_family_sweep",
                args.run_prefix_base,
            )
        )
    if args.target in {"livebench", "both"}:
        jobs.append(
            (
                "prefix_family_sweep_livebench_dgx_rerun",
                "prefix_family_sweep_livebench",
                f"{args.run_prefix_base}livebench_",
            )
        )

    first = True
    for job_name, config_name, run_name_prefix in jobs:
        _remote_subprocess(
            job_name=job_name,
            remote_timeout=args.remote_timeout,
            mode=args.mode,
            tail_lines=args.tail_lines,
            config_name=config_name,
            run_name_prefix=run_name_prefix,
            skip_sync=(args.skip_sync or not first),
        )
        first = False


if __name__ == "__main__":
    main()
