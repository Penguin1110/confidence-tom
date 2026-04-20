from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_PIPELINE = ROOT / "experiments" / "run_remote_pipeline.py"


def _run(cmd: list[str]) -> None:
    print("$", shlex.join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _launch_remote(
    *,
    job_name: str,
    config_name: str,
    remote_timeout: int,
    mode: str,
    tail_lines: int,
    skip_sync: bool,
    limit_override: int | None,
    run_analysis: bool | None,
    extra_overrides: list[str],
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
            "cd /home/karl/confidence-tom && "
            f"uv run python experiments/run_prefix_family_sweep.py --config-name {shlex.quote(config_name)}"
        ),
    ]
    if limit_override is not None:
        remote_cmd[-1] += f" dataset.limit={int(limit_override)}"
    if run_analysis is not None:
        remote_cmd[-1] += f" launcher.run_analysis={'true' if run_analysis else 'false'}"
    for override in extra_overrides:
        remote_cmd[-1] += f" {shlex.quote(override)}"
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
        description="Launch remote Qwen-local family sweep jobs on the company machine."
    )
    parser.add_argument("--target", choices=["olympiad", "livebench", "both"], default="both")
    parser.add_argument("--mode", choices=["run", "start", "status", "tail"], default="start")
    parser.add_argument("--remote-timeout", type=int, default=7200)
    parser.add_argument("--tail-lines", type=int, default=120)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-analysis", choices=["true", "false"])
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    jobs: list[tuple[str, str]] = []
    if args.target in {"olympiad", "both"}:
        jobs.append(("qwen_local_olympiad", "prefix_family_sweep_qwen_local"))
    if args.target in {"livebench", "both"}:
        jobs.append(("qwen_local_livebench", "prefix_family_sweep_livebench_qwen_local"))

    first = True
    for job_name, config_name in jobs:
        _launch_remote(
            job_name=job_name,
            config_name=config_name,
            remote_timeout=args.remote_timeout,
            mode=args.mode,
            tail_lines=args.tail_lines,
            skip_sync=(args.skip_sync or not first),
            limit_override=args.limit,
            run_analysis=None if args.run_analysis is None else args.run_analysis == "true",
            extra_overrides=list(args.overrides),
        )
        first = False


if __name__ == "__main__":
    main()
